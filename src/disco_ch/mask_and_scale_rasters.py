"""
mask_and_scale_rasters.py

Masks every .tif in a folder using a binary mask raster (keep where mask == 1),
reproject-matching everything onto the mask's grid in the Swiss coordinate
system (EPSG:2056, "CH1903+ / LV95"), then scales and writes each result out
as a compact integer raster.

Pipeline per input raster:
    1. Reproject the mask (nearest-neighbor, since it's binary/categorical)
       to the target CRS once, up front.
    2. Reproject-match each data raster onto that mask grid (same CRS,
       resolution, transform, and extent) using a resampling method
       appropriate for continuous data (bilinear by default).
    3. Set every pixel where the mask isn't exactly 1 to NoData.
    4. Scale to an integer dtype (largest power-of-10 factor that preserves
       up to 2 decimals without overflow) and write to disk with LZW
       compression.

If mask_path is None, step 3 (masking) is skipped, but each raster is still
reprojected to target_crs (on its own native resolution/extent, since there's
no mask grid to match onto) before being scaled and saved.

Requires: rioxarray, rasterio, xarray, numpy
"""

import os
import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling

# Swiss coordinate system (new): CH1903+ / LV95
TARGET_CRS = "EPSG:2056"


# --------------------------------------------------------------------------
# Scaling (as provided, refactored into a DataArray-based core so it can be
# reused both for files on disk and for rasters we've already masked in
# memory, without an extra round-trip to disk in between).
# --------------------------------------------------------------------------

def scale_and_save_as_int_from_da(da, output_path, dtype="int16"):
    """
    Takes an in-memory rioxarray DataArray (nodata already represented as
    NaN, e.g. from rxr.open_rasterio(masked=True) or a `.where()` mask),
    scales it by the largest power-of-10 factor (10000, 1000, 100, 10, or 1)
    that preserves up to ~4 decimal points without overflowing the target
    int dtype, then writes it out as that dtype.

    A new nodata value (the dtype's min value) is set for the output, since
    the original nodata isn't representable in an int type. The applied
    scale_factor is stored in the output's metadata so values can be
    recovered later (true_value = stored_value / scale_factor... note the
    attribute itself is stored as 1/scale_factor per GDAL convention, see
    below).
    """
    dtype_info = np.iinfo(dtype)
    int_min, int_max = dtype_info.min, dtype_info.max
    valid = da.values[~np.isnan(da.values)]
    if valid.size == 0:
        raise ValueError(f"No valid data found in raster for {output_path}")
    max_abs = np.nanmax(np.abs(valid))
    # Pick the largest scale factor that keeps the scaled max within range,
    # leaving int_min reserved as the nodata sentinel.
    scale_factor = 1
    for candidate in (10000, 1000, 100, 10, 1):
        if max_abs * candidate <= int_max:
            scale_factor = candidate
            break
    nodata_value = int_min
    scaled = (da * scale_factor).round()
    scaled = scaled.fillna(nodata_value).astype(dtype)
    scaled.rio.write_nodata(nodata_value, inplace=True)
    scaled.attrs["scale_factor"] = 1.0 / scale_factor  # GDAL convention: true = stored * scale_factor
    scaled.rio.to_raster(
        output_path,
        compress="LZW",
        predictor=2,  # horizontal differencing predictor: improves LZW ratio for integer data
        tiled=True,  # tiling generally improves compression + access performance
    )
    return output_path, scale_factor, nodata_value


def scale_and_save_as_int(input_path, output_path, dtype="int16"):
    """
    Reads a TIF with rioxarray (nodata -> NaN) and delegates to
    scale_and_save_as_int_from_da. Kept for standalone use / parity with
    the original function signature.
    """
    da = rxr.open_rasterio(input_path, masked=True)
    return scale_and_save_as_int_from_da(da, output_path, dtype=dtype)


# --------------------------------------------------------------------------
# Reprojection + masking
# --------------------------------------------------------------------------

def load_mask_in_target_crs(mask_path, target_crs=TARGET_CRS):
    """
    Opens the binary mask raster and reprojects it to the target CRS using
    nearest-neighbor resampling (appropriate for categorical/binary data,
    so we never invent fractional mask values at the edges).
    """
    mask_da = rxr.open_rasterio(mask_path, masked=True)
    if mask_da.rio.crs is None:
        raise ValueError(f"Mask raster '{mask_path}' has no CRS defined.")
    if mask_da.rio.crs.to_string() != target_crs:
        mask_da = mask_da.rio.reproject(target_crs, resampling=Resampling.nearest)
    # Squeeze band dim if it's a single-band mask, so `.where` comparisons
    # broadcast cleanly against multi-band data rasters too.
    if "band" in mask_da.dims and mask_da.sizes.get("band", 1) == 1:
        mask_da = mask_da.squeeze("band", drop=True)
    return mask_da


def reproject_to_crs(da, target_crs=TARGET_CRS, resampling=Resampling.bilinear):
    """
    Reprojects `da` to target_crs on its own native resolution/extent (no
    mask grid to match onto). Used when no mask is supplied.
    """
    if da.rio.crs is not None and da.rio.crs.to_string() == target_crs:
        return da
    return da.rio.reproject(target_crs, resampling=resampling)


def mask_and_reproject_match(da, mask_da, resampling=Resampling.bilinear):
    """
    Reproject-matches `da` onto `mask_da`'s grid (CRS, resolution, transform,
    extent), then masks out every pixel where mask_da != 1.
    """
    matched = da.rio.reproject_match(mask_da, resampling=resampling)
    # reproject_match can leave coordinate float noise that makes xarray
    # think the grids don't line up exactly; align on mask_da's coords.
    matched = matched.assign_coords({
        "x": mask_da["x"],
        "y": mask_da["y"],
    })

    mask_values = mask_da.values
    if "band" in matched.dims and "band" not in mask_da.dims:
        # Broadcast a single 2D mask against a multi-band data raster.
        keep = mask_values == 1
        masked = matched.where(xr.DataArray(keep, dims=("y", "x"),
                                             coords={"y": mask_da["y"], "x": mask_da["x"]}))
    else:
        masked = matched.where(mask_da == 1)

    # Preserve CRS/spatial metadata after `.where` (xarray sometimes drops
    # the rioxarray-managed spatial_ref accessor state).
    masked.rio.write_crs(mask_da.rio.crs, inplace=True)
    return masked


# --------------------------------------------------------------------------
# Folder driver
# --------------------------------------------------------------------------

def mask_and_scale_directory(
    input_dir,
    mask_path,
    output_dir,
    dtype="int16",
    target_crs=TARGET_CRS,
    resampling=Resampling.bilinear,
    overwrite=False,
):
    """
    For every .tif/.tiff in input_dir:
      1. Reproject-match it onto the mask's grid in `target_crs`.
      2. Mask out pixels where the mask != 1.
      3. Scale to an integer dtype and write to output_dir under the same
         filename.

    Parameters
    ----------
    input_dir : str
        Folder containing the .tif files to process.
    mask_path : str or None
        Path to the binary mask raster (1 = keep, anything else = discard).
        If None, masking is skipped, but each raster is still reprojected
        to target_crs (on its own native resolution/extent) before scaling.
    output_dir : str
        Folder to write the masked + scaled outputs to.
    dtype : str
        Target integer dtype for the scaled output (e.g. "int16", "int32").
    target_crs : str
        Output CRS, defaults to EPSG:2056 (Swiss CH1903+ / LV95).
    resampling : rasterio.enums.Resampling
        Resampling method used when reproject-matching the *data* rasters
        (the mask itself always uses nearest-neighbor). Use
        Resampling.nearest instead if your data is categorical.
    overwrite : bool
        If False (default), files already present in output_dir are skipped
        instead of being reprocessed.

    Returns
    -------
    list of (output_path, scale_factor, nodata_value) tuples, one per file.
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_da = None
    if mask_path is not None:
        print(f"Loading mask '{mask_path}' and reprojecting to {target_crs} ...")
        mask_da = load_mask_in_target_crs(mask_path, target_crs=target_crs)
    else:
        print(f"No mask provided -- reprojecting to {target_crs} without masking.")

    tif_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))
    )
    if not tif_files:
        print(f"No .tif/.tiff files found in '{input_dir}'.")
        return []

    results = []
    for fname in tif_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        if not overwrite and os.path.exists(out_path):
            print(f"Skipping {fname} (already exists at {out_path}).")
            continue

        print(f"Processing {fname} ...")

        da = rxr.open_rasterio(in_path, masked=True)
        if mask_da is not None:
            processed = mask_and_reproject_match(da, mask_da, resampling=resampling)
        else:
            processed = reproject_to_crs(da, target_crs=target_crs, resampling=resampling)
        result = scale_and_save_as_int_from_da(processed, out_path, dtype=dtype)
        results.append(result)
        print(f"  -> wrote {out_path} (scale_factor={result[1]}, nodata={result[2]})")

    return results


if __name__ == "__main__":
    # Edit these paths and call the function directly - no CLI args needed.
    # input_dir = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\output\2026\mosaic"
    input_dir = r"F:\cb_overflow\2026_09_02_FORCE\FORCE\DiscoCH_rbf_10_20_30\output\level5norm\2026\mosaic\CCI"
    mask_path = None # r"C:\Users\bloomcol\Desktop\DiscoCH_Mask_2026_2625EVIdiff1500_CCImingt1000_MogliBeech_swissEOwald_wm75.tif"
    output_dir = r"F:\cb_overflow\2026_09_01_FORCE_Backup\FORCE\output\2026\CCI_2026_08_14_RBF_10_20_30_unclipped"

    mask_and_scale_directory(
        input_dir=input_dir,
        mask_path=mask_path,
        output_dir=output_dir,
        dtype="int16",              # or "int32" if int16's range is too tight
        target_crs=TARGET_CRS,      # EPSG:2056, Swiss CH1903+ / LV95
        resampling=Resampling.bilinear,  # use Resampling.nearest for categorical data
    )
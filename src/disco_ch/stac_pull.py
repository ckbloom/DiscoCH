import os
import gc
import json
import time
import pystac_client
from datetime import datetime, UTC
import xarray as xr
import rioxarray as rxr
import rasterio
import warnings
import numpy as np
import shutil
import dask
import matplotlib.pyplot as plt
import glob
import math
from matplotlib.colors import LinearSegmentedColormap
from rioxarray.exceptions import NoDataInBounds
from disco_ch.apply_model_stac import apply_disco

# Suppress specific warning
warnings.filterwarnings(
    "ignore",
    message="angle from rectified to skew grid parameter lost in conversion to CF",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning
)

# GDAL/VSI tuning for network-bound COG reads. Memory is no longer the
# constraint (36GB used of 128GB), so we can afford a much bigger block
# cache and larger local read-ahead cache than the earlier 256MB cap.
os.environ.setdefault("GDAL_CACHEMAX", "2048")
# Avoid extra HTTP calls (directory listings) that some VSI curl paths issue
# when opening a file, on top of the actual data reads.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
# Let GDAL reuse/multiplex HTTP connections instead of opening a fresh one
# per range request, and cache more of what it reads locally in-process.
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", str(512 * 1024 * 1024))  # 512MB/file
os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE", "YES")

# Chunk size for lazy raster reads. Bigger chunks = fewer, larger HTTP range
# requests, which is more network-efficient than many small ones -- worth
# doing now that RAM headroom allows it. Tune this to your machine/network;
# if you start seeing memory pressure again, drop it back down.
RASTER_CHUNK_SIZE = {"x": 2048, "y": 2048}

# dask.compute() with scheduler='threads' defaults its pool to os.cpu_count().
# That's the right size for CPU-bound work, but the ~232s "Executed all
# tasks" step above is dominated by network wait (HTTP range requests to
# the STAC hrefs), where threads spend most of their time blocked on I/O,
# not burning CPU. A larger pool lets more requests be in flight
# concurrently. Tune based on observed speedup / server-side rate limits --
# there's a point of diminishing (or negative) returns if the remote
# server throttles concurrent connections.
IO_BOUND_NUM_WORKERS = max(16, (os.cpu_count() or 4) * 4)


def _grids_match(da_a, da_b, atol=1e-6):
    """
    Returns True if two rioxarray DataArrays are already on the same grid
    (CRS, shape, and affine transform), meaning reproject_match would be a
    no-op warp. Use this to skip reproject_match calls entirely when
    they're not actually needed -- on a fixed national grid (as SwissEO
    products typically are), this is true for most/all dates and turns an
    expensive warp into a cheap comparison.
    """
    try:
        if da_a.rio.crs != da_b.rio.crs:
            return False
        if tuple(da_a.shape[-2:]) != tuple(da_b.shape[-2:]):
            return False
        ta = da_a.rio.transform()
        tb = da_b.rio.transform()
        return all(abs(a - b) <= atol for a, b in zip(ta, tb))
    except Exception:
        # If anything about this check fails (e.g. missing CRS), don't
        # block the pipeline -- just fall back to reproject_match.
        return False


def _reproject_match_if_needed(da, match_da, **reproject_kwargs):
    """
    Only calls .rio.reproject_match() if da isn't already on match_da's
    grid. Skips a full GDAL warp for the common case where the source is
    already on the target's fixed national grid.
    """
    if _grids_match(da, match_da):
        return da
    return da.rio.reproject_match(match_da, **reproject_kwargs)



# Vegetation Indices - https://force-eo.readthedocs.io/en/latest/components/higher-level/tsa/indices.html#indices
def ndv(nir, red):
    return (nir - red) / (nir + red)  # nir: B08, red: B04


def ndm(nir, swir1):
    return (nir - swir1) / (nir + swir1)  # nir: B08, swir1: B11


def evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))  # nir: B08, red: B04, blue: B02


def cire(rededge3, rededge1):
    return (rededge3 / rededge1) - 1  # rededge3: B07, rededge1: B05


def cci(green, red):
    return (green - red) / (green + red)  # green: B03, red: B04


def compute_vis(bands):
    """Computes the five canonical VIs from a bands dict (see load_and_process_assets())."""
    return {
        "NDVI": ndv(bands["nir"], bands["red"]),
        "EVI": evi(bands["nir"], bands["red"], bands["blue"]),
        "NDMI": ndm(bands["nir"], bands["swir"]),
        "CIRE": cire(bands["rededge3"], bands["rededge1"]),
        "CCI": cci(bands["green"], bands["red"]),
    }


def print_metadata(item):
    """
    Prints asset keys and band info directly from the STAC Item in v200.
    """
    print(f"--- STAC Item Assets for {item.id} ---")

    bands_10m = []
    bands_20m = []
    masks = []

    for key, asset in item.assets.items():
        if key.endswith("_10m.tif"):
            if "mask" in key:
                masks.append(key)
            else:
                bands_10m.append(key)
        elif key.endswith("_20m.tif"):
            bands_20m.append(key)

    print("10m Bands/Assets:")
    for b in sorted(bands_10m):
        print(f"  - {b}")

    print("\n20m Bands/Assets:")
    for b in sorted(bands_20m):
        print(f"  - {b}")

    print("\nMasks (10m):")
    for m in sorted(masks):
        print(f"  - {m}")


def pull_from_stac(stac_loc='https://data.geo.admin.ch/api/stac/v0.9/', year=2018, end_date=None, start_date=None):
    """
    Connect to the swisstopo stac and collect datasets
    """
    service = pystac_client.Client.open(stac_loc)
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    if end_date is None:
        end_date = f'{year}-09-30'
    if start_date is None:
        start_date = f'{year}-03-01'

    item_search = service.search(collections=['ch.swisstopo.swisseo_s2-sr_v200'], datetime=f'{start_date}/{end_date}')

    # Filter out items that do not match the expected datetime ID format
    valid_items = []
    for item in item_search.items():
        try:
            # We attempt to parse the date just to check if the format is correct
            datetime.strptime(item.id, "%Y-%m-%dt%H%M%S")
            valid_items.append(item)
        except ValueError:
            # If it fails, print a message and skip it
            print(f"Skipping unexpected item ID: {item.id}")
            continue

    print(f"Found {len(valid_items)} valid images in {year}")

    # Sort only the items we know have valid IDs
    item_list = sorted(valid_items, key=lambda x: datetime.strptime(x.id, "%Y-%m-%dt%H%M%S"))

    return item_list


def load_minmax_metadata(year, folder="minmax"):
    path = os.path.join(folder, f"vi_minmax_{year}_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_minmax_metadata(year, processed_dates, folder="minmax"):
    os.makedirs(folder, exist_ok=True)

    metadata = {
        "year": year,
        "processed_dates": processed_dates,
        "last_updated": datetime.now(UTC).isoformat()
    }

    path = os.path.join(folder, f"vi_minmax_{year}_meta.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_minmax_rasters(vi_min, vi_max, year, folder="minmax"):
    os.makedirs(folder, exist_ok=True)

    for k in vi_min:
        min_temp = f"{folder}/{k}_min_{year}_temp.tif"
        max_temp = f"{folder}/{k}_max_{year}_temp.tif"

        if os.path.exists(min_temp):
            os.remove(min_temp)
        if os.path.exists(max_temp):
            os.remove(max_temp)

        vi_min[k].rio.to_raster(min_temp)
        vi_max[k].rio.to_raster(max_temp)

        min_final = f"{folder}/{k}_min_{year}.tif"
        max_final = f"{folder}/{k}_max_{year}.tif"

        shutil.move(min_temp, min_final)
        shutil.move(max_temp, max_final)


def load_minmax_rasters(year, folder="minmax", chunks='default'):
    if folder is None:
        return None, None
    else:
        vi_min = {}
        vi_max = {}

        if chunks == 'default':
            chunks = {'x': 1024, 'y': 1024}

        for vi in ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]:
            min_path = f"{folder}/{vi}_min_{year}.tif"
            max_path = f"{folder}/{vi}_max_{year}.tif"

            if os.path.exists(min_path):
                vi_min[vi] = rxr.open_rasterio(min_path, chunks=chunks)
                vi_max[vi] = rxr.open_rasterio(max_path, chunks=chunks)

        if len(vi_min) == 0:
            return None, None

        return vi_min, vi_max


def get_item_datetime(item):
    return datetime.strptime(item.id, "%Y-%m-%dt%H%M%S")


def new_image_check(start_date, end_date, existing_data_loc, stac_loc='https://data.geo.admin.ch/api/stac/v0.9/'):
    year = int(start_date.split("-")[0])

    items = pull_from_stac(stac_loc=stac_loc, year=year, end_date=end_date, start_date=start_date)

    if existing_data_loc is not None:
        existing_meta = load_minmax_metadata(year, existing_data_loc)
    else:
        existing_meta = None

    processed_dates = set(existing_meta["processed_dates"]) if existing_meta else set()

    new_items = [item for item in items if get_item_datetime(item).isoformat() not in processed_dates]

    print(f"Already processed: {len(processed_dates)}")
    print(f"New images to process: {len(new_items)}")

    if len(new_items) == 0:
        return None, None, None
    else:
        return new_items


def load_and_process_assets(item, forest_mask, bbox=None, band_metadata=False, verbose=0):
    """
    Load bands from a STAC item, align 20m -> 10m, and return selected bands and valid mask.

    Memory notes (v200 vs v100):
    - v200 opens 9 single-band files per item instead of v100's 3 stacked
      files. Each open_rasterio() call holds its own GDAL dataset handle
      and block cache until closed/garbage collected. We now explicitly
      close the *source* lazy DataArrays right after computing, instead of
      relying on `del` + a single end-of-function gc.collect() (which
      doesn't call the underlying rasterio .close()).
    - Bands are computed in two smaller dask.compute() batches (10m group,
      20m/mask group) rather than one call materializing all 8 arrays
      simultaneously, which lowers peak RSS during this step.
    """
    tick = time.time()
    key_problem = None
    assets = item.assets

    try:
        b02_key = next(k for k in assets.keys() if k.endswith("_b02_10m.tif"))
        b03_key = next(k for k in assets.keys() if k.endswith("_b03_10m.tif"))
        b04_key = next(k for k in assets.keys() if k.endswith("_b04_10m.tif"))
        b08_key = next(k for k in assets.keys() if k.endswith("_b08_10m.tif"))
        b11_key = next(k for k in assets.keys() if k.endswith("_b11_20m.tif"))
        b05_key = next(k for k in assets.keys() if k.endswith("_b05_20m.tif"))
        b07_key = next(k for k in assets.keys() if k.endswith("_b07_20m.tif"))
        tmask_key = next(k for k in assets.keys() if k.endswith("_terrainmask_10m.tif"))
        cmask_key = next(k for k in assets.keys() if k.endswith("_cloudmask_10m.tif"))
    except StopIteration as e:
        key_problem = 1
        print(f"  WARNING: Problem loading the v200 single-band assets within this item! Skipping. {e}")

    if key_problem is not None:
        return None, None

    if band_metadata:
        print_metadata(item)

    open_kwargs = dict(masked=True, chunks=RASTER_CHUNK_SIZE)

    red = rxr.open_rasterio(assets[b04_key].href, **open_kwargs)
    green = rxr.open_rasterio(assets[b03_key].href, **open_kwargs)
    blue = rxr.open_rasterio(assets[b02_key].href, **open_kwargs)
    nir = rxr.open_rasterio(assets[b08_key].href, **open_kwargs)
    swir = rxr.open_rasterio(assets[b11_key].href, **open_kwargs)
    rededge1 = rxr.open_rasterio(assets[b05_key].href, **open_kwargs)
    rededge3 = rxr.open_rasterio(assets[b07_key].href, **open_kwargs)
    terrain_mask = rxr.open_rasterio(assets[tmask_key].href, **open_kwargs)
    cloud_mask = rxr.open_rasterio(assets[cmask_key].href, **open_kwargs)

    lazy_sources = [red, green, blue, nir, swir, rededge1, rededge3, terrain_mask, cloud_mask]

    if bbox is not None:
        try:
            red = red.rio.clip_box(*bbox)
            green = green.rio.clip_box(*bbox)
            blue = blue.rio.clip_box(*bbox)
            nir = nir.rio.clip_box(*bbox)
            swir = swir.rio.clip_box(*bbox)
            rededge1 = rededge1.rio.clip_box(*bbox)
            rededge3 = rededge3.rio.clip_box(*bbox)
            terrain_mask = terrain_mask.rio.clip_box(*bbox)
            cloud_mask = cloud_mask.rio.clip_box(*bbox)
        except NoDataInBounds:
            print("  No data in bbox, skipping asset")
            for src in lazy_sources:
                try:
                    src.close()
                except Exception:
                    pass
            return None, None
    else:
        # No bbox means the FULL national-extent scene is loaded for every
        # single band, on every date. That is almost certainly the single
        # biggest memory lever available here. Strongly prefer always
        # passing a bbox unless you genuinely need the whole country.
        warnings.warn(
            "load_and_process_assets called with bbox=None: loading full "
            "national-extent rasters for all 9 bands. This is the most "
            "common cause of memory blowups in the v200 pipeline.",
            RuntimeWarning,
        )

    if verbose == 1:
        load_time = time.time()
        print(f'  Loaded Data and Masks in {load_time - tick:.2f} seconds')
    else:
        load_time = tick

    red = red.astype("float32", copy=False)
    green = green.astype("float32", copy=False)
    blue = blue.astype("float32", copy=False)
    nir = nir.astype("float32", copy=False)

    if verbose == 1:
        astype10_time = time.time()
        print(f'  Converted 10m to float32 in {astype10_time - load_time:.2f} seconds')
    else:
        astype10_time = load_time

    # --- 20m -> 10m upsampling ---
    num_warp_threads = os.cpu_count() or 4

    swir_re1_re3 = xr.concat(
        [swir.squeeze('band', drop=True),
         rededge1.squeeze('band', drop=True),
         rededge3.squeeze('band', drop=True)],
        dim='band'
    ).assign_coords(band=[1, 2, 3])

    swir_re1_re3 = swir_re1_re3.rio.reproject_match(
        red, dtype="float32", num_threads=num_warp_threads
    )

    swir = swir_re1_re3.isel(band=[0]).assign_coords(band=[1])
    rededge1 = swir_re1_re3.isel(band=[1]).assign_coords(band=[1])
    rededge3 = swir_re1_re3.isel(band=[2]).assign_coords(band=[1])
    del swir_re1_re3

    if verbose == 1:
        astype20_time = time.time()
        print(f'  Converted 20m to 10m and float32 in {astype20_time - astype10_time:.2f} seconds')
    else:
        astype20_time = astype10_time

    valid_mask = ((cloud_mask != 1) & (terrain_mask <= 63) & (red > 0) & (forest_mask == 1))

    if verbose == 1:
        mask_time = time.time()
        print(f'  Formatted cloud and terrain mask in {mask_time - astype20_time:.2f} seconds')
    else:
        mask_time = astype20_time

    bands = {
        "red": red,
        "green": green,
        "blue": blue,
        "nir": nir,
        "swir": swir,
        "rededge1": rededge1,
        "rededge3": rededge3
    }

    if verbose == 1:
        print('  Executing tasks with Dask')

    # Compute 10m-native bands and the mask together (they share chunk
    # alignment), then the reprojected 20m bands separately, instead of one
    # dask.compute() call materializing all 8 arrays at their peak size at
    # once. This trims peak RSS during this function without changing
    # results.
    group_a_keys = ["red", "green", "blue", "nir"]
    group_b_keys = ["swir", "rededge1", "rededge3"]

    computed_a = dask.compute(*(bands[k] for k in group_a_keys), valid_mask, scheduler='threads')
    computed_b = dask.compute(*(bands[k] for k in group_b_keys), scheduler='threads')

    bands = dict(zip(group_a_keys, computed_a[:-1]))
    bands.update(dict(zip(group_b_keys, computed_b)))
    valid_mask = computed_a[-1]

    if verbose == 1:
        ex_time = time.time()
        print(f'  Executed all tasks in {ex_time - mask_time:.2f} seconds')

    # Explicitly close the original lazily-opened rasterio/GDAL datasets.
    # `del` alone drops the Python reference but relies on refcounting/GC to
    # trigger rasterio's __del__, which is not guaranteed to happen
    # promptly (especially with dask's threaded scheduler holding
    # references). Closing explicitly releases the GDAL dataset + block
    # cache immediately.
    for src in lazy_sources:
        try:
            src.close()
        except Exception:
            pass
    del lazy_sources, red, green, blue, nir, swir, rededge1, rededge3, terrain_mask, cloud_mask
    gc.collect()

    return bands, valid_mask


def build_template(national_template, bbox, output=False):
    national_da = (
        rxr.open_rasterio(national_template)
        .squeeze(drop=True)
    )

    bbox_template = national_da.rio.clip_box(*bbox)

    bbox_template = (
        bbox_template
        .astype("float32")
        .where(bbox_template != 255)
    )

    national_da.close()
    del national_da
    gc.collect()

    if output:
        template_dir = os.path.dirname(national_template)
        template_name = os.path.splitext(os.path.basename(national_template))[0]

        out_path = os.path.join(
            template_dir,
            f"{template_name}_bbox_template.tif"
        )

        bbox_template.rio.to_raster(out_path)

    return bbox_template


def plot_disco_result(raster, item_date, output_dir, min_pixel_count=100, show=True):
    import contextily as cx

    if isinstance(raster, str):
        da = xr.open_dataarray(raster).squeeze()
    elif isinstance(raster, xr.DataArray):
        da = raster.squeeze()
    else:
        raise TypeError(
            "raster must be a file path or an xarray.DataArray (rioxarray) to plot"
        )
    plot_data = da.where(da != -9999)

    if int(plot_data.count()) < min_pixel_count:
        print("This asset does not have enough valid pixels to plot")
        return False

    disco_colors = ['#0A2F1F', '#228B22', '#9ACD32', '#FFFF00', '#FFFFE0']
    custom_cmap = LinearSegmentedColormap.from_list("disco_smooth", disco_colors)

    plot_obj = plot_data.plot(
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        add_colorbar=False,
        size=8,
        alpha=1,
        aspect='auto',
    )

    ax = plot_obj.axes

    try:
        cx.add_basemap(ax, crs=plot_data.rio.crs.to_string(), source=cx.providers.Esri.WorldTopoMap)
    except Exception as e:
        print(f"  Warning: Could not load basemap: {e}")

    cbar = plt.colorbar(plot_obj, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Discoloration Probability', fontweight='bold')

    ax.set_title(f"Discoloration Probability: {item_date}", fontsize=14)
    ax.axis("off")

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"Disco_Proba_{item_date}.png"), dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close('all')  # was plt.close() -- closes only the current figure;
    # 'all' guarantees no lingering Figure objects survive across a season
    # loop of run_after_each_update() calls (matplotlib keeps every open
    # figure alive in pyplot's global state until explicitly closed).


def plot_disco_grid(raster_dir, pattern="Disco_Proba_*.tif", cols=3):
    import contextily as cx

    files = sorted(glob.glob(os.path.join(raster_dir, pattern)))
    if not files:
        print("No files found to grid!")
        return

    num_files = len(files)
    rows = math.ceil(num_files / cols)

    disco_colors = ['#D4E157', '#AED581', '#FFEB3B', '#FF9800', '#F44336', '#D32F2F']
    custom_cmap = LinearSegmentedColormap.from_list("disco_smooth", disco_colors)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    if num_files > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    print(f"Generating satellite grid for {num_files} images. This may take a moment...")

    for i, file_path in enumerate(files):
        ax = axes_flat[i]

        da = xr.open_dataarray(file_path).squeeze()
        plot_data = da.where(da != -9999)

        file_name = os.path.basename(file_path)
        date_str = file_name.replace("Disco_Proba_", "").replace(".tif", "")

        im = plot_data.plot(
            ax=ax,
            cmap=custom_cmap,
            vmin=0,
            vmax=1,
            add_colorbar=False,
            alpha=0.6,
            zorder=2
        )

        ax.set_aspect('equal', adjustable='datalim')

        try:
            crs_str = plot_data.rio.crs.to_string()
            cx.add_basemap(ax, crs=crs_str, source=cx.providers.Esri.WorldImagery, zoom='auto', zorder=1)
        except Exception as e:
            print(f"  Warning: Basemap failed for {date_str}: {e}")

        ax.set_title(f"Date: {date_str}", fontsize=12, fontweight='bold')
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.03)
    cbar.set_label('Discoloration Probability (Continuous Scale)', fontweight='bold', fontsize=14)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.suptitle("Seasonal Forest Health Summary: Satellite Overlay", fontsize=24, fontweight='bold', y=0.98)

    grid_out = os.path.join(raster_dir, "Satellite_Summary_Grid.png")
    plt.savefig(grid_out, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print(f"Finished! Summary saved to: {grid_out}")


def update_vi_min_max(items_to_process, year_of_interest, existing_data, forest_mask, template, bbox,
                      band_metadata=False, run_after_each_update=False, disco_model=None, output_dir=None,
                      plot_results=True, model_months=None):
    """
    Updates (or creates) VI min max rasters and metadata.

    :param model_months: optional iterable of month numbers (e.g. (5, 6, 7)
        for May-July). When set, min/max is still updated for every item in
        `items_to_process` (so the normalization baseline reflects the full
        season you fetched), but normalization + disco model application
        (the `run_after_each_update` block) only runs for items whose date
        falls in these months. Pass None (default) to run the model for
        every processed item, matching the old behavior. If you want the
        min/max baseline itself restricted to May-July too, simply fetch
        `items_to_process` with that narrower date range in the first
        place (e.g. via pull_from_stac(start_date=..., end_date=...)) and
        leave model_months=None.

    ** Memory fix **
    Previously vi_min/vi_max were only ever computed back down to concrete
    numpy arrays inside `save_minmax_rasters()`, which is only called when
    `existing_data is not None`. When existing_data is None (in-memory /
    live-update workflows), fmin/fmax/where/transpose kept chaining lazily
    on top of each other for every single date in the loop, so the dask
    graph -- and every date's underlying VI array, captured by closure --
    stayed alive in memory simultaneously for the whole season. That is
    the main cause of the OOM.

    Fix: after each per-VI min/max update we now force `.compute()`
    (via `.load()`) regardless of whether existing_data is set, so each
    iteration starts from a small, concrete numpy-backed array rather than
    an ever-growing lazy graph.
    """

    vi_min = None
    vi_max = None
    existing_meta = None

    drains_forest_mask = rxr.open_rasterio(forest_mask)
    if bbox is not None:
        drains_forest_mask = drains_forest_mask.rio.clip_box(*bbox)
    else:
        warnings.warn(
            "update_vi_min_max called with bbox=None: the forest mask and "
            "every band for every date will be processed at full national "
            "extent. Pass a bbox unless you really need the whole country.",
            RuntimeWarning,
        )

    for i, item in enumerate(items_to_process):
        print(f"\n Processing {i + 1}/{len(items_to_process)} : {item.id}")

        if band_metadata:
            print_metadata(item)

        bands, valid = load_and_process_assets(item, drains_forest_mask, bbox, band_metadata=band_metadata, verbose=1)

        if bands is None:
            continue

        tick = time.time()

        valid = valid.astype('uint8').transpose('band', 'y', 'x')

        vis = compute_vis(bands)

        vi_time = time.time()
        print(f'  Created VIs in {vi_time - tick:.2f} seconds')

        if vi_min is None:
            if existing_data is not None:
                try:
                    vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data, chunks='default')
                except FileNotFoundError:
                    vi_min, vi_max = None, None

            if vi_min is None:
                print(f'  No existing VI min or max data was found. Creating new files.')

                template_raster = template.squeeze(drop=True).astype("float32")

                vi_min = {k: template_raster.copy(deep=True) for k in vis.keys()}
                vi_max = {k: template_raster.copy(deep=True) for k in vis.keys()}

                comp_time = time.time()
                print(f'  Initialized annual Min Max from Template in {comp_time - vi_time:.2f} seconds')

        valid_aligned = valid.rio.reproject_match(vi_min['NDVI'])

        for k in vis:
            vi = vis[k]

            vi = vi.rio.reproject_match(vi_min[k]).chunk({'x': 1024, 'y': 1024})
            vi = vi.where(valid_aligned == 1, np.nan)

            vi_min[k] = xr.ufuncs.fmin(vi_min[k], vi)
            vi_max[k] = xr.ufuncs.fmax(vi_max[k], vi)

            vi_min[k] = vi_min[k].where(np.isfinite(vi_min[k]), np.nan).transpose('band', 'y', 'x')
            vi_max[k] = vi_max[k].where(np.isfinite(vi_max[k]), np.nan).transpose('band', 'y', 'x')

            vi_min[k] = vi_min[k].load()
            vi_max[k] = vi_max[k].load()

            if existing_data is not None:
                save_minmax_rasters({k: vi_min[k]}, {k: vi_max[k]}, year_of_interest, existing_data)

        del vis, vi
        gc.collect()

        comp_time = time.time()
        print(f'  Compared to existing Min Max in {comp_time - vi_time:.2f} seconds')

        if existing_data is not None:
            existing_meta = load_minmax_metadata(year_of_interest, existing_data)
        processed_dates = set(existing_meta["processed_dates"]) if existing_meta else set()
        updated_dates = sorted(set(processed_dates) | {get_item_datetime(item).isoformat()})
        if existing_data is not None:
            save_minmax_metadata(year_of_interest, updated_dates, existing_data)
        else:
            existing_meta = {
                "year": year_of_interest,
                "processed_dates": updated_dates,
                "last_updated": datetime.now(UTC).isoformat()
            }

        backup_time = time.time()
        print(f"  Min/Max saved + metadata updated in {backup_time - comp_time:.2f} seconds")

        item_month = get_item_datetime(item).month
        should_run_model = run_after_each_update and (model_months is None or item_month in model_months)

        if run_after_each_update and not should_run_model:
            print(f"  Skipping normalization/model application: month {item_month} not in model_months={model_months}")

        if should_run_model:
            print("  Running normalization and model application")

            # Reuse the bands/valid mask already loaded for the min/max
            # update above instead of re-fetching + re-clipping + re-
            # reprojecting all 9 full-country rasters a second time for
            # the same item.
            normalized_vis = normalize_vis(
                item,
                vi_min,
                vi_max,
                forest_mask,
                bbox,
                bands=bands,
                valid=valid
            )

            item_date = get_item_datetime(item).date().isoformat()

            disco_out = None

            if output_dir is not None:
                if disco_model is None:
                    for vi_name, da in normalized_vis.items():
                        out_path = os.path.join(
                            output_dir,
                            f"{vi_name}_{item_date}.tif"
                        )
                        da.transpose("band", "y", "x").rio.to_raster(out_path)
                else:
                    disco_out = os.path.join(
                        output_dir,
                        f"Disco_Proba_{item_date}.tif"
                    )

            if disco_model is not None:
                disco_da = apply_disco(normalized_vis, disco_model, disco_out)

                if plot_results:
                    plot_disco_result(disco_da, item_date, output_dir, show=True)

            # normalized_vis / disco_da held per-date intermediates -- drop
            # them explicitly rather than waiting for the next loop
            # iteration to overwrite the names.
            del normalized_vis
            if disco_model is not None:
                del disco_da
            gc.collect()

        # Bands/valid for this date are no longer needed once VIs and any
        # per-date model output are done with.
        del bands, valid, valid_aligned
        gc.collect()

    drains_forest_mask.close()
    del drains_forest_mask
    gc.collect()

    return vi_min, vi_max


def normalize_vis(closest_data, vi_min, vi_max, forest_mask, bounding, bands=None, valid=None):
    """
    :param bands: optional pre-loaded bands dict (from load_and_process_assets)
        for this same item. If provided, skips re-loading/re-clipping/
        re-reprojecting all 9 rasters from the STAC href a second time.
        Pass None (default) to keep the old standalone behavior, e.g. when
        calling this on its own via pull_closest_from_stac.
    :param valid: optional pre-loaded valid mask matching `bands`. Must be
        supplied together with `bands` (both or neither).
    """
    reused_inputs = bands is not None and valid is not None

    if not reused_inputs:
        drains_forest_mask = rxr.open_rasterio(forest_mask)
        if bounding is not None:
            drains_forest_mask = drains_forest_mask.rio.clip_box(*bounding)

        bands, valid = load_and_process_assets(closest_data, drains_forest_mask, bounding)

        drains_forest_mask.close()

    if bands is None:
        print('Data is Unavailable at this Time Step')
        return None

    # Cheap no-ops if `valid` was pre-loaded and already in this shape/dtype.
    valid = valid.astype('uint8').transpose('band', 'y', 'x')

    valid_aligned = valid.rio.reproject_match(vi_min['NDVI'])

    vis = compute_vis(bands)

    normalized = {}
    for k in vis:
        vi_matched = vis[k].rio.reproject_match(vi_min[k]).chunk({'x': 1024, 'y': 1024})
        vi_matched = vi_matched.where(valid_aligned == 1, np.nan)

        normalized[k] = ((vi_matched - vi_min[k]) / (vi_max[k] - vi_min[k])).load()
        normalized[k] = normalized[k].where(np.isfinite(normalized[k]), np.nan)

    del bands, valid, valid_aligned, vis
    gc.collect()

    return normalized


# ---------------------------------------------------------------------
# Interpolated (RBF) STAC pipeline
#
# The raw pipeline above (update_vi_min_max / normalize_vis) folds each
# STAC scene into min/max the moment it's pulled. That doesn't give
# rbf_interp anything to interpolate -- RBF needs forward/backward real
# observations around a *fixed* output-date calendar, the same way the
# FORCE pipeline pre-builds a full TSA archive before ever calling
# rbf_interp.run_year(). STAC scenes arrive one (or a few) at a time, so
# instead of a premade archive, every new scene's QC-masked VI raster is
# appended to a growing per-VI directory archive (one small single-band
# file per date -- appending a band to a national-extent multiband GeoTIFF
# would mean rewriting the whole file, and everyone else's already-written
# bands, on every new scene). Once enough of that archive exists for a
# fixed-cadence output date to become eligible (target_date + wait_days <=
# the latest archived date), rbf_interp is run for it and the
# *interpolated* value -- not the raw scene -- is what feeds min/max and
# the disco model, mirroring force_pull.run_tsa_workflow().
# ---------------------------------------------------------------------

def _archive_vi_dir(archive_root, vi_key):
    return os.path.join(archive_root, vi_key)


def archive_vi_raster(vi_da, vi_key, date_iso, archive_root):
    """
    Persists one date's QC-masked, grid-aligned VI raster into the per-VI
    archive used by the interpolated STAC pipeline as rbf_interp's data
    source (archive_root/<VI>/<date>.tif).

    If two scenes land on the same calendar date (e.g. two satellite
    overpasses), the new raster is averaged into the existing one
    per-pixel (NaN-skipping) rather than overwritten -- mirroring
    rbf_interp's own same-day-duplicate handling for premade FORCE
    archives (_collapse_bands_to_daily).
    """
    vi_dir = _archive_vi_dir(archive_root, vi_key)
    os.makedirs(vi_dir, exist_ok=True)
    out_path = os.path.join(vi_dir, f"{date_iso}.tif")

    new_arr = np.asarray(vi_da.squeeze(drop=True).values, dtype="float32")

    if os.path.exists(out_path):
        with rasterio.open(out_path) as src:
            existing_arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels expected
            new_arr = np.nanmean(np.stack([existing_arr, new_arr]), axis=0).astype("float32")

    tmp_path = out_path + ".tmp"
    with rasterio.open(
        tmp_path, "w", driver="GTiff", height=new_arr.shape[0], width=new_arr.shape[1],
        count=1, dtype="float32", crs=vi_da.rio.crs, transform=vi_da.rio.transform(),
        nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(new_arr, 1)
        dst.set_band_description(1, date_iso)
    shutil.move(tmp_path, out_path)

    return out_path


def latest_archived_date(archive_root, vi_keys):
    """Most recent date archived across all of `vi_keys`' archive folders, or None if empty."""
    latest = None
    for vi_key in vi_keys:
        vi_dir = _archive_vi_dir(archive_root, vi_key)
        if not os.path.isdir(vi_dir):
            continue
        for fname in os.listdir(vi_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() != ".tif":
                continue
            try:
                d = datetime.strptime(stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if latest is None or d > latest:
                latest = d
    return latest


def _wrap_interpolated_band(arr_2d, transform, crs):
    """
    Wraps one rbf_interp output date's 2D numpy array (see
    rbf_interp.delayed_interpolate_series_from_archive()) as a
    (band, y, x) DataArray with a working `.rio` accessor -- proper x/y
    coordinate arrays derived from `transform`, not just transform
    metadata -- so it behaves like any other VI band for
    reproject_match/clip_box downstream.
    """
    h, w = arr_2d.shape
    xs, _ = rasterio.transform.xy(transform, [0] * w, range(w))
    _, ys = rasterio.transform.xy(transform, range(h), [0] * h)

    da = xr.DataArray(
        arr_2d[None, :, :].astype("float32"), dims=("band", "y", "x"),
        coords={"band": [1], "y": ys, "x": xs},
    )
    da = da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da


def update_vi_min_max_interpolated(items_to_process, year_of_interest, existing_data, forest_mask,
                                    template, bbox, archive_root, band_metadata=False,
                                    disco_model=None, output_dir=None, plot_results=True,
                                    model_months=None, widths=None, wait_days=None,
                                    max_backward_gap_days=None, max_radius_days=None,
                                    season_start="05-01", season_end="07-24", step_days=5,
                                    despike=True, despike_threshold_factor=None, despike_max_iter=None,
                                    cross_vi_despike=False, chunk_size=None):
    """
    Interpolated counterpart to update_vi_min_max(). Two phases:

    1. Archive: every scene in `items_to_process` has its QC-masked VI
       rasters computed and appended to the per-VI directory archive under
       `archive_root`, regardless of month -- this is cheap (no min/max or
       model work) and just builds up the backward/forward context
       rbf_interp needs. `items_to_process` can span a much wider date
       range than the model season; pulling early scenes now means the
       first in-season output dates already have real historical context
       once interpolation starts.

    2. Interpolate: once archived, any fixed-cadence output date (see
       rbf_interp.growing_season_dates()) that has newly become eligible
       (target_date + wait_days <= the latest archived date -- see
       rbf_interp.eligible_output_dates()) is run through rbf_interp for
       every VI, and the *interpolated* raster -- not any single raw scene
       -- is folded into the running seasonal min/max and, within
       `model_months`, normalized and passed to the disco model. This
       mirrors force_pull.run_tsa_workflow() running on rbf_interp's
       premade output, just computed incrementally as new scenes arrive.

       Naturally a no-op before `season_start`: eligible_output_dates()
       returns nothing until the first output date is due, so pulling
       scenes far ahead of the model season costs archiving only.

    :param archive_root: parent folder for the per-VI raw-scene archive
        (archive_root/<VI>/<date>.tif). Kept separate from `existing_data`
        (min/max cache) -- it grows independently and is never rewritten,
        only appended to. Also used as the `existing_data_loc` passed to
        new_image_check() by the caller, so already-archived scenes aren't
        re-pulled on the next run.
    :param widths, wait_days, max_backward_gap_days, max_radius_days: see
        rbf_interp.delayed_smooth_one_date(); default to rbf_interp's own
        defaults when left None.
    :param season_start, season_end, step_days: define the fixed output-date
        calendar that min/max and the model are evaluated on -- typically
        much narrower than the range `items_to_process` was pulled over
        (see class-level note above).
    :param model_months, band_metadata, disco_model, output_dir,
        plot_results: as in update_vi_min_max().
    :param despike, despike_threshold_factor, despike_max_iter: see
        rbf_interp.despike_daily_cube() -- despike=True (the default)
        removes triplet-residual outliers from each VI's raw series
        before smoothing. threshold_factor/max_iter default to
        rbf_interp's own defaults when left None.
    :param cross_vi_despike: see rbf_interp.delayed_interpolate_series_multi_vi().
        Default False despikes each VI independently, one at a time
        (lower peak memory). If True, a date flagged as an outlier by any
        one VI's own residual test is excluded from every VI at that
        pixel (since contamination corrupts the shared raw bands, not one
        VI's formula in isolation), but requires every VI's full raw
        daily cube resident in memory at once.
    :param chunk_size: see rbf_interp._delayed_interpolate_series_generic().
        None (default) processes each VI's whole spatial extent at once;
        an integer pixel size processes it in chunk_size x chunk_size
        spatial windows to cap peak memory.
    :return: (vi_min, vi_max), or (None, None) if nothing has been
        archived yet / no output date is newly eligible.
    """
    from src.disco_ch.rbf_interp import (
        growing_season_dates, eligible_output_dates, delayed_interpolate_series_from_archive_multi_vi,
        DEFAULT_WIDTHS_DAYS, DEFAULT_WAIT_DAYS, DEFAULT_MAX_BACKWARD_GAP_DAYS, DEFAULT_MAX_RADIUS_DAYS,
        DEFAULT_DESPIKE_THRESHOLD_FACTOR, DEFAULT_DESPIKE_MAX_ITER,
    )
    widths = widths if widths is not None else DEFAULT_WIDTHS_DAYS
    wait_days = wait_days if wait_days is not None else DEFAULT_WAIT_DAYS
    max_backward_gap_days = max_backward_gap_days if max_backward_gap_days is not None else DEFAULT_MAX_BACKWARD_GAP_DAYS
    max_radius_days = max_radius_days if max_radius_days is not None else DEFAULT_MAX_RADIUS_DAYS
    despike_threshold_factor = despike_threshold_factor if despike_threshold_factor is not None else DEFAULT_DESPIKE_THRESHOLD_FACTOR
    despike_max_iter = despike_max_iter if despike_max_iter is not None else DEFAULT_DESPIKE_MAX_ITER

    vi_keys = ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]
    template_grid = template.squeeze(drop=True).astype("float32")

    # --- Phase 1: archive every new scene's QC-masked VI rasters ---
    if items_to_process:
        drains_forest_mask = rxr.open_rasterio(forest_mask)
        if bbox is not None:
            drains_forest_mask = drains_forest_mask.rio.clip_box(*bbox)
        else:
            warnings.warn(
                "update_vi_min_max_interpolated called with bbox=None: the "
                "forest mask and every band for every scene will be "
                "processed at full national extent. Pass a bbox unless you "
                "really need the whole country.",
                RuntimeWarning,
            )

        archived_meta = load_minmax_metadata(year_of_interest, archive_root)
        archived_processed = set(archived_meta["processed_dates"]) if archived_meta else set()

        for i, item in enumerate(items_to_process):
            print(f"\n Archiving {i + 1}/{len(items_to_process)} : {item.id}")

            if band_metadata:
                print_metadata(item)

            bands, valid = load_and_process_assets(item, drains_forest_mask, bbox,
                                                     band_metadata=band_metadata, verbose=1)
            if bands is None:
                continue

            valid = valid.astype('uint8').transpose('band', 'y', 'x')
            valid_aligned = valid.rio.reproject_match(template_grid)

            vis = compute_vis(bands)
            item_date = get_item_datetime(item).date().isoformat()

            for k in vi_keys:
                vi_aligned = vis[k].rio.reproject_match(template_grid).chunk({'x': 1024, 'y': 1024})
                vi_aligned = vi_aligned.where(valid_aligned == 1, np.nan).load()
                archive_vi_raster(vi_aligned, k, item_date, archive_root)

            archived_processed = archived_processed | {get_item_datetime(item).isoformat()}
            save_minmax_metadata(year_of_interest, sorted(archived_processed), archive_root)

            del bands, valid, valid_aligned, vis
            gc.collect()

        drains_forest_mask.close()
        del drains_forest_mask
        gc.collect()

    # --- Phase 2: interpolate + fold newly eligible output dates into min/max + model ---
    as_of_date = latest_archived_date(archive_root, vi_keys)
    if as_of_date is None:
        print("Nothing archived yet -- nothing to interpolate.")
        return None, None

    all_output_dates = growing_season_dates(year_of_interest, season_start, season_end, step_days)
    existing_meta = load_minmax_metadata(year_of_interest, existing_data) if existing_data else None
    processed = set(existing_meta["processed_dates"]) if existing_meta else set()

    new_output_dates = [
        d for d in eligible_output_dates(all_output_dates, as_of_date, wait_days)
        if d.isoformat() not in processed
    ]

    if not new_output_dates:
        print("No output dates are newly eligible yet -- nothing to interpolate.")
        if existing_data is not None:
            try:
                return load_minmax_rasters(year_of_interest, existing_data, chunks='default')
            except FileNotFoundError:
                pass
        return None, None

    print(f"Newly eligible output dates ({len(new_output_dates)}): "
          f"{new_output_dates[0]} .. {new_output_dates[-1]}, every {step_days}d, wait_days={wait_days}")

    archive_dirs = {k: _archive_vi_dir(archive_root, k) for k in vi_keys}
    results = delayed_interpolate_series_from_archive_multi_vi(
        archive_dirs, new_output_dates, widths=widths, wait_days=wait_days,
        max_backward_gap_days=max_backward_gap_days, max_radius_days=max_radius_days,
        despike=despike, despike_threshold_factor=despike_threshold_factor,
        despike_max_iter=despike_max_iter, cross_vi_despike=cross_vi_despike,
        chunk_size=chunk_size,
    )
    interpolated = {k: results[k][0] for k in vi_keys}
    transform, crs = next(iter(results.values()))[1:]

    vi_min = vi_max = None
    if existing_data is not None:
        try:
            vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data, chunks='default')
        except FileNotFoundError:
            vi_min, vi_max = None, None

    if vi_min is None:
        print("  No existing VI min or max data was found. Creating new files.")
        vi_min = {k: template_grid.copy(deep=True) for k in vi_keys}
        vi_max = {k: template_grid.copy(deep=True) for k in vi_keys}

    for i, output_date in enumerate(new_output_dates):
        date_str = output_date.isoformat()
        print(f"\n Processing interpolated output date {i + 1}/{len(new_output_dates)} : {date_str}")

        vi_bands = {k: _wrap_interpolated_band(interpolated[k][i], transform, crs) for k in vi_keys}

        vi_aligned_bands = {}
        for k in vi_keys:
            vi_aligned = _reproject_match_if_needed(vi_bands[k], vi_min[k]).chunk({'x': 1024, 'y': 1024})
            vi_aligned_bands[k] = vi_aligned

            vi_min[k] = xr.ufuncs.fmin(vi_min[k], vi_aligned)
            vi_max[k] = xr.ufuncs.fmax(vi_max[k], vi_aligned)
            vi_min[k] = vi_min[k].where(np.isfinite(vi_min[k]), np.nan).transpose('band', 'y', 'x').load()
            vi_max[k] = vi_max[k].where(np.isfinite(vi_max[k]), np.nan).transpose('band', 'y', 'x').load()

            if existing_data is not None:
                save_minmax_rasters({k: vi_min[k]}, {k: vi_max[k]}, year_of_interest, existing_data)

        processed = processed | {date_str}
        if existing_data is not None:
            save_minmax_metadata(year_of_interest, sorted(processed), existing_data)

        date_month = output_date.month
        should_run_model = disco_model is not None and (model_months is None or date_month in model_months)
        if disco_model is not None and not should_run_model:
            print(f"  Skipping model: month {date_month} not in model_months={model_months}")

        if should_run_model:
            print("  Running normalization and model application")

            normalized_vis = {}
            for k in vi_keys:
                normalized_vis[k] = ((vi_aligned_bands[k] - vi_min[k]) / (vi_max[k] - vi_min[k])).load()
                normalized_vis[k] = normalized_vis[k].where(np.isfinite(normalized_vis[k]), np.nan)

            disco_out = os.path.join(output_dir, f"Disco_Proba_{date_str}.tif") if output_dir is not None else None
            disco_da = apply_disco(normalized_vis, disco_model, disco_out)

            if plot_results:
                plot_disco_result(disco_da, date_str, output_dir, show=True)

            del normalized_vis, disco_da
            gc.collect()

        del vi_bands, vi_aligned_bands
        gc.collect()

    return vi_min, vi_max


def pull_closest_from_stac(target_date_str, stac_loc='https://data.geo.admin.ch/api/stac/v0.9/',
                           collection='ch.swisstopo.swisseo_s2-sr_v200'):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

    service = pystac_client.Client.open(stac_loc)
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    start_date = f'{target_date.year}-04-01'
    end_date = f'{target_date.year}-09-30'
    item_search = service.search(collections=[collection], datetime=f'{start_date}/{end_date}')
    item_list = list(item_search.items())

    if not item_list:
        print(f"No items found in {target_date.year}")
        return None

    closest_item = min(item_list, key=lambda item: abs(get_item_datetime(item) - target_date))

    print(f"Closest image to {target_date_str} is {closest_item.id} ({get_item_datetime(closest_item).date()})")
    return closest_item


if __name__ == '__main__':
    items = pull_from_stac(year=2026, start_date='2026-07-01', end_date='2026-07-15')

    if items:
        print_metadata(items[0])
    else:
        print("No items found matching the search criteria.")
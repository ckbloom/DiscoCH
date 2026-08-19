"""
force_pull.py

Replicates the seasonal min/max -> normalize -> disco-model workflow from
stac_pull.py, but sourced from pre-computed FORCE multilayer VI
TIFs (e.g. "20260101-20261231_001-365_HL_TSA_SEN2L_NDVI_TSS.tif") instead
of raw Sentinel-2 bands pulled from swisstopo STAC.

Key differences from the STAC pipeline:
  - VIs (NDVI, EVI, NDMI, CIRE, CCI) are already computed -- one multiband
    GeoTIFF per VI, one band per acquisition date(+satellite).
  - There is no separate cloud/terrain mask asset. The TSA
    product's own nodata value already encodes invalid pixels (read via
    masked=True), and combine that with the forest mask.
  - Band -> date mapping comes from each band's *description* string
    (e.g. "20260405_SEN2A"). The satellite suffix is ignored; only the
    8-digit date is used.

"""

import os
import re
import csv
import gc
import glob
import shutil
import time
import warnings
import contextlib
from datetime import datetime, UTC
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
import rasterio.warp
from rasterio.merge import merge as rio_merge

from src.disco_ch.apply_model_stac import apply_disco

# Shared helpers reused from the STAC pipeline module.
from src.disco_ch.stac_pull import (
    load_minmax_metadata,
    save_minmax_metadata,
    save_minmax_rasters,
    load_minmax_rasters,
    build_template,
    plot_disco_result,
    _reproject_match_if_needed,
)

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

# Default set of VIs. Overridable per-call via the `vi_keys` parameter on
# build_date_index() / update_vi_min_max_tsa() / run_tsa_workflow() /
# run_tsa_workflow_tiled() -- VI_KEYS remains the fallback when vi_keys=None.
VI_KEYS = ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]


def _resolve_vi_keys(vi_keys):
    """
    Normalizes the `vi_keys` argument accepted throughout this module into
    an ordered {canonical_name: file_token} mapping.

    Accepts:
      - None: uses the default VI_KEYS as an identity mapping, e.g.
        {"NDVI": "NDVI", "EVI": "EVI", ...}.
      - a plain list of names, e.g. ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]:
        also treated as an identity mapping. This is the right choice
        whenever the VI TIF filenames on disk already use the canonical
        tokens apply_disco expects.
      - a dict {canonical_name: file_token}: use this when a given year's
        TIF filenames use different tokens than apply_disco's hardcoded
        feature keys -- e.g. some FORCE naming truncates VI names
        (NDVI -> NDV, NDMI -> NDM, CIRE -> CRE). `file_token` is used
        ONLY for locating/matching files on disk (find_vi_file). Every
        dict built downstream of that (vi_paths, vi_date_maps, vi_stacks,
        vi_min, vi_max, vi_bands, normalized_vis) is keyed by
        `canonical_name` instead, so apply_disco's hardcoded
        vis["CCI"]/vis["EVI"]/vis["CIRE"]/vis["NDMI"]/vis["NDVI"] lookups
        always find what they expect regardless of what the files on disk
        are actually named. `canonical_name` should always be one of
        apply_disco's expected feature keys: "NDVI", "EVI", "NDMI",
        "CIRE", "CCI".

        Example, for a year where filenames use truncated VI tokens:
            vi_keys = {
                "NDVI": "NDV", "EVI": "EVI", "NDMI": "NDM",
                "CIRE": "CRE", "CCI": "CCI",
            }
    """
    if vi_keys is None:
        vi_keys = VI_KEYS
    if isinstance(vi_keys, dict):
        return dict(vi_keys)
    return {k: k for k in vi_keys}

RASTER_CHUNK_SIZE = {"x": 2048, "y": 2048}

# Matches either:
#   - FORCE-style: 8-digit date + underscore + satellite, e.g. "20260405_SEN2A"
#   - RBF-style:   ISO date only, e.g. "2026-03-01"
DATE_RE_COMPACT = re.compile(r"^(\d{8})_")
DATE_RE_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _parse_band_date(desc):
    """
    Parses a band description into an ISO date string, or None if it
    doesn't match either supported format.
    """
    m = DATE_RE_COMPACT.match(desc)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()

    m = DATE_RE_ISO.match(desc)
    if m:
        # Already ISO -- just validate it's a real date.
        return datetime.strptime(m.group(1), "%Y-%m-%d").date().isoformat()

    return None


def _raster_crs(path):
    """Cheaply reads a raster's CRS (header only, no pixel data)."""
    with rasterio.open(path) as src:
        return src.crs


def _reproject_bbox_if_needed(bbox, src_crs, dst_crs):
    """
    Reprojects a (minx, miny, maxx, maxy) bbox from src_crs to dst_crs via
    its corner points. Returns bbox unchanged if either CRS is missing or
    they already match.
    """
    if bbox is None or src_crs is None or dst_crs is None or src_crs == dst_crs:
        return bbox
    return rasterio.warp.transform_bounds(src_crs, dst_crs, *bbox)


def _open_reference_grid(path, chunks=None):
    """
    Lazily opens one band of a raster purely to serve as a
    reproject_match() target for its exact pixel grid (CRS, shape,
    transform) -- no pixel data is materialized for this. Used to align
    the (small, tile-clipped) forest mask/template onto a TSA tile's
    native grid once per tile, rather than re-warping every VI band onto
    the mask's grid inside the per-date loop.
    """
    if chunks is None:
        chunks = RASTER_CHUNK_SIZE
    return rxr.open_rasterio(path, chunks=chunks).isel(band=0)


def _valid_mask(vi_aligned, forest_mask_da, reference):
    """Boolean validity mask for one already-aligned VI band: real
    (non-NaN) data, intersected with the forest mask (reprojected/matched
    onto `reference`'s grid) when one is given. forest_mask_da=None skips
    the forest constraint entirely -- real data everywhere counts as
    valid -- which is what lets run_tsa_workflow()'s forest_mask_path=None
    process the full dataset unclipped instead of masking to forest
    pixels."""
    if forest_mask_da is not None:
        forest_aligned = _reproject_match_if_needed(forest_mask_da, reference)
        return (forest_aligned == 1) & (~np.isnan(vi_aligned))
    return ~np.isnan(vi_aligned)


# ---------------------------------------------------------------------
# File discovery / band <-> date mapping
# ---------------------------------------------------------------------

def find_vi_file(directory, vi_key, year_of_interest, pattern_template="{year}*_{vi}_TSS.tif"):
    """
    Locate the single multiband TIF for a given VI + year within
    `directory`. Filenames encode the year at the start (e.g.
    '20260101-20261231_..._NDVI_TSS.tif')
    """
    pattern = pattern_template.format(year=year_of_interest, vi=vi_key)
    matches = sorted(glob.glob(os.path.join(directory, pattern)))

    if not matches:
        raise FileNotFoundError(
            f"No TSA TIF found for VI '{vi_key}' / year {year_of_interest} "
            f"in {directory} (pattern '{pattern}')"
        )
    if len(matches) > 1:
        warnings.warn(
            f"Multiple files matched for VI '{vi_key}' / year {year_of_interest}: "
            f"{matches}. Using the first one: {matches[0]}"
        )
    return matches[0]


def get_band_dates(tif_path, verbose=False):
    """
    Returns {date_iso_string: [1-based_band_indices]}, parsed from each
    band's description (e.g. '20260405_SEN2A' -> '2026-04-05'). The
    satellite suffix is ignored. If multiple bands share a date (e.g. two
    satellite overpasses on the same day), all of their band indices are
    kept in the list -- the caller averages them per-pixel rather than
    picking one arbitrarily.

    :param verbose: if True, prints one line per overlapping date listing
        the exact band indices involved. If False (default), overlap
        detail is left to the caller (build_date_index prints a single
        summary line per VI file instead).
    """
    with rasterio.open(tif_path) as src:
        descriptions = src.descriptions  # tuple, 1 per band, band-index order

    date_to_bands = {}
    for i, desc in enumerate(descriptions, start=1):
        if not desc:
            continue
        date_str = _parse_band_date(desc)
        if date_str is None:
            continue

        date_to_bands.setdefault(date_str, []).append(i)

    if verbose:
        for date_str, band_indices in sorted(date_to_bands.items()):
            if len(band_indices) > 1:
                print(
                    f"  NOTE: {len(band_indices)} bands share date {date_str} in "
                    f"{os.path.basename(tif_path)} (bands {band_indices}); "
                    f"will be averaged per-pixel"
                )

    return date_to_bands


def build_date_index(tsa_dir, year_of_interest, vi_pattern_template="{year}*_{vi}_TSS.tif",
                      verbose=False, vi_keys=None):
    """
    For every VI, find its TIF (matching `year_of_interest`) and its
    date->band-indices map.

    :param verbose: passed through to get_band_dates(); if False (default),
        overlapping-band dates are summarized as a single count per VI
        file instead of one line per date.
    :param vi_keys: which VIs to look for, and (optionally) how their
        file-naming token differs from apply_disco's canonical feature
        name. Accepts a plain list (canonical name == file token) or a
        {canonical_name: file_token} dict for years where TIF filenames
        use a different abbreviation than the canonical name -- see
        _resolve_vi_keys() for details. Defaults to VI_KEYS when None.

    Returns (all keyed by canonical_name, regardless of what file_token
    was used to locate the file on disk):
        vi_paths: {canonical_name: path}
        vi_date_maps: {canonical_name: {date_iso: [band_indices]}}
        all_dates: sorted list of every date seen in any VI
        common_dates: sorted list of dates present in every VI (these are
            the only dates eligible for model application, since the model
            needs all requested features)
    """
    vi_key_map = _resolve_vi_keys(vi_keys)

    vi_paths = {}
    vi_date_maps = {}

    for canonical, file_token in vi_key_map.items():
        path = find_vi_file(tsa_dir, file_token, year_of_interest, vi_pattern_template)
        vi_paths[canonical] = path
        vi_date_maps[canonical] = get_band_dates(path, verbose=verbose)

        n_dates = len(vi_date_maps[canonical])
        n_overlap = sum(1 for bands in vi_date_maps[canonical].values() if len(bands) > 1)
        overlap_note = f" ({n_overlap} overlapping dates averaged)" if n_overlap else ""
        token_note = f" [file token '{file_token}']" if file_token != canonical else ""
        print(f"  {canonical}{token_note}: {os.path.basename(path)} -> {n_dates} dated bands{overlap_note}")

    all_dates = sorted(set().union(*vi_date_maps.values()))
    common_dates = sorted(set.intersection(*(set(d.keys()) for d in vi_date_maps.values())))

    print(f"  Total distinct dates across all VIs: {len(all_dates)}")
    print(f"  Dates present in ALL {len(vi_key_map)} VIs (model-eligible): {len(common_dates)}")

    return vi_paths, vi_date_maps, all_dates, common_dates


# ---------------------------------------------------------------------
# Lazy VI stack access
# ---------------------------------------------------------------------

def open_vi_stacks(vi_paths, chunks=None):
    """
    Lazily open each VI's multiband TIF once. Callers should .close() the
    returned handles when finished with the whole run.
    """
    if chunks is None:
        chunks = RASTER_CHUNK_SIZE

    stacks = {}
    for vi, path in vi_paths.items():
        stacks[vi] = rxr.open_rasterio(path, masked=True, chunks=chunks)
    return stacks


def get_vi_band(vi_stack, band_indices, bbox=None, scale_factor=None):
    """
    Select one date's band(s) out of a lazy multiband VI stack, clip to
    bbox if needed, and return a float32 (band, y, x) DataArray with band=[1] --
    matching the shape convention used elsewhere in the pipeline.

    :param band_indices: a single 1-based band index, or a list of them.
        When multiple indices are given (multiple satellite overpasses on
        the same date), they are averaged per-pixel with NaN/nodata
        pixels skipped rather than treated as zero -- so a pixel that's
        valid in only one of the overlapping bands still gets that value,
        and a pixel invalid in all of them stays NaN.
    :param scale_factor: optional multiplier applied after selection, for
        the case where the TSA product stores VIs as scaled integers
        (e.g. NDVI * 10000 as int16). Leave as None if the values are
        already in natural VI units.
    """
    if isinstance(band_indices, int):
        band_indices = [band_indices]

    da = vi_stack.sel(band=band_indices)

    if bbox is not None:
        # Only clip if the requested bbox is different from the raster's own bounds
        rb = da.rio.bounds()
        if not (np.isclose(bbox[0], rb[0]) and np.isclose(bbox[1], rb[1]) and
                np.isclose(bbox[2], rb[2]) and np.isclose(bbox[3], rb[3])):
            da = da.rio.clip_box(*bbox)

    da = da.astype("float32", copy=False)

    if scale_factor is not None:
        da = da * scale_factor

    if len(band_indices) > 1:
        da = da.mean(dim="band", skipna=True)
        da = da.expand_dims("band")

    da = da.assign_coords(band=[1])

    for stale_attr in ("long_name", "band_descriptions"):
        da.attrs.pop(stale_attr, None)

    return da.compute()

# ---------------------------------------------------------------------
# Min/max update + normalization, TSA version
# ---------------------------------------------------------------------

def dates_to_process(all_dates, year_of_interest, existing_data, start_date=None, end_date=None):
    """
    Filters `all_dates` (list of YYYY-MM-DD strings) down to dates in
    `year_of_interest`, optionally further restricted to [start_date,
    end_date] (inclusive, 'YYYY-MM-DD' strings), not already recorded in
    the existing metadata for that year. Mirrors new_image_check() from
    the STAC pipeline, but keyed on plain dates instead of full item
    datetimes.
    """
    existing_meta = load_minmax_metadata(year_of_interest, existing_data) if existing_data else None
    processed = set(existing_meta["processed_dates"]) if existing_meta else set()

    candidates = [d for d in all_dates if d.startswith(str(year_of_interest))]
    if start_date is not None:
        candidates = [d for d in candidates if d >= start_date]
    if end_date is not None:
        candidates = [d for d in candidates if d <= end_date]

    new_dates = [d for d in candidates if d not in processed]

    print(f"  Already processed: {len(processed)}")
    print(f"  New dates to process: {len(new_dates)}")

    return new_dates


def _recover_incomplete_apply(existing_data, year_of_interest, vi_date_maps, vi_stacks,
                               vi_min, vi_max, forest_mask_da, bbox, output_dir,
                               disco_model, model_months, scale_factors, vi_keys,
                               plot_results):
    """
    Guards against one specific crash window: a date's running min/max was
    already saved and the date was already recorded in `processed_dates`,
    but the disco model apply for that same date never finished writing
    its output tif (e.g. the process was killed between the metadata save
    and apply_disco() completing). Since dates_to_process() filters out
    anything already in `processed_dates`, such a date would otherwise be
    silently skipped forever on every future resume -- its model output
    would just never exist, with no error or warning.

    IMPORTANT LIMITATION: vi_min/vi_max are a single cumulative running
    snapshot, not a history -- there's no way to reconstruct what min/max
    looked like as of an *earlier* date once later dates have been folded
    in. So this can only ever recover the single most-recently-processed
    date (the one whose min/max update is still exactly reflected in the
    current cached vi_min/vi_max). If a crash happened further back and
    additional dates were processed since, that earlier date's model
    output is permanently unrecoverable via this mechanism -- only this
    guard against the *latest* date closes the window going forward.

    No-ops (does nothing) if there's no cached metadata, if the last
    processed date didn't qualify for a model run in the first place
    (incomplete VI coverage or outside model_months), or if its output
    tif already exists.
    """
    if existing_data is None or disco_model is None or output_dir is None:
        return

    existing_meta = load_minmax_metadata(year_of_interest, existing_data)
    if not existing_meta or not existing_meta.get("processed_dates"):
        return

    last_date = sorted(existing_meta["processed_dates"])[-1]

    date_present_in = [vi for vi in vi_keys if last_date in vi_date_maps.get(vi, {})]
    if len(date_present_in) != len(vi_keys):
        return  # model was never supposed to run for this date

    date_month = datetime.strptime(last_date, "%Y-%m-%d").month
    if model_months is not None and date_month not in model_months:
        return  # model was never supposed to run for this date

    expected_out = os.path.join(output_dir, f"Disco_Proba_{last_date}.tif")
    if os.path.exists(expected_out):
        return  # apply already completed last run -- nothing to recover

    print(f"  RECOVERY: {last_date} was marked processed but its model "
          f"output is missing ({expected_out}). Re-running the apply step "
          f"using the current cached min/max, which already reflects this "
          f"date. (Only the most-recently-processed date is recoverable "
          f"this way; earlier interruptions can't be reconstructed.)")

    scale_factors = scale_factors or {}
    vi_bands = {}
    for vi in vi_keys:
        band_indices = vi_date_maps[vi][last_date]
        vi_bands[vi] = get_vi_band(
            vi_stacks[vi], band_indices, bbox=bbox,
            scale_factor=scale_factors.get(vi)
        )

    normalized_vis = normalize_vis_tsa(vi_bands, vi_min, vi_max, forest_mask_da)
    disco_da = apply_disco(normalized_vis, disco_model, expected_out)

    if plot_results:
        plot_disco_result(disco_da, last_date, output_dir, show=True)

    print(f"  RECOVERY complete for {last_date} -> {expected_out}")

    del vi_bands, normalized_vis, disco_da
    gc.collect()


def update_vi_min_max_tsa(dates_to_run, vi_date_maps, vi_stacks, year_of_interest,
                           existing_data, forest_mask_da, template, bbox,
                           disco_model=None, output_dir=None, plot_results=True,
                           model_months=None, scale_factors=None, vi_keys=None):
    """
    TSA equivalent of update_vi_min_max(). Walks each date in
    `dates_to_run`, updates the running per-VI min/max, and -- for dates
    present in all requested VIs and matching `model_months` -- normalizes
    and runs the disco model.

    Before processing any new dates, also checks whether the most recently
    processed date from a prior run is missing its model output (see
    _recover_incomplete_apply() for the exact crash window this guards
    against) and re-runs just the apply step for it if so.

    :param scale_factors: optional {vi_key: multiplier} dict, see
        get_vi_band()'s scale_factor param. Keyed by canonical VI name
        (e.g. "NDVI"), not by file_token, when vi_keys is a mapping dict.
        Pass None to leave all VIs unscaled.
    :param vi_keys: list of canonical VI keys this run covers (must match
        the canonical keys already present in vi_date_maps/vi_stacks --
        i.e. pass the same vi_keys, list or {canonical: file_token} dict,
        that was used to build them). Defaults to VI_KEYS when None. Only
        the canonical names are used here; any file_token mapping is
        irrelevant at this point since vi_date_maps/vi_stacks are already
        keyed by canonical name.
    """
    vi_keys = list(_resolve_vi_keys(vi_keys).keys())
    scale_factors = scale_factors or {}

    vi_min = None
    vi_max = None
    existing_meta = None

    if existing_data is not None:
        try:
            vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data, chunks='default')
        except FileNotFoundError:
            vi_min, vi_max = None, None

    if vi_min is None:
        print("  No existing VI min/max found. Initializing from template.")
        template_raster = template.squeeze(drop=True).astype("float32")
        vi_min = {k: template_raster.copy(deep=True) for k in vi_keys}
        vi_max = {k: template_raster.copy(deep=True) for k in vi_keys}
    else:
        # Only relevant on resume -- a freshly initialized cache has no
        # prior processed date to check.
        _recover_incomplete_apply(
            existing_data, year_of_interest, vi_date_maps, vi_stacks,
            vi_min, vi_max, forest_mask_da, bbox, output_dir,
            disco_model, model_months, scale_factors, vi_keys, plot_results,
        )

    for i, date_str in enumerate(dates_to_run):
        print(f"\n Processing {i + 1}/{len(dates_to_run)} : {date_str}")

        date_present_in = [vi for vi in vi_keys if date_str in vi_date_maps[vi]]
        if not date_present_in:
            continue

        vi_bands = {}
        for vi in date_present_in:
            print(vi)
            band_indices = vi_date_maps[vi][date_str]

            vi_bands[vi] = get_vi_band(
                vi_stacks[vi], band_indices, bbox=bbox,
                scale_factor=scale_factors.get(vi)
            )

        for k in date_present_in:
            vi_da = vi_bands[k]
            vi_aligned = _reproject_match_if_needed(vi_da, vi_min[k]).chunk({'x': 1024, 'y': 1024})
            valid = _valid_mask(vi_aligned, forest_mask_da, vi_min[k])
            vi_aligned = vi_aligned.where(valid, np.nan)

            vi_min[k] = xr.ufuncs.fmin(vi_min[k], vi_aligned)
            vi_max[k] = xr.ufuncs.fmax(vi_max[k], vi_aligned)

            vi_min[k] = vi_min[k].where(np.isfinite(vi_min[k]), np.nan).transpose('band', 'y', 'x').load()
            vi_max[k] = vi_max[k].where(np.isfinite(vi_max[k]), np.nan).transpose('band', 'y', 'x').load()

            if existing_data is not None:
                save_minmax_rasters({k: vi_min[k]}, {k: vi_max[k]}, year_of_interest, existing_data)

        # Record this date as processed regardless of whether it hit every
        # VI, so a later resume doesn't re-scan it.
        if existing_data is not None:
            existing_meta = load_minmax_metadata(year_of_interest, existing_data)
        processed = set(existing_meta["processed_dates"]) if existing_meta else set()
        updated = sorted(processed | {date_str})
        if existing_data is not None:
            save_minmax_metadata(year_of_interest, updated, existing_data)
        else:
            existing_meta = {
                "year": year_of_interest,
                "processed_dates": updated,
                "last_updated": datetime.now(UTC).isoformat()
            }

        is_complete_date = len(date_present_in) == len(vi_keys)
        date_month = datetime.strptime(date_str, "%Y-%m-%d").month
        should_run_model = (
            disco_model is not None
            and is_complete_date
            and (model_months is None or date_month in model_months)
        )

        if disco_model is not None and not is_complete_date:
            print(f"  Skipping model: date {date_str} missing from "
                  f"{sorted(set(vi_keys) - set(date_present_in))}")
        elif disco_model is not None and not should_run_model:
            print(f"  Skipping model: month {date_month} not in model_months={model_months}")

        if should_run_model:
            print("  Running normalization and model application")

            normalized_vis = normalize_vis_tsa(vi_bands, vi_min, vi_max, forest_mask_da)

            disco_out = None
            if output_dir is not None:
                disco_out = os.path.join(output_dir, f"Disco_Proba_{date_str}.tif")

            disco_da = apply_disco(normalized_vis, disco_model, disco_out)

            if plot_results:
                plot_disco_result(disco_da, date_str, output_dir, show=True)

            del normalized_vis, disco_da
            gc.collect()

        del vi_bands
        gc.collect()

    return vi_min, vi_max


def normalize_vis_tsa(vi_bands, vi_min, vi_max, forest_mask_da):
    """
    TSA equivalent of normalize_vis(): given a dict of already-loaded VI
    bands for one date (all 5 keys required), normalize each against its
    running min/max and mask by the forest mask -- or, if
    forest_mask_da is None, just by real (non-NaN) data, i.e. the full
    dataset unclipped.
    """
    normalized = {}
    for k, vi_da in vi_bands.items():
        vi_aligned = _reproject_match_if_needed(vi_da, vi_min[k]).chunk({'x': 1024, 'y': 1024})
        valid = _valid_mask(vi_aligned, forest_mask_da, vi_min[k])
        vi_aligned = vi_aligned.where(valid, np.nan)

        normalized[k] = ((vi_aligned - vi_min[k]) / (vi_max[k] - vi_min[k])).load()
        normalized[k] = normalized[k].where(np.isfinite(normalized[k]), np.nan)

    return normalized


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_tsa_workflow(tsa_dir, year_of_interest, forest_mask_path, template_path, bbox,
                      existing_data="minmax_tsa", output_dir=None, disco_model=None,
                      plot_results=True, model_months=None,
                      vi_pattern_template="{year}*_{vi}_TSS.tif", scale_factors=None,
                      start_date=None, end_date=None, verbose=False, vi_keys=None):
    """
    End-to-end TSA pipeline: discover VI files/dates -> filter to unprocessed
    dates in `year_of_interest` (optionally windowed to [start_date,
    end_date]) -> update seasonal min/max -> normalize + apply the disco
    model for dates present in all 5 VIs.

    :param tsa_dir: directory containing the 5 multiband VI TIFs.
    :param year_of_interest: e.g. 2026. Used both to scope which dates get
        processed and to namespace the min/max cache files.
    :param start_date / end_date: optional 'YYYY-MM-DD' bounds narrowing
        which dates within `year_of_interest` get processed, e.g. to match
        a growing-season window. Leave both None to process every new date
        found in that year.
    :param forest_mask_path: path to the forest mask raster (same as the
        STAC pipeline's `forest_mask` argument). Pass None to skip forest
        masking entirely -- every real (non-NaN) pixel counts as valid,
        so the full dataset is processed unclipped by forest cover
        (still subject to `bbox`, if one is given).
    :param template_path: path to the national/AOI template raster used to
        initialize vi_min/vi_max when no cache exists yet.
    :param bbox: (minx, miny, maxx, maxy) clip box. Strongly recommended --
        omitting it processes at full raster extent for every VI/date.
    :param existing_data: folder for cached min/max rasters + metadata.
        Kept separate from the STAC pipeline's folder by default (this
        pipeline's processed_dates are plain dates, not STAC datetimes)
        -- pass None to run purely in-memory without caching/resume.
    :param output_dir: where to write Disco_Proba_<date>.tif + plots.
    :param disco_model: path to the pickled sklearn model, or None to
        just build/update min-max without running the model.
    :param model_months: optional iterable of month ints restricting which
        dates get the normalize+model step (min/max still updates for
        every processed date). None runs the model for every complete date.
    :param scale_factors: optional {vi_key: multiplier}, see get_vi_band().
        Keyed by canonical VI name (e.g. "NDVI"), not by file_token, when
        vi_keys is a mapping dict.
    :param verbose: if True, prints one line per overlapping-band date
        (see get_band_dates()). If False (default), overlaps are
        summarized as a single count per VI file instead.
    :param vi_keys: which VIs to process, and (optionally) how their
        file-naming token differs from apply_disco's canonical feature
        name. Accepts either:
          - a plain list, e.g. ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]
            (or a subset) -- assumes the TIF filenames already use these
            exact tokens.
          - a {canonical_name: file_token} dict for years where TIF
            filenames use a different abbreviation than the canonical
            name apply_disco expects, e.g. for filenames using truncated
            VI tokens:
                vi_keys = {"NDVI": "NDV", "EVI": "EVI", "NDMI": "NDM",
                           "CIRE": "CRE", "CCI": "CCI"}
            `file_token` is only used to locate files on disk; every VI
            dict from that point on (including what's ultimately handed
            to apply_disco) is keyed by `canonical_name`, so apply_disco's
            hardcoded vis["CCI"]/vis["EVI"]/etc. lookups always resolve
            correctly regardless of the on-disk naming. See
            _resolve_vi_keys() for full details.
        Defaults to VI_KEYS when None. The model still needs a date
        present in every one of these VIs to run (see is_complete_date in
        update_vi_min_max_tsa), so passing a shorter list also changes
        which dates are model-eligible.
    """
    vi_keys = vi_keys if vi_keys is not None else VI_KEYS

    print(f"Scanning TSA directory: {tsa_dir}")
    vi_paths, vi_date_maps, all_dates, _common_dates = build_date_index(
        tsa_dir, year_of_interest, vi_pattern_template, verbose=verbose, vi_keys=vi_keys
    )

    dates_to_run = dates_to_process(all_dates, year_of_interest, existing_data, start_date, end_date)
    if not dates_to_run:
        print("Nothing new to process.")
        return None, None

    # `bbox` is expressed in the TSA tile's own CRS (that's what
    # get_tile_bbox() reads it from). The forest mask / template are
    # typically national-extent rasters that may be in a different CRS,
    # so reproject the bbox to each one's own CRS before clipping it --
    # otherwise the clip can land entirely outside the raster's extent.
    tsa_crs = _raster_crs(next(iter(vi_paths.values()))) if bbox is not None else None

    template_bbox = _reproject_bbox_if_needed(bbox, tsa_crs, _raster_crs(template_path))
    template = build_template(template_path, template_bbox)

    forest_mask_da = None
    if forest_mask_path is not None:
        forest_mask_da = rxr.open_rasterio(forest_mask_path)
        if bbox is not None:
            forest_bbox = _reproject_bbox_if_needed(bbox, tsa_crs, forest_mask_da.rio.crs)
            forest_mask_da = forest_mask_da.rio.clip_box(*forest_bbox)

    if bbox is None:
        extent_note = "forest mask and every VI band" if forest_mask_da is not None else "every VI band"
        warnings.warn(
            f"run_tsa_workflow called with bbox=None: {extent_note} will be "
            f"processed at full extent. Pass a bbox unless you really need "
            f"the whole raster.",
            RuntimeWarning,
        )

    if bbox is not None:
        reference_grid = _open_reference_grid(next(iter(vi_paths.values())))
        try:
            template = _reproject_match_if_needed(template, reference_grid)
            if forest_mask_da is not None:
                forest_mask_da = _reproject_match_if_needed(forest_mask_da, reference_grid)
        finally:
            reference_grid.close()
            del reference_grid
            gc.collect()

    vi_stacks = open_vi_stacks(vi_paths)

    try:
        vi_min, vi_max = update_vi_min_max_tsa(
            dates_to_run, vi_date_maps, vi_stacks, year_of_interest,
            existing_data, forest_mask_da, template, bbox,
            disco_model=disco_model, output_dir=output_dir,
            plot_results=plot_results, model_months=model_months,
            scale_factors=scale_factors, vi_keys=vi_keys,
        )
    finally:
        for vi, stack in vi_stacks.items():
            try:
                stack.close()
            except Exception:
                pass
        if forest_mask_da is not None:
            try:
                forest_mask_da.close()
            except Exception:
                pass
        del vi_stacks, forest_mask_da
        gc.collect()

    return vi_min, vi_max


# ---------------------------------------------------------------------
# Grid-tile handling (X0052_Y0064-style folders)
# ---------------------------------------------------------------------

# Repo-relative path to the CSV listing the FORCE grid tiles that actually
# need processing (Tile_ID column, e.g. "X0055_Y0061") -- see
# load_force_grid_ids(). Computed from this file's own location so it
# resolves correctly regardless of the caller's working directory.
DEFAULT_FORCE_GRID_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "CH_FORCE_Grids.csv",
)


def load_force_grid_ids(csv_path=DEFAULT_FORCE_GRID_CSV):
    """
    Reads the Tile_ID column out of the FORCE grid CSV (data/CH_FORCE_Grids.csv
    by default) -- the list of 30km grid tiles that actually need
    processing. Returns a set of tile IDs, e.g. {"X0055_Y0061", ...}.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return {row["Tile_ID"].strip() for row in csv.DictReader(f) if row.get("Tile_ID")}


def discover_tiles(root_dir, tile_regex=r"^X\d{4}_Y\d{4}$", grid_csv=DEFAULT_FORCE_GRID_CSV):
    """
    Finds immediate subdirectories of `root_dir` whose names match the
    30km-grid naming convention (e.g. 'X0052_Y0064'). Returns their full
    paths, sorted for reproducible ordering/resumability.

    :param grid_csv: path to a CSV with a Tile_ID column (defaults to
        data/CH_FORCE_Grids.csv) restricting the result to only the tile
        IDs listed there -- the grids that actually need processing.
        Pass None to disable filtering and return every regex-matching
        subfolder regardless of the CSV. If the given/default path
        doesn't exist, filtering is skipped with a warning rather than
        raising, so callers whose repo layout doesn't include the CSV
        still work.
    """
    pattern = re.compile(tile_regex)
    tiles = [
        os.path.join(root_dir, name)
        for name in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, name)) and pattern.match(name)
    ]

    if grid_csv is not None:
        try:
            allowed = load_force_grid_ids(grid_csv)
        except FileNotFoundError:
            warnings.warn(
                f"discover_tiles: grid_csv '{grid_csv}' not found -- "
                f"processing every matching tile folder, unfiltered.",
                RuntimeWarning,
            )
        else:
            before = len(tiles)
            tiles = [t for t in tiles if os.path.basename(t) in allowed]
            skipped = before - len(tiles)
            if skipped:
                print(f"  Filtered out {skipped} tile folder(s) not listed in {grid_csv} "
                      f"({len(tiles)} remaining)")

    return tiles


def get_tile_bbox(tile_dir, year_of_interest, vi_pattern_template="{year}*_{vi}_TSS.tif",
                   vi_key="NDVI", vi_keys=None):
    """
    Reads the tile's own spatial bounds straight from one of its VI TIFs
    (defaults to NDVI, for `year_of_interest`), so we don't need the
    caller to hand-specify a bbox per tile -- the tile's own extent *is*
    the bbox.

    :param vi_key: which canonical VI to probe for bounds.
    :param vi_keys: the active VI set for this run -- a plain list or a
        {canonical_name: file_token} dict (see _resolve_vi_keys()). If
        given and `vi_key` isn't one of its canonical names (e.g. a
        custom vi_keys that excludes NDVI), falls back to the first
        canonical name instead of raising on a missing file. When
        vi_keys is a mapping dict, the file is located on disk using its
        file_token, not the canonical name.
    """
    if vi_keys is not None:
        vi_key_map = _resolve_vi_keys(vi_keys)
        if vi_key not in vi_key_map:
            vi_key = next(iter(vi_key_map))
        file_token = vi_key_map[vi_key]
    else:
        file_token = vi_key

    path = find_vi_file(tile_dir, file_token, year_of_interest, vi_pattern_template)
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def mosaic_outputs(tile_output_dirs, mosaic_dir, pattern="Disco_Proba_*.tif", nodata=-9999):
    """
    Merges same-named outputs (e.g. Disco_Proba_2026-06-15.tif) across all
    tile output directories into a single national/AOI mosaic per date.
    Tiles that don't have a given date simply leave a gap, which
    rasterio.merge handles natively via nodata.
    """
    os.makedirs(mosaic_dir, exist_ok=True)

    files_by_name = {}
    for tile_dir in tile_output_dirs:
        for f in glob.glob(os.path.join(tile_dir, pattern)):
            files_by_name.setdefault(os.path.basename(f), []).append(f)

    for fname, paths in sorted(files_by_name.items()):
        out_path = os.path.join(mosaic_dir, fname)

        if len(paths) == 1:
            # Only one tile produced this date -- no merge needed.
            shutil.copy(paths[0], out_path)
            print(f"  {fname}: single tile, copied directly -> {out_path}")
            continue

        srcs = [rasterio.open(p) for p in paths]
        try:
            mosaic_arr, mosaic_transform = rio_merge(srcs, nodata=nodata)
            meta = srcs[0].meta.copy()
            meta.update({
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": mosaic_transform,
                "nodata": nodata,
            })
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic_arr)
            print(f"  {fname}: mosaicked {len(paths)} tiles -> {out_path}")
        finally:
            for s in srcs:
                s.close()


def _process_one_tile(tile_path, year_of_interest, forest_mask_path, template_path,
                       existing_data_root, output_root, disco_model, model_months,
                       vi_pattern_template, scale_factors, plot_tile_results,
                       start_date, end_date, verbose, dask_threads_per_worker,
                       vi_keys=None, log_dir=None):
    """
    Runs run_tsa_workflow() for a single tile. Defined at module level (not
    nested) so it's picklable for ProcessPoolExecutor.

    :param dask_threads_per_worker: caps each worker process's internal
        dask/rioxarray thread pool, so N parallel tile processes don't each
        spawn their own full-width thread pool and oversubscribe the CPU.
        Pass None to leave dask's default scheduler behavior untouched.
    :param vi_keys: list or {canonical_name: file_token} dict of VIs to
        process for this tile -- see _resolve_vi_keys(). Defaults to
        VI_KEYS when None (resolved inside run_tsa_workflow / get_tile_bbox).
    :param log_dir: if given, all of this tile's normal stdout/stderr
        chatter (every print() inside build_date_index/update_vi_min_max_tsa/
        etc.) is redirected to log_dir/<tile_id>.log instead of the shared
        console -- this is what keeps concurrent tiles' output from
        interleaving into an unreadable mess. Only a single start line and a
        single done/error line per tile are still printed to the real
        console. Pass None to disable redirection and print live, as before
        (the right choice when running a single tile / max_workers=1).
    Returns (tile_id, tile_output_dir_or_None, error_message_or_None).
    """
    # Cap thread usage inside this worker process before any raster I/O
    # happens. Import dask here (rather than at module top) so this stays
    # a pure per-process side effect.
    if dask_threads_per_worker is not None:
        import dask
        dask.config.set(scheduler="threads", num_workers=dask_threads_per_worker)
        os.environ.setdefault("GDAL_NUM_THREADS", str(dask_threads_per_worker))

    tile_id = os.path.basename(tile_path)
    log_path = os.path.join(log_dir, f"{tile_id}.log") if log_dir else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    start_msg = f"=== Tile {tile_id} (pid {os.getpid()}) starting"
    start_msg += f" -- full log: {log_path} ===" if log_path else " ==="
    print(start_msg)

    t0 = time.time()
    tile_output_dir = None
    error_msg = None

    log_file_handle = open(log_path, "a") if log_path else None
    try:
        with contextlib.ExitStack() as stack:
            if log_file_handle is not None:
                stack.enter_context(contextlib.redirect_stdout(log_file_handle))
                stack.enter_context(contextlib.redirect_stderr(log_file_handle))

            # Everything below this point -- including every print() inside
            # get_tile_bbox/run_tsa_workflow/build_date_index/
            # update_vi_min_max_tsa/etc. -- goes to log_file_handle instead
            # of the shared console when log_dir is set.
            print(f"--- Tile {tile_id} (pid {os.getpid()}), started {datetime.now(UTC).isoformat()} ---")

            try:
                tile_bbox = get_tile_bbox(tile_path, year_of_interest, vi_pattern_template, vi_keys=vi_keys)
            except FileNotFoundError as e:
                error_msg = str(e)
                raise

            tile_existing_data = os.path.join(existing_data_root, tile_id) if existing_data_root else None
            tile_output_dir = os.path.join(output_root, tile_id) if output_root else None
            if tile_output_dir:
                os.makedirs(tile_output_dir, exist_ok=True)

            try:
                run_tsa_workflow(
                    tsa_dir=tile_path,
                    year_of_interest=year_of_interest,
                    forest_mask_path=forest_mask_path,
                    template_path=template_path,
                    bbox=tile_bbox,
                    existing_data=tile_existing_data,
                    output_dir=tile_output_dir,
                    disco_model=disco_model,
                    plot_results=plot_tile_results,
                    model_months=model_months,
                    vi_pattern_template=vi_pattern_template,
                    scale_factors=scale_factors,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=verbose,
                    vi_keys=vi_keys,
                )
            except Exception as e:
                # Surface the error with the tile ID attached rather than
                # letting the worker die silently -- as_completed() still
                # needs a result (or a re-raised exception) to report back
                # to the parent.
                tile_output_dir = None
                error_msg = f"{type(e).__name__}: {e}"
                raise
    except Exception:
        # Already captured into error_msg above; swallow here so we can
        # still print a clean one-line summary to the real console below.
        pass
    finally:
        if log_file_handle is not None:
            log_file_handle.close()

    elapsed_min = (time.time() - t0) / 60
    if error_msg:
        print(f"=== Tile {tile_id} FAILED after {elapsed_min:.1f} min: {error_msg}"
              + (f" -- see {log_path} ===" if log_path else " ==="))
    else:
        print(f"=== Tile {tile_id} done in {elapsed_min:.1f} min"
              + (f" -- log: {log_path} ===" if log_path else " ==="))

    return tile_id, tile_output_dir, error_msg


def run_tsa_workflow_tiled(root_dir, year_of_interest, forest_mask_path, template_path,
                            existing_data_root="minmax_tsa", output_root=None,
                            disco_model=None, model_months=None,
                            vi_pattern_template="{year}*_{vi}_TSS.tif", scale_factors=None,
                            tile_regex=r"^X\d{4}_Y\d{4}$",
                            plot_tile_results=False, plot_mosaic_results=True,
                            mosaic=True, start_date=None, end_date=None, verbose=False,
                            max_workers=1, dask_threads_per_worker=2, vi_keys=None,
                            log_dir=None, grid_csv=DEFAULT_FORCE_GRID_CSV):
    """
    Runs run_tsa_workflow() independently over every tile folder under
    `root_dir`, then (optionally) mosaics the per-date model outputs into
    single national/AOI rasters.

    Processing tile-by-tile rather than as one national raster keeps each
    iteration's working set small (a 30km tile is ~3000x3000px at 10m --
    trivial memory footprint) and makes the run resumable/restartable per
    tile via each tile's own min/max cache folder, instead of risking a
    single all-or-nothing national run. Tiles are independent of each
    other (separate cache folders, separate output dirs, own bbox), so
    with max_workers > 1 they run concurrently in separate processes via
    ProcessPoolExecutor.

    :param root_dir: directory containing the X####_Y#### tile folders.
    :param existing_data_root: parent folder for per-tile min/max caches;
        each tile gets its own subfolder (existing_data_root/<tile_id>)
        so caches never collide across tiles. Pass None to skip caching.
    :param output_root: parent folder for per-tile outputs
        (output_root/<tile_id>) plus the final output_root/mosaic/ dir.
    :param plot_tile_results: whether to plot each tile's individual
        per-date output (usually not useful -- defaults off). Strongly
        recommended to leave False when max_workers > 1: matplotlib is
        not reliably safe across multiple processes sharing a backend.
    :param plot_mosaic_results: whether to plot the final mosaicked
        per-date output after all tiles are merged.
    :param forest_mask_path / template_path: expected to cover the full
        extent of all tiles (e.g. national rasters); each tile clips its
        own region out of them automatically via get_tile_bbox().
        forest_mask_path=None skips forest masking for every tile -- see
        run_tsa_workflow()'s forest_mask_path -- processing each tile's
        full extent (still subject to that tile's own bbox) instead of
        restricting to forest pixels.
    :param start_date / end_date: optional 'YYYY-MM-DD' bounds narrowing
        which dates within `year_of_interest` get processed, applied
        identically to every tile.
    :param verbose: if True, prints per-date detail for overlapping bands;
        if False (default), just a per-VI-file overlap count.
    :param max_workers: number of tiles to process concurrently. 1 (the
        default) preserves the original sequential behavior. Each tile
        has been observed to use ~34 GB RSS in this pipeline, so pick
        max_workers based on available RAM (e.g. 2-3 on a 128 GB machine)
        rather than just CPU count -- memory, not CPU, is the binding
        constraint here. Start conservative and check actual peak RSS
        per worker before increasing.
    :param dask_threads_per_worker: caps the internal dask/rioxarray
        thread pool inside each worker process. Only relevant when
        max_workers > 1 -- prevents N worker processes from each
        spawning a full-width thread pool and oversubscribing the CPU.
        Ignored (dask's default scheduler behavior applies) when
        max_workers == 1.
    :param vi_keys: which VIs to process, applied identically to every
        tile. Accepts a plain list (assumes TIF filenames already use
        canonical VI tokens) or a {canonical_name: file_token} dict for
        years where filenames use different abbreviations than
        apply_disco's canonical feature names (e.g.
        {"NDVI": "NDV", "EVI": "EVI", "NDMI": "NDM", "CIRE": "CRE", "CCI": "CCI"}).
        See _resolve_vi_keys() for full details. Defaults to VI_KEYS when
        None.
    :param log_dir: directory to write one <tile_id>.log file per tile.
        Only takes effect when max_workers > 1 -- with multiple tiles
        printing concurrently from separate processes, interleaved
        console output becomes unreadable, so each tile's full chatter
        (every print() inside build_date_index/update_vi_min_max_tsa/
        etc.) is redirected to its own log file instead, and only a
        one-line start/done/error summary per tile reaches the console.
        Defaults to output_root/logs (or existing_data_root/logs if
        output_root isn't set, or './tile_logs' as a last resort) when
        left None and max_workers > 1. Ignored entirely when
        max_workers == 1, where output still streams live as before.
    :param grid_csv: path to a CSV with a Tile_ID column (defaults to
        data/CH_FORCE_Grids.csv) restricting processing to only the grid
        tiles listed there -- see discover_tiles(). Pass None to process
        every tile folder found under `root_dir`, unfiltered.
    """
    vi_keys = vi_keys if vi_keys is not None else VI_KEYS

    if max_workers > 1 and log_dir is None:
        log_dir = os.path.join(output_root or existing_data_root or ".", "logs")

    tiles = discover_tiles(root_dir, tile_regex, grid_csv=grid_csv)
    print(f"Found {len(tiles)} tile folders under {root_dir}")

    tile_output_dirs = []

    if max_workers <= 1:
        # Original sequential path, unchanged.
        for tile_path in tiles:
            tile_id, tile_out, err = _process_one_tile(
                tile_path, year_of_interest, forest_mask_path, template_path,
                existing_data_root, output_root, disco_model, model_months,
                vi_pattern_template, scale_factors, plot_tile_results,
                start_date, end_date, verbose, dask_threads_per_worker=None,
                vi_keys=vi_keys,
            )
            if err:
                print(f"  Skipping tile {tile_id}: {err}")
            elif tile_out:
                tile_output_dirs.append(tile_out)
    else:
        print(f"Processing tiles with max_workers={max_workers} "
              f"(dask_threads_per_worker={dask_threads_per_worker})")
        print(f"Per-tile logs: {log_dir}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_one_tile, tile_path, year_of_interest, forest_mask_path,
                    template_path, existing_data_root, output_root, disco_model,
                    model_months, vi_pattern_template, scale_factors,
                    plot_tile_results, start_date, end_date, verbose,
                    dask_threads_per_worker, vi_keys, log_dir,
                ): tile_path
                for tile_path in tiles
            }
            for fut in as_completed(futures):
                tile_path = futures[fut]
                fallback_id = os.path.basename(tile_path)
                try:
                    tile_id, tile_out, err = fut.result()
                except Exception as e:
                    # Worker process crashed outright (e.g. OOM kill) rather
                    # than raising cleanly inside _process_one_tile.
                    print(f"  Tile {fallback_id} failed hard: {type(e).__name__}: {e}")
                    continue
                if err:
                    print(f"  Skipping tile {tile_id}: {err}")
                elif tile_out:
                    tile_output_dirs.append(tile_out)

    if mosaic and output_root and disco_model is not None and tile_output_dirs:
        mosaic_dir = os.path.join(output_root, "mosaic")
        print(f"\nMosaicking tile outputs into {mosaic_dir}")
        mosaic_outputs(tile_output_dirs, mosaic_dir)

        if plot_mosaic_results:
            for f in sorted(glob.glob(os.path.join(mosaic_dir, "Disco_Proba_*.tif"))):
                date_str = os.path.basename(f).replace("Disco_Proba_", "").replace(".tif", "")
                plot_disco_result(f, date_str, mosaic_dir, show=True)


if __name__ == '__main__':
    # See run_tsa.py for the configured entry point (mirrors run_stac.py's
    # style: variables up top, single run_tsa_workflow_tiled() call below).
    print(
        "This module defines the TSA pipeline functions. Run run_tsa.py "
        "to actually process data."
    )
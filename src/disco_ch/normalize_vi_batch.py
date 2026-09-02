"""
normalize_vi_batch.py

Normalizes RBF-interpolated VI rasters (see force_tsi_batch.run_tsi_tiles())
against already-computed, per-pixel min/max rasters -- (value - min) /
(max - min), per pixel, broadcast across every band -- and writes the
result back out. No model is applied, and no min/max is computed or
updated here: this only APPLIES premade min/max rasters (e.g. the
minmax_dir cache built elsewhere by stac_pull.save_minmax_rasters() /
force_pull.run_tsa_workflow_tiled()). For a workflow that also computes/
updates min/max and runs the discoloration model in one pass, see
force_pull.run_tsa_workflow_tiled() / normalize_vis_tsa() instead.

Like the rest of the FORCE pipeline (force_tsi_batch.run_tsi_tiles(),
force_pull.run_tsa_workflow_tiled()), the min/max rasters are themselves
TILED, not one shared national raster per VI: force_pull's own
run_tsa_workflow_tiled() gives each tile its own min/max cache subfolder
(existing_data_root/<tile_id>/, see force_pull._process_one_tile()), and
minmax_paths()/normalize_vi_tiles() below expect that exact same layout
-- minmax_dir/<tile_id>/<vi>_min_<year>.tif and .../<vi>_max_<year>.tif.
Because each tile's min/max rasters were themselves built at that tile's
own extent, reproject_match (see _load_minmax_array()) is normally a
same-grid no-op; it still runs as a safety net in case a tile's min/max
raster doesn't line up pixel-for-pixel with its VI file.

Structured like force_tsi_batch.py: discover_tile_dirs()/find_tile_vi_file()
are reused as-is for FORCE-tiled archive discovery, and job execution
(sequential or parallel, skip-existing, progress printing, a run-params
record) mirrors run_tsi_tiles(). No stepwise/incremental counterpart is
provided here (unlike force_tsi_stepwise.py) -- every run recomputes
every requested (tile, VI) pair from scratch, skipping only files whose
output already exists (skip_existing).

By default, normalize_vi_tiles() also mosaics every tile's normalized
output into AOI-wide rasters afterward (mosaic=True), the same way
force_pull.run_tsa_workflow_tiled()/mosaic_outputs() does for its own
per-tile outputs -- see mosaic_vi_tiles_by_date(). Unlike a single big
multiband mosaic (every date merged across every tile all at once, which
is a real memory hazard for a full growing season x a national tile
count), it merges ONE BAND (one calendar timestep) at a time and writes
ONE single-band file per timestep -- rasterio.merge()'s own `indexes`
parameter means only that one band is ever actually read off disk per
tile, so peak memory stays bounded to a single timestep's worth of data
regardless of how many dates the input files hold. `start_date`/`end_date`
on normalize_vi_tiles() restrict which timesteps get mosaicked/exported
this way (independent of which dates the per-tile normalized files
themselves hold -- every date is still normalized; this only narrows
what's exported as a mosaic).

If tile_paths sit on slow/networked storage, mosaic_vi_tiles_by_date()'s
own parallelism (max_workers, across DATES) doesn't help a single slow
date -- every tile is still read serially within whichever process
handles that date. extract_vi_tile_subsets() fixes that by parallelizing
ACROSS TILES instead: it pulls just the requested date range out of
every tile (one combined read per tile, not one per date) into small,
local, tiled + band-interleaved files, concurrently -- see
scripts/mosaic_vi_tiles.py for the full extract-then-mosaic workflow.

One VI per call, matching how the min/max rasters are stored (one pair
per VI, per tile) -- e.g. just run vi="NDVI"; call again for other VIs
if you need more than one.
"""

import os
import re
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.merge import merge as merge_rasters

from src.disco_ch.force_tsi_batch import (
    discover_tile_dirs,
    find_tile_vi_file,
    DEFAULT_NODATA_INT16,
    _quantize_int16,
    _format_duration,
    _print_summary,
    _write_run_params,
)
from src.disco_ch.stac_pull import _reproject_match_if_needed


# Matches diff_vi_years.py's own default for RBF-interpolated (TSI) output
# -- NOT force_tsi_batch.DEFAULT_VI_PATTERN_TEMPLATE, which matches the
# raw (pre-interpolation) TSS files instead.
DEFAULT_VI_PATTERN_TEMPLATE = "{year}*_{vi}_TSS_TSI.tif"

# Matches stac_pull.save_minmax_rasters()/load_minmax_rasters()'s own
# per-tile-folder naming convention (e.g. "NDVI_min_2026.tif" inside
# minmax_dir/<tile_id>/) -- see module docstring.
DEFAULT_MINMAX_TEMPLATE = "{vi}_{bound}_{year}.tif"


def _predictor_for_dtype(dtype):
    """LZW predictor value matching mask_and_scale_rasters.py's own
    to_raster() convention -- 2 (horizontal differencing) for integer
    dtypes, 3 (floating point) for float dtypes. Passing predictor=2 for
    a float array is invalid/ineffective, so this picks the right one
    for whichever dtype is actually being written."""
    return 3 if np.dtype(dtype).kind == "f" else 2


def minmax_paths(minmax_dir, tile_id, vi, year, template=DEFAULT_MINMAX_TEMPLATE):
    """(min_path, max_path) for one VI/year within one tile's own min/max
    subfolder -- minmax_dir/<tile_id>/<vi>_min_<year>.tif and .../max --
    matching force_pull.run_tsa_workflow_tiled()'s per-tile cache layout
    (existing_data_root/<tile_id>/, see force_pull._process_one_tile())."""
    tile_minmax_dir = os.path.join(minmax_dir, tile_id)
    min_path = os.path.join(tile_minmax_dir, template.format(vi=vi, bound="min", year=year))
    max_path = os.path.join(tile_minmax_dir, template.format(vi=vi, bound="max", year=year))
    return min_path, max_path


# ---------------------------------------------------------------------
# Single-file normalization
# ---------------------------------------------------------------------

def _load_minmax_array(path, match_path):
    """Reads a single-band min/max raster and reproject-matches it onto
    `match_path`'s grid (see module docstring) -- skips the warp entirely
    in the common case where both are already on the same grid (expected
    here, since each tile's min/max rasters are built at that tile's own
    extent). Returns a 2D float32 numpy array, NaN where the min/max
    raster itself has no data."""
    da = rxr.open_rasterio(path, masked=True)
    match_da = rxr.open_rasterio(match_path)  # metadata (crs/transform/shape) only
    da = _reproject_match_if_needed(da, match_da)
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)
    return da.values.astype("float32")


def normalize_vi_file(in_path, out_path, min_path, max_path, input_scale_factor=10000,
                       scale_factor=10000, clip=True, nodata_value=DEFAULT_NODATA_INT16):
    """
    Normalizes every band of `in_path` (a multiband VI tif) against a
    single-band min raster and a single-band max raster (both covering
    the same tile/extent as `in_path` -- see minmax_paths()):
    (value - min) / (max - min), per pixel, broadcast across every band.
    Band descriptions (e.g. ISO dates, as written by force_tsi_batch's
    TSI output) are carried over unchanged.

    :param input_scale_factor: `in_path`'s own raw pixel values are
        divided by this before normalizing (default 10000, matching
        force_tsi_batch's own int16 TSI scaling convention, e.g. a
        stored 6500 is really NDVI 0.65). The min/max rasters (e.g. from
        stac_pull.save_minmax_rasters()) are assumed to already be in
        raw, unscaled VI units -- pass None if `in_path` is already
        unscaled (no division applied).
    :param scale_factor: normalized values are multiplied by this, then
        rounded and written as int16 (matching force_tsi_batch's own TSI
        scaling convention -- e.g. a normalized 0.65 is stored as 6500).
        Pass None to skip scaling and write raw float32 values instead
        (NaN as nodata).
    :param clip: if True (default), clip normalized values to [0, 1]
        before scaling/writing -- a pixel outside the [min, max] used to
        normalize is rare but not impossible (e.g. min/max built from a
        different date range than `in_path`'s own data). Pass False to
        preserve out-of-range values as-is.
    :return: out_path.
    """
    vi_min = _load_minmax_array(min_path, in_path)
    vi_max = _load_minmax_array(max_path, in_path)

    with rasterio.open(in_path) as src:
        arr = src.read(masked=True).astype("float32").filled(np.nan)
        transform, crs = src.transform, src.crs
        descriptions = src.descriptions

    if input_scale_factor is not None:
        arr = arr / input_scale_factor

    with np.errstate(divide="ignore", invalid="ignore"):
        # max == min (or either is NaN) produces inf/-inf/NaN here, which is
        # expected and handled immediately below via np.isfinite -- silence
        # numpy's warning for exactly this division rather than globally.
        normalized = (arr - vi_min[None, :, :]) / (vi_max[None, :, :] - vi_min[None, :, :])
    normalized = np.where(np.isfinite(normalized), normalized, np.nan)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n, h, w = normalized.shape

    if scale_factor is None:
        with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=n,
                            dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                            compress="LZW", predictor=_predictor_for_dtype("float32"),
                            tiled=True) as dst:
            for i in range(n):
                dst.write(normalized[i].astype("float32"), i + 1)
                dst.set_band_description(i + 1, descriptions[i] or "")
    else:
        quantized = _quantize_int16(normalized * scale_factor, nodata_value)
        with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=n,
                            dtype="int16", crs=crs, transform=transform, nodata=nodata_value,
                            compress="LZW", predictor=_predictor_for_dtype("int16"),
                            tiled=True) as dst:
            for i in range(n):
                dst.write(quantized[i], i + 1)
                dst.set_band_description(i + 1, descriptions[i] or "")

    return out_path


# ---------------------------------------------------------------------
# Shared job execution (sequential or parallel), one file at a time --
# mirrors force_tsi_batch._run_one()/_execute_jobs(). Each job carries
# its OWN min_path/max_path (tiled, not a single shared pair -- see
# module docstring), unlike force_tsi_batch's jobs, which share one set
# of RBF parameters across every file.
# ---------------------------------------------------------------------

def _run_one(in_path, out_path, min_path, max_path, input_scale_factor, scale_factor,
             clip, skip_existing):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor."""
    if skip_existing and os.path.exists(out_path):
        return in_path, out_path, "skipped"
    try:
        normalize_vi_file(in_path, out_path, min_path, max_path, input_scale_factor,
                           scale_factor, clip)
    except Exception as e:
        return in_path, out_path, f"failed: {type(e).__name__}: {e}"
    return in_path, out_path, "written"


def _execute_jobs(job_specs, input_scale_factor, scale_factor, clip, skip_existing,
                   max_workers, label_fn):
    """Sequential-or-parallel execution core shared by normalize_vi_batch()
    and normalize_vi_tiles(). job_specs is a list of (in_path, out_path,
    min_path, max_path, key) -- `key` identifies the job for progress
    printing/results.

    :return: {key: (in_path, out_path, status)}
    """
    results = {}
    total = len(job_specs)
    if total == 0:
        print("Nothing to do (0 jobs).")
        return results

    start = time.monotonic()

    def progress(done):
        elapsed = time.monotonic() - start
        eta = (f", ~{_format_duration(elapsed / done * (total - done))} remaining"
               if 0 < done < total else "")
        return f"[{done}/{total}, {_format_duration(elapsed)} elapsed{eta}]"

    if max_workers <= 1:
        for i, (in_path, out_path, min_path, max_path, key) in enumerate(job_specs, start=1):
            result = _run_one(in_path, out_path, min_path, max_path, input_scale_factor,
                               scale_factor, clip, skip_existing)
            print(f"  {progress(i)} {label_fn(key)}: {result[2]}")
            results[key] = result
        return results

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, in_path, out_path, min_path, max_path,
                             input_scale_factor, scale_factor, clip, skip_existing):
                (in_path, out_path, key)
            for in_path, out_path, min_path, max_path, key in job_specs
        }
        for fut in as_completed(futures):
            in_path, out_path, key = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = (in_path, out_path, f"failed: {type(e).__name__}: {e}")
            done += 1
            print(f"  {progress(done)} {label_fn(key)}: {result[2]}")
            results[key] = result

    return results


# ---------------------------------------------------------------------
# Flat file-list processing (no tile-directory structure assumed) --
# mirrors force_tsi_batch.find_tss_files()/run_tsi_batch().
# ---------------------------------------------------------------------

def find_vi_files(root_dir, pattern="*_TSS_TSI.tif"):
    """Glob every VI tif under root_dir (recursively) matching pattern --
    a flat, structure-agnostic finder. If your archive follows FORCE's
    usual one-directory-per-tile layout (min/max included), prefer
    normalize_vi_tiles()."""
    return sorted(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))


def normalize_vi_batch(job_inputs, out_dir, input_scale_factor=10000, scale_factor=10000,
                       clip=True, max_workers=1, skip_existing=True, suffix="_NORM"):
    """
    Runs normalize_vi_file() over an arbitrary flat list of (in_path,
    min_path, max_path) triples, writing each to out_dir with `suffix`
    appended before the extension. Each input file brings its OWN
    min/max raster pair (the min/max rasters are tiled just like the VI
    files themselves -- see module docstring) -- there's no single
    shared min/max pair to pass separately, unlike a naive flat batch.

    :param job_inputs: iterable of (in_path, min_path, max_path) --
        e.g. built by pairing find_vi_files() results with
        minmax_paths(minmax_dir, tile_id, vi, year) for whatever tile_id
        each in_path belongs to.
    :param skip_existing: if True, a file whose output already exists is
        left untouched instead of being recomputed.
    :return: list of (in_path, out_path, status) -- status is "written",
        "skipped", or "failed: <message>", in the same order as job_inputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    job_inputs = list(job_inputs)
    job_specs = []
    for in_path, min_path, max_path in job_inputs:
        base, ext = os.path.splitext(os.path.basename(in_path))
        out_path = os.path.join(out_dir, f"{base}{suffix}{ext}")
        job_specs.append((in_path, out_path, min_path, max_path, in_path))  # key = in_path (unique)

    start = time.monotonic()
    results_by_key = _execute_jobs(
        job_specs, input_scale_factor, scale_factor, clip, skip_existing, max_workers,
        label_fn=lambda key: key,
    )
    ordered = [results_by_key[in_path] for in_path, _min, _max in job_inputs]
    _print_summary([status for _, _, status in ordered], time.monotonic() - start)
    return ordered


# ---------------------------------------------------------------------
# Band -> label resolution, shared by extract_vi_tile_subsets() and
# mosaic_vi_tiles_by_date() -- handles BOTH a multiband, per-band-dated
# archive (e.g. TSI/normalized VI output) AND a single-band file with no
# date information at all (e.g. a min/max raster -- stac_pull.
# save_minmax_rasters() writes no band description), so the same
# extract/mosaic pipeline works for either without choking on files that
# were never date-stamped in the first place.
# ---------------------------------------------------------------------

def _resolve_band_labels(path, start_date=None, end_date=None):
    """
    Maps a file's bands to labels used for filtering/naming: real ISO
    date strings (from band descriptions) when present, matching
    force_tsi_batch's TSI convention -- start_date/end_date filter these
    as before. A file with NO per-band date descriptions at all (e.g. a
    single-band min/max raster) has nothing to date-filter, so every
    band is kept as-is and labeled "" (a lone band -- no per-band name
    needed) or "band2", "band3", ... for the 2nd band onward (only
    reached if a non-dated file happens to have more than one band,
    which isn't a documented convention here but is handled so filenames
    still can't collide).

    :return: (labels, band_indices) -- parallel lists (one entry per
        band to keep), band_indices 1-based (rasterio band numbers).
    :raises ValueError: if the file HAS date descriptions but none
        survive the start_date/end_date filter.
    """
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)

    dated = sorted({d for d in descriptions if d})
    if dated:
        labels = dated
        if start_date is not None:
            labels = [d for d in labels if d >= start_date]
        if end_date is not None:
            labels = [d for d in labels if d <= end_date]
        if not labels:
            raise ValueError(
                f"No bands left after filtering to [{start_date}, {end_date}] "
                f"(available: {dated})"
            )
        return labels, [descriptions.index(d) + 1 for d in labels]

    if start_date is not None or end_date is not None:
        print(f"  ({os.path.basename(path)} has no per-band date descriptions -- "
              f"start_date/end_date ignored, keeping every band)")
    n = len(descriptions)
    labels = [""] if n == 1 else [f"band{i + 1}" for i in range(n)]
    return labels, list(range(1, n + 1))


# ---------------------------------------------------------------------
# Parallel per-tile extraction -- pulls just the requested date-range
# band subset out of each tile file, in parallel ACROSS TILES, onto
# local disk, before mosaicking. This is the actual fix for a mosaic
# that "churns on the first date forever": mosaic_vi_tiles_by_date()
# (and rasterio.merge() underneath it) reads every tile SERIALLY within
# whichever process/date is handling it -- with max_workers=1 that's one
# process reading every tile one at a time for the first date; with
# max_workers>1 it parallelizes across DATES, but each individual date's
# own tile reads are still serial. If tile_paths sit on a slow/networked
# drive and just ONE tile read is slow, nothing else can happen until it
# finishes. Extracting per TILE in parallel fixes exactly that: tiles are
# fully independent, so N workers read N different tiles' data
# concurrently, each paying its own (possibly slow) I/O cost only once,
# overlapped with every other tile's I/O instead of queued behind it.
# ---------------------------------------------------------------------

def _extract_one_tile(in_path, out_path, band_indices, band_descriptions):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor.

    Reads ONLY band_indices from in_path in a SINGLE read call (one pass
    over whatever I/O `in_path` requires, covering every requested date
    at once -- not one read per date), and writes them to out_path as a
    small, local, tiled + BAND-interleaved multiband tif. interleave=
    "band" (unlike the GDAL default, PIXEL) stores each band's blocks
    separately, so a later per-band read (by mosaic_vi_tiles_by_date())
    only ever touches that one band's own data, not every band packed
    into the same block."""
    with rasterio.open(in_path) as src:
        arr = src.read(indexes=band_indices)
        transform, crs, nodata = src.transform, src.crs, src.nodata

    n, h, w = arr.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=n,
                        dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata,
                        compress="LZW", predictor=_predictor_for_dtype(arr.dtype),
                        tiled=True, interleave="band") as dst:
        dst.write(arr)
        for i, band_idx in enumerate(band_indices, start=1):
            dst.set_band_description(i, band_descriptions[band_idx - 1] or "")
    return out_path


def extract_vi_tile_subsets(tile_paths, scratch_dir, start_date=None, end_date=None,
                            max_workers=4, skip_existing=True):
    """
    Extracts just the [start_date, end_date] band subset out of every
    tile in tile_paths, IN PARALLEL ACROSS TILES, writing small local
    tiled + band-interleaved files to scratch_dir. Meant to run BEFORE
    mosaic_vi_tiles_by_date() -- see the section docstring above for why
    this (not more mosaic-side parallelism) is the fix for a slow first
    date: it parallelizes the actually-slow per-tile reads themselves,
    each paid exactly once (one combined read per tile, not one per
    date), instead of serially inside whichever date/process reads them.

    :param scratch_dir: local, fast-disk directory to write the small
        extracted per-tile files to (one per input tile, same basename).
    :param start_date, end_date: optional 'YYYY-MM-DD' bounds -- see
        mosaic_vi_tiles_by_date(). None (default) extracts every date.
    :param max_workers: tiles are fully independent, so this parallelizes
        cleanly. Pick this based on how many concurrent reads your
        storage can actually sustain -- a network share may saturate
        well before your CPU core count does, so more isn't always
        better; try a handful (e.g. 4-8) before assuming higher helps.
    :param skip_existing: if True (default), a tile whose extracted file
        already exists in scratch_dir is left untouched -- re-running
        this after a partial/interrupted extraction only fills in what's
        missing.
    :return: (extracted_paths, labels) -- extracted_paths in the same
        order as tile_paths; labels is the sorted list of band labels
        actually extracted (ISO dates for a dated archive; "" for a lone
        undated band, e.g. a min/max raster -- see _resolve_band_labels()).
    """
    if not tile_paths:
        raise ValueError("extract_vi_tile_subsets: no tile raster paths given")

    with rasterio.open(tile_paths[0]) as src0:
        descriptions = list(src0.descriptions)
    labels, band_indices = _resolve_band_labels(tile_paths[0], start_date, end_date)

    os.makedirs(scratch_dir, exist_ok=True)
    # FORCE tile filenames commonly share the exact same basename across
    # every tile (the tile identity lives in the FOLDER name, not the
    # filename) -- prefixing with the parent directory name keeps every
    # tile's extracted file distinct instead of silently overwriting one
    # another down to a single tile's worth of data.
    jobs = [(p, os.path.join(scratch_dir, f"{os.path.basename(os.path.dirname(p))}__{os.path.basename(p)}"))
            for p in tile_paths]

    extracted = {}
    total = len(jobs)
    start = time.monotonic()

    def progress(done):
        elapsed = time.monotonic() - start
        eta = (f", ~{_format_duration(elapsed / done * (total - done))} remaining"
               if 0 < done < total else "")
        return f"[{done}/{total}, {_format_duration(elapsed)} elapsed{eta}]"

    if max_workers <= 1:
        for i, (in_path, out_path) in enumerate(jobs, start=1):
            if skip_existing and os.path.exists(out_path):
                extracted[in_path] = out_path
            else:
                extracted[in_path] = _extract_one_tile(in_path, out_path, band_indices, descriptions)
            print(f"  {progress(i)} {os.path.basename(in_path)}: extracted")
        return [extracted[p] for p in tile_paths], labels

    to_submit = {}
    for in_path, out_path in jobs:
        if skip_existing and os.path.exists(out_path):
            extracted[in_path] = out_path
        else:
            to_submit[in_path] = out_path

    done = len(extracted)
    if done:
        print(f"  [{done}/{total}] already extracted, skipping")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_one_tile, in_path, out_path, band_indices, descriptions): in_path
            for in_path, out_path in to_submit.items()
        }
        for fut in as_completed(futures):
            in_path = futures[fut]
            extracted[in_path] = fut.result()
            done += 1
            print(f"  {progress(done)} {os.path.basename(in_path)}: extracted")

    return [extracted[p] for p in tile_paths], labels


# ---------------------------------------------------------------------
# Per-date single-band file discovery/extraction -- a DIFFERENT archive
# shape than everything above: force_pull.py's update_vi_min_max_tsa()
# (export_normalized_vi_dir param, see _export_stepwise_da()) writes ONE
# single-band file PER DATE, under <tile_id>/<vi>/<vi>_<date>_NORM.tif,
# instead of one multiband file per tile holding a whole year/date-range.
# Each file's date lives in its FILENAME -- scale_and_save_as_int_from_da()
# (what _export_stepwise_da() delegates to) never sets a band description,
# so _resolve_band_labels() would see every one of these as an "undated"
# lone-band file. find_tile_vi_file() also can't be reused as-is here: it
# raises if more than one file matches, but this layout has many matches
# per (tile, vi) by design (one per date).
#
# The fix is to STACK each tile's own per-date files into one small local
# multiband file during extraction (band descriptions = parsed date
# strings) -- same output shape/convention _extract_one_tile() already
# produces, so mosaic_vi_tiles_by_date() downstream needs no changes at
# all once this phase is done.
# ---------------------------------------------------------------------

# {vi} is filled in twice over: once for the <vi> subfolder, once for the
# filename prefix -- matching _export_stepwise_da()'s
# out_dir/<vi>/<vi>_<date>_NORM.tif convention.
DEFAULT_DATED_FILE_PATTERN = "{vi}/{vi}_*_NORM.tif"
DATED_FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def find_tile_vi_dated_files(tile_dir, file_token, pattern=DEFAULT_DATED_FILE_PATTERN,
                              start_date=None, end_date=None):
    """
    Finds every per-date single-band file for one VI within one tile
    directory (see DEFAULT_DATED_FILE_PATTERN / module section docstring
    above) -- unlike find_tile_vi_file(), MULTIPLE matches are expected
    (one per date), not an error. Each match's date is parsed straight
    out of its FILENAME, since these files carry no per-band date
    description of their own, then optionally filtered to
    [start_date, end_date] ('YYYY-MM-DD' strings, inclusive).

    :return: sorted list of (date_str, path), ascending by date. Empty if
        the tile has no files for this VI at all (not an error -- mirrors
        find_tile_vi_file()'s callers, which skip a tile missing a VI).
    :raises ValueError: if a matched filename doesn't contain a
        YYYY-MM-DD date -- an unexpected naming drift worth failing
        loudly on, rather than silently mis-ordering/mis-labeling bands
        downstream.
    """
    pattern_filled = pattern.format(vi=file_token)
    matches = sorted(glob.glob(os.path.join(tile_dir, pattern_filled)))

    dated = []
    for path in matches:
        m = DATED_FILENAME_DATE_RE.search(os.path.basename(path))
        if not m:
            raise ValueError(f"Could not parse a YYYY-MM-DD date from filename: {path!r}")
        date_str = m.group(1)
        if start_date is not None and date_str < start_date:
            continue
        if end_date is not None and date_str > end_date:
            continue
        dated.append((date_str, path))

    return sorted(dated)


def _extract_one_tile_from_dated_files(dated_files, out_path):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor.

    Stacks one tile's per-date single-band files (already date-filtered,
    ascending -- see find_tile_vi_dated_files()) into ONE small, local,
    tiled + band-interleaved multiband file, each band's description set
    to its source file's own date string -- giving _resolve_band_labels()
    real per-band dates to work with downstream even though none of the
    individual source files carry one themselves. Same output
    shape/convention as _extract_one_tile().
    """
    arrays = []
    transform = crs = nodata = dtype = None
    for _date_str, path in dated_files:
        with rasterio.open(path) as src:
            arr = src.read(1)
            if transform is None:
                transform, crs, nodata, dtype = src.transform, src.crs, src.nodata, arr.dtype
        arrays.append(arr)

    stacked = np.stack(arrays, axis=0)
    n, h, w = stacked.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=n,
                        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
                        compress="LZW", predictor=_predictor_for_dtype(dtype),
                        tiled=True, interleave="band") as dst:
        dst.write(stacked)
        for i, (date_str, _path) in enumerate(dated_files, start=1):
            dst.set_band_description(i, date_str)
    return out_path


def extract_vi_tile_subsets_from_dated_files(tile_dated_files, scratch_dir, max_workers=4,
                                              skip_existing=True):
    """
    Like extract_vi_tile_subsets(), but for the per-date single-band file
    layout (see find_tile_vi_dated_files()) instead of one multiband file
    per tile: stacks each tile's own (already date-filtered) per-date
    files into one small local multiband file, in parallel ACROSS TILES
    -- same rationale as extract_vi_tile_subsets() for why this (not more
    mosaic-side parallelism) is what actually overlaps slow per-tile I/O.

    :param tile_dated_files: {tile_id: [(date_str, path), ...]} -- e.g.
        built by calling find_tile_vi_dated_files() once per tile,
        already filtered to whatever date window you want extracted. A
        tile mapped to an empty list is skipped (not an error).
    :return: (extracted_paths, labels) -- extracted_paths for every tile
        with at least one file, in sorted tile_id order; labels is the
        sorted, deduplicated list of every date actually present across
        all tiles (same return shape as extract_vi_tile_subsets(), so
        mosaic_vi_tiles_by_date() needs no changes to consume it).
    :raises ValueError: if no tile has any file at all (nothing to
        extract).
    """
    tile_ids = sorted(tid for tid, files in tile_dated_files.items() if files)
    if not tile_ids:
        raise ValueError("extract_vi_tile_subsets_from_dated_files: no dated files found for any tile")

    os.makedirs(scratch_dir, exist_ok=True)
    jobs = [(tid, tile_dated_files[tid], os.path.join(scratch_dir, f"{tid}__stacked.tif"))
            for tid in tile_ids]
    all_labels = sorted({d for tid in tile_ids for d, _p in tile_dated_files[tid]})

    extracted = {}
    total = len(jobs)
    start = time.monotonic()

    def progress(done):
        elapsed = time.monotonic() - start
        eta = (f", ~{_format_duration(elapsed / done * (total - done))} remaining"
               if 0 < done < total else "")
        return f"[{done}/{total}, {_format_duration(elapsed)} elapsed{eta}]"

    if max_workers <= 1:
        for i, (tid, dated_files, out_path) in enumerate(jobs, start=1):
            if skip_existing and os.path.exists(out_path):
                extracted[tid] = out_path
            else:
                extracted[tid] = _extract_one_tile_from_dated_files(dated_files, out_path)
            print(f"  {progress(i)} {tid}: extracted ({len(dated_files)} date(s))")
        return [extracted[tid] for tid in tile_ids], all_labels

    to_submit = {}
    for tid, dated_files, out_path in jobs:
        if skip_existing and os.path.exists(out_path):
            extracted[tid] = out_path
        else:
            to_submit[tid] = (dated_files, out_path)

    done = len(extracted)
    if done:
        print(f"  [{done}/{total}] already extracted, skipping")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_one_tile_from_dated_files, dated_files, out_path): tid
            for tid, (dated_files, out_path) in to_submit.items()
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            extracted[tid] = fut.result()
            done += 1
            print(f"  {progress(done)} {tid}: extracted")

    return [extracted[tid] for tid in tile_ids], all_labels


# ---------------------------------------------------------------------
# Per-timestep mosaicking (memory-bounded) -- one output file per band,
# not one multiband file holding every date.
# ---------------------------------------------------------------------

def _mosaic_out_path(out_dir, vi, label, suffix):
    """f"{vi}_{label}{suffix}.tif", or f"{vi}{suffix}.tif" when label is ""
    (a lone undated band, e.g. a min/max raster -- see _resolve_band_labels())
    -- avoids a redundant/empty middle segment in that case."""
    stem = f"{vi}_{label}{suffix}" if label else f"{vi}{suffix}"
    return os.path.join(out_dir, f"{stem}.tif")


def _write_mosaic_band(out_path, mosaic_arr, transform, crs, label, nodata_value):
    """Writes one single-band mosaic (see mosaic_vi_tiles_by_date()), using
    the same compress="LZW" + predictor + tiled=True convention as
    mask_and_scale_rasters.scale_and_save_as_int_from_da() -- tiled output
    also matters for READ speed, not just write size: a later merge() (or
    any windowed read) of a tiled file can fetch just the blocks it needs
    instead of decompressing a whole row-strip, which is exactly what
    makes repeated mosaic reads slow on a non-tiled source."""
    with rasterio.open(out_path, "w", driver="GTiff", height=mosaic_arr.shape[1],
                        width=mosaic_arr.shape[2], count=1, dtype=mosaic_arr.dtype,
                        crs=crs, transform=transform, nodata=nodata_value,
                        compress="LZW", predictor=_predictor_for_dtype(mosaic_arr.dtype),
                        tiled=True) as dst:
        dst.write(mosaic_arr[0], 1)
        dst.set_band_description(1, label)


def _mosaic_one_band(tile_paths, band_idx, label, out_path, nodata_value):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor
    -- opens tile_paths itself (once each, since this call only ever
    handles ONE band, so there's no repeated-open cost to avoid here)."""
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, transform = merge_rasters(srcs, indexes=[band_idx], nodata=nodata_value)
    finally:
        for s in srcs:
            s.close()
    with rasterio.open(tile_paths[0]) as src0:
        crs = src0.crs
    _write_mosaic_band(out_path, mosaic_arr, transform, crs, label, nodata_value)
    return out_path


def mosaic_vi_tiles_by_date(tile_paths, out_dir, vi, start_date=None, end_date=None,
                            nodata_value=DEFAULT_NODATA_INT16, suffix="_MOSAIC", max_workers=1):
    """
    Mosaics every tile in tile_paths ONE BAND AT A TIME, writing a
    separate single-band output file per band instead of one large
    multiband file holding every band at once.

    Works against BOTH a multiband, per-band-dated archive (e.g.
    force_tsi_batch's TSI output, or normalize_vi_batch.normalize_vi_file()'s
    normalized output) -- one output file per calendar timestep, matching
    each band's own ISO date description -- AND a single-band file with
    no date information at all (e.g. a min/max raster, which
    stac_pull.save_minmax_rasters() writes with no band description) --
    one output file for that lone band, no date-based naming/filtering
    involved. See _resolve_band_labels() for exactly how a file's bands
    are mapped to labels in each case; start_date/end_date only apply
    (and only need to) when real per-band dates are actually present.

    This keeps peak memory bounded to a single band's worth of data
    across every tile, rather than every tile's ENTIRE band range at once
    (the previous approach -- merging every band of every tile in one
    rasterio.merge() call -- is a real memory hazard once tile_paths
    holds a full growing season x a national tile count). rasterio.merge()'s
    own `indexes` parameter is passed straight through to each source's
    read, so only the one requested band is ever actually read off disk
    per tile.

    Bands are completely independent of each other (each is its own
    separate merge + its own output file), so this is a plain
    embarrassingly-parallel job -- no dask/chunked-array machinery is
    needed here, just ProcessPoolExecutor (see max_workers), the same
    approach the rest of this module already uses for per-tile jobs.

    :param tile_paths: per-tile rasters (e.g. normalize_vi_tiles()'s own
        outputs, or a set of per-tile min/max rasters), all assumed to
        share the same band count/order (true for a single
        normalize_vi_tiles() call over one VI/year, or one VI's min/max
        rasters across tiles) -- band descriptions are read from
        tile_paths[0] only and reused for every other tile.
    :param start_date, end_date: optional 'YYYY-MM-DD' strings restricting
        which timesteps get mosaicked/exported -- ignored (with a notice)
        if tile_paths[0] has no per-band date descriptions to filter.
    :param suffix: appended (with the label) to build each output
        filename -- see _mosaic_out_path().
    :param max_workers: 1 (default) processes bands sequentially, opening
        every tile ONCE up front and keeping it open for every band --
        GDAL's per-dataset block cache means later bands can reuse
        already-decoded blocks instead of hitting disk again, so this is
        the cheaper choice on a single core. max_workers > 1 instead
        mosaics several bands concurrently in separate processes (each
        opens tile_paths itself); prefer this when I/O latency, not CPU,
        is the bottleneck (e.g. tiles on a network share) -- concurrent
        reads overlap that latency instead of paying it serially.
    :return: sorted list of output paths written, one per band.
    """
    if not tile_paths:
        raise ValueError("mosaic_vi_tiles_by_date: no tile raster paths given")

    with rasterio.open(tile_paths[0]) as src0:
        crs = src0.crs
    labels, band_indices = _resolve_band_labels(tile_paths[0], start_date, end_date)

    os.makedirs(out_dir, exist_ok=True)
    written = []

    if max_workers <= 1:
        srcs = [rasterio.open(p) for p in tile_paths]
        try:
            for label, band_idx in zip(labels, band_indices):
                out_path = _mosaic_out_path(out_dir, vi, label, suffix)
                print(f"  {label or vi}: mosaicking {len(tile_paths)} tile(s)...", flush=True)
                mosaic_arr, transform = merge_rasters(srcs, indexes=[band_idx], nodata=nodata_value)
                _write_mosaic_band(out_path, mosaic_arr, transform, crs, label, nodata_value)
                written.append(out_path)
                print(f"  {label or vi}: done -> {out_path}")
        finally:
            for s in srcs:
                s.close()
        return sorted(written)

    print(f"  Mosaicking {len(labels)} band(s) with max_workers={max_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for label, band_idx in zip(labels, band_indices):
            out_path = _mosaic_out_path(out_dir, vi, label, suffix)
            futures[executor.submit(_mosaic_one_band, tile_paths, band_idx, label,
                                     out_path, nodata_value)] = label
        for fut in as_completed(futures):
            label = futures[fut]
            out_path = fut.result()
            written.append(out_path)
            print(f"  {label or vi}: done -> {out_path}")

    return sorted(written)


# ---------------------------------------------------------------------
# Tile-directory processing, matching FORCE's one-folder-per-tile layout
# -- mirrors force_tsi_batch.run_tsi_tiles().
# ---------------------------------------------------------------------

def normalize_vi_tiles(root_dir, vi, minmax_dir, year, out_dir, file_token=None,
                        input_scale_factor=10000, scale_factor=10000, clip=True,
                        max_workers=1, skip_existing=True,
                        suffix="_NORM", vi_pattern_template=DEFAULT_VI_PATTERN_TEMPLATE,
                        minmax_template=DEFAULT_MINMAX_TEMPLATE,
                        tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include",
                        mosaic=True, mosaic_dir=None, mosaic_start_date=None, mosaic_end_date=None,
                        mosaic_max_workers=1):
    """
    Normalizes a single VI across every tile in a FORCE-tiled archive
    (root_dir/<tile_id>/<vi file>) against that SAME tile's own premade
    min/max raster pair (minmax_dir/<tile_id>/<vi>_min_<year>.tif and
    .../max -- see minmax_paths()). Output mirrors the input layout:
    out_dir/<tile_id>/<basename><suffix>.tif.

    A tile missing its VI file OR its min/max raster pair is skipped
    (printed as "not found", not a fatal error) -- every other tile
    still runs.

    :param vi: canonical VI name to process (e.g. "NDVI") -- one VI per
        call; run this again for other VIs if you need more than one.
    :param file_token: the token this archive's filenames actually use
        for `vi`, if different from the canonical name (e.g. 2018 FORCE
        naming uses "NDV" for canonical "NDVI" -- see force_pull.VI_KEYS
        / force_tsi_batch.run_tsi_tiles()'s vi_keys). None (default)
        uses `vi` itself.
    :param minmax_dir, year, minmax_template: locate each tile's premade
        min/max rasters via minmax_paths() -- see its docstring for the
        per-tile naming convention.
    :param input_scale_factor, scale_factor, clip: forwarded to
        normalize_vi_file() for every job -- see its docstring.
    :param tiles_csv, tiles_column, tiles_csv_mode: optional CSV
        restricting which tiles get processed -- see discover_tile_dirs().
    :param mosaic: if True (default), merge every tile's normalized output
        (both freshly written and pre-existing/skipped ones) into AOI-wide
        rasters afterward, via mosaic_vi_tiles_by_date() -- ONE band (one
        calendar timestep) at a time, written as ONE single-band file per
        timestep, keeping peak memory bounded to a single timestep's worth
        of data rather than every tile's entire date range at once.
        Skipped (with a message) if no tile actually has an output to
        mosaic. Every tile is expected to share the same band count/order
        (true for a single normalize_vi_tiles() call over one VI/year,
        since every tile was built from the same vi_pattern_template/year)
        -- see mosaic_vi_tiles_by_date()'s own caveat.
    :param mosaic_dir: where to write the mosaics. Defaults to
        out_dir/mosaic.
    :param mosaic_start_date, mosaic_end_date: optional 'YYYY-MM-DD'
        strings restricting which timesteps get mosaicked/exported --
        forwarded to mosaic_vi_tiles_by_date(). None (default) exports
        every date the normalized files hold. This only narrows what gets
        exported as a mosaic -- every date is still normalized per tile
        regardless of this setting.
    :param mosaic_max_workers: forwarded to mosaic_vi_tiles_by_date() --
        1 (default) mosaics dates sequentially (opening every tile once,
        reused across dates); > 1 mosaics several dates concurrently in
        separate processes instead. See its own docstring for when each
        is the better choice.
    :return: {tile_id: (in_path, out_path, status)}
    """
    call_params = dict(locals())  # snapshot of every argument -- see _write_run_params()

    file_token = file_token or vi
    tile_dirs = discover_tile_dirs(root_dir, tiles_csv, tiles_column, tiles_csv_mode)
    print(f"Found {len(tile_dirs)} tile(s) under {root_dir}")

    job_specs = []
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        tile_out_dir = os.path.join(out_dir, tile_id)
        os.makedirs(tile_out_dir, exist_ok=True)

        try:
            in_path = find_tile_vi_file(tile_dir, file_token, vi_pattern_template, year=year)
        except FileNotFoundError as e:
            print(f"  {tile_id}/{vi}: not found -- {e}")
            continue

        min_path, max_path = minmax_paths(minmax_dir, tile_id, vi, year, minmax_template)
        if not (os.path.exists(min_path) and os.path.exists(max_path)):
            print(f"  {tile_id}/{vi}: min/max raster(s) not found -- {min_path}, {max_path}")
            continue

        base, ext = os.path.splitext(os.path.basename(in_path))
        out_path = os.path.join(tile_out_dir, f"{base}{suffix}{ext}")
        job_specs.append((in_path, out_path, min_path, max_path, tile_id))

    start = time.monotonic()
    results_by_key = _execute_jobs(
        job_specs, input_scale_factor, scale_factor, clip, skip_existing, max_workers,
        label_fn=lambda key: f"{key}/{vi}",
    )
    summary = _print_summary([status for _, _, status in results_by_key.values()], time.monotonic() - start)
    _write_run_params(out_dir, "normalize_vi_batch.normalize_vi_tiles", call_params, summary)

    if mosaic:
        tile_out_paths = [out_path for _, out_path, status in results_by_key.values()
                           if status in ("written", "skipped")]
        if tile_out_paths:
            target_mosaic_dir = mosaic_dir or os.path.join(out_dir, "mosaic")
            mosaic_nodata = DEFAULT_NODATA_INT16 if scale_factor is not None else np.nan
            print(f"\nMosaicking {len(tile_out_paths)} tile(s) into {target_mosaic_dir} "
                  f"(one file per timestep)")
            mosaic_vi_tiles_by_date(
                tile_out_paths, target_mosaic_dir, vi,
                start_date=mosaic_start_date, end_date=mosaic_end_date,
                nodata_value=mosaic_nodata, suffix=f"{suffix}_MOSAIC",
                max_workers=mosaic_max_workers,
            )
        else:
            print("\nNo tile outputs available to mosaic.")

    return results_by_key


# Example (edit paths/params below and just run this file for a single VI,
# e.g. NDVI -- mirrors force_tsi_batch.py's own bottom-of-file example):
#
if __name__ == "__main__":
    vi_to_process = ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]
    file_tokens_2026 = {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}
    for vi in vi_to_process:
        results = normalize_vi_tiles(
            root_dir=r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\level4_rbf\2026",
            vi=vi,
            file_token=file_tokens_2026[vi],
            minmax_dir=r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\minmax\2026",  # holds one <tile_id>/ subfolder per tile
            year=2026,
            out_dir=r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\level5_norm\2026",
            input_scale_factor=None,  # divides raw TSI pixel values before normalizing
            scale_factor=10000,   # None to write raw float32 [0, 1] values instead
            clip=True,            # clip normalized values to [0, 1]
            max_workers=4,
            skip_existing=True,   # rerun-safe: only computes tiles without output yet
            tiles_csv=None,        # e.g. CH_FORCE_Grids.csv (column "Tile_ID") to restrict tiles
            mosaic=False,               # one single-band mosaic file per timestep, memory-bounded
            # mosaic_dir=None,         # defaults to out_dir/mosaic
            mosaic_start_date="2026-06-01",  # only export/mosaic timesteps in this window
            mosaic_end_date="2026-08-14",    # None/None (default) exports every date
        )

        for tile_id, (in_path, out_path, status) in results.items():
            if status != "written":
                print(f"{tile_id}/{vi}: {status}")

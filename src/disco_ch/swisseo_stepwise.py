"""
swisseo_stepwise.py

Incremental force_tsi.py-based despike + RBF interpolation for the
swisseo/STAC per-VI archive -- the same idea as force_tsi_stepwise.py,
just reading from that archive's directory-of-per-date-files layout
instead of a premade FORCE multiband TSS TIF.

WHY THIS EXISTS
----------------
rbf_interp.py used to be the one module that handled BOTH FORCE (premade
multiband TSS TIF) and swisseo (archive_root/<VI>/<date>.tif, one file
per newly-arrived scene) inputs, via its own multi-width ensemble +
isolated-point-fallback interpolation. The force_tsi.py family
(force_tsi.py/force_tsi_batch.py/force_tsi_stepwise.py) replaced that
with a simpler, param-file-matching despike()/rbf_interpolate() -- but
so far only for FORCE's input layout. This module is the swisseo-input
counterpart: same force_tsi.py compute (despike, rbf_interpolate, same
toggles), same force_tsi_stepwise.py incremental/eligibility/merge
design, just a different READ side.

READING THE SWISSEO ARCHIVE
------------------------------
Modeled directly on rbf_interp.py's own archive readers
(load_vi_cube_from_archive() / load_vi_cube_from_archive_window() /
archive_grid_meta()) and stac_pull.archive_vi_raster()'s write
convention: one directory per VI (archive_root/<VI>/), one single-band
file per calendar date (<ISO date>.tif, e.g. "2026-03-15.tif"), same-day
duplicates already averaged together at write time. Dates come from
FILENAMES here, not band descriptions (there's only one band per file,
unlike FORCE's premade multiband TIF) -- see _dates_from_archive() /
load_vi_cube_from_archive(). Once read into (dates, daily_cube), the
shape is identical to what force_tsi.load_tss() returns, so
despike()/rbf_interpolate() (and force_tsi_batch's weight-caching
_precompute_rbf_weights()/_apply_rbf_weights(), reused as-is) work
completely unchanged.

There's no tile-directory structure here (swisseo's archive is one
national/AOI-extent raster per date, not FORCE's 30km-tile layout), so
there's no discover_tile_dirs()/tiles_csv/year/vi_pattern_template
equivalent -- `vi_keys` is just a plain list of VI names, and each one's
archive directory is archive_root/<vi_key>.

OUTPUT STAYS FLOAT32, NOT INT16
----------------------------------
force_tsi_batch.py/force_tsi_stepwise.py write FORCE output as int16
because FORCE's OWN source files already are int16 with an implicit
x10000-ish scale (see force_tsi_batch.py's module docstring point 7) --
rounding the interpolated result back to that same already-established
integer scale is lossless. Swisseo's archive has no such convention:
archive_vi_raster() writes genuine float32 VI values with no scale
factor at all (see stac_pull.py). Quantizing that straight to int16
without a real, deliberately-chosen scale factor would silently destroy
most of its precision (e.g. an NDVI of 0.62 rounds to "1"), so this
module keeps the source's own float32 precision on the way out instead
of guessing a scale factor on your behalf. If you later want the same
disk-space win FORCE gets, that would mean explicitly choosing and
documenting a scale factor for each VI -- worth a deliberate follow-up
if file size becomes a problem, not a silent default here.

SAME TOGGLES AS force_tsi_stepwise.py
----------------------------------------
wait_days/eligible_output_dates()/recommended_wait_days() (imported
directly, unchanged), rbf_sigma/rbf_cutoff/above_noise/below_noise,
despike (skips force_tsi.despike() entirely when False), chunk_size
(bounds READ+COMPUTE memory; rarely needed in steady state, same
reasoning as force_tsi_stepwise.py), blas_threads/verbose_chunks, and a
timestamped run-parameters JSON record (force_tsi_batch._write_run_params(),
reused directly) written into `out_dir`.

Usage (from a notebook, e.g. on a schedule):

    from datetime import date
    from src.disco_ch.swisseo_stepwise import run_swisseo_stepwise

    vi_keys = ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]

    results = run_swisseo_stepwise(
        r"B:\\bloomc\\DiscoCH_2026_08_03\\swisseo\\archive",
        vi_keys,
        out_dir=r"B:\\bloomc\\DiscoCH_2026_08_03\\swisseo\\level4_rbf",
        date_range=(date(2026, 3, 1), date(2026, 10, 31)),
        doy_range=(1, 365), int_day=5, wait_days=10,
        rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
        above_noise=3.0, below_noise=1.0, despike=True,
        chunk_size=None, max_workers=4,
    )
    # Run again later (e.g. tomorrow, once more scenes have been archived):
    # only the newly-eligible dates get computed and merged in.
"""

import os
import glob
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import Window

from src.disco_ch.force_tsi import (
    despike as despike_fn,
    output_dates_from_range,
)
from src.disco_ch.force_tsi_batch import (
    _iter_chunk_windows,
    _blas_limiter,
    _precompute_rbf_weights,
    _apply_rbf_weights,
    _format_duration,
    _write_run_params,
)
from src.disco_ch.force_tsi_stepwise import (
    DEFAULT_WAIT_DAYS,
    eligible_output_dates,
    recommended_wait_days,
    _existing_output_dates,
    _read_existing_output,
)


# ---------------------------------------------------------------------
# Reading a swisseo per-VI archive directory (archive_root/<VI>/<date>.tif)
# ---------------------------------------------------------------------

def _dates_from_archive(archive_dir):
    """Acquisition-date calendar for a swisseo per-VI archive directory,
    parsed from filenames alone -- no file is even opened, unlike
    force_tsi_batch._dates_from_tss() (which still has to open the TIF
    for its band descriptions), since swisseo's one-file-per-date layout
    already encodes each date in its filename. Empty list if the
    directory doesn't exist yet (e.g. no scenes archived for this VI at
    all so far)."""
    if not os.path.isdir(archive_dir):
        return []
    dates = []
    for fname in os.listdir(archive_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".tif":
            continue
        try:
            dates.append(datetime.strptime(stem, "%Y-%m-%d").date())
        except ValueError:
            continue
    return sorted(dates)


def load_vi_cube_from_archive(archive_dir):
    """Reads a swisseo per-VI archive directory (one single-band file per
    date -- see stac_pull.archive_vi_raster()) into (dates, daily_cube,
    transform, crs), the same shape force_tsi.load_tss() returns from a
    premade FORCE multiband TSS TIF -- modeled directly on
    rbf_interp.load_vi_cube_from_archive()."""
    paths = sorted(glob.glob(os.path.join(archive_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    dates = []
    arrays = []
    transform = None
    crs = None
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            d = datetime.strptime(stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True).astype("float32").filled(np.nan)
            if transform is None:
                transform = src.transform
                crs = src.crs
        dates.append(d)
        arrays.append(arr)

    if not dates:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    daily_cube = np.stack(arrays, axis=0)
    return dates, daily_cube, transform, crs


def load_vi_pixel_series_from_archive(archive_dir, row, col, buffer=0):
    """Like load_vi_cube_from_archive(), but only reads a small window
    around (row, col) from each per-date file -- the fast path for
    single-pixel diagnostics (see swisseo_plot.py). Modeled directly on
    rbf_interp.load_vi_pixel_series_from_archive().

    Returns dates, daily_cube (windowed), and pixel row/col within that
    window (0, 0 for buffer=0, since the window edge may have been
    clamped to the raster bounds near an edge/corner pixel).
    """
    paths = sorted(glob.glob(os.path.join(archive_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    # All archived dates share the same grid (see stac_pull.archive_vi_raster()),
    # so the window only needs computing once, from the first file.
    with rasterio.open(paths[0]) as first_src:
        row_off = max(row - buffer, 0)
        col_off = max(col - buffer, 0)
        row_end = min(row + buffer + 1, first_src.height)
        col_end = min(col + buffer + 1, first_src.width)
    window = Window(col_off=col_off, row_off=row_off,
                     width=col_end - col_off, height=row_end - row_off)

    dates = []
    arrays = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            d = datetime.strptime(stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        with rasterio.open(path) as src:
            arr = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        dates.append(d)
        arrays.append(arr)

    if not dates:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    daily_cube = np.stack(arrays, axis=0)
    return dates, daily_cube, row - row_off, col - col_off


def load_vi_cube_from_archive_window(archive_dir, window):
    """Like load_vi_cube_from_archive(), but only reads the given spatial
    `window` from each per-date file -- the read side of chunked
    processing (see chunk_size in run_tsi_stepwise_archive()). Modeled
    directly on rbf_interp.load_vi_cube_from_archive_window()."""
    paths = sorted(glob.glob(os.path.join(archive_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    dates = []
    arrays = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            d = datetime.strptime(stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        with rasterio.open(path) as src:
            arr = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        dates.append(d)
        arrays.append(arr)

    if not dates:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    daily_cube = np.stack(arrays, axis=0)
    return dates, daily_cube


def archive_grid_meta(archive_dir):
    """(width, height, transform, crs) for a swisseo per-VI archive
    directory, from its first dated file, without reading any pixel
    data -- used to size chunk_size windows. Modeled directly on
    rbf_interp.archive_grid_meta()."""
    paths = sorted(glob.glob(os.path.join(archive_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")
    with rasterio.open(paths[0]) as src:
        return src.width, src.height, src.transform, src.crs


# ---------------------------------------------------------------------
# Output: float32, merged into an existing file -- see module docstring
# ---------------------------------------------------------------------

def _write_float32_stack(path, stack, transform, crs, dates, nodata_value=np.nan):
    """Writes an already-float32 (n, h, w) stack to a temporary path,
    then atomically swaps it into place (os.replace()) -- same
    crash-safety rationale as force_tsi_stepwise._write_int16_stack(),
    just float32/NaN instead of int16 (see module docstring)."""
    n, h, w = stack.shape
    tmp_path = path + ".tmp"
    with rasterio.open(tmp_path, "w", driver="GTiff", height=h, width=w, count=n,
                        dtype="float32", crs=crs, transform=transform,
                        nodata=nodata_value, compress="deflate") as dst:
        for i in range(n):
            dst.write(stack[i].astype("float32"), i + 1)
            dst.set_band_description(i + 1, dates[i].isoformat())
    os.replace(tmp_path, path)


def _merge_and_write_output(out_path, new_dates, new_stack, transform, crs, nodata_value=np.nan):
    """Combines whatever's already in out_path (read back unchanged, via
    force_tsi_stepwise._read_existing_output() -- dtype-agnostic, so it
    works here despite this module using float32 instead of int16) with
    newly computed bands, and (re)writes the union sorted by date. See
    force_tsi_stepwise.py's MERGING INTO AN EXISTING FILE docstring
    section for why this has to rewrite the whole file."""
    combined = {}
    if os.path.exists(out_path):
        existing_dates, existing_arr, transform, crs, existing_nodata = _read_existing_output(out_path)
        if existing_nodata is not None:
            nodata_value = existing_nodata
        combined.update(zip(existing_dates, existing_arr))
    combined.update(zip(new_dates, new_stack.astype("float32")))

    all_dates = sorted(combined)
    stack = np.stack([combined[d] for d in all_dates])
    _write_float32_stack(out_path, stack, transform, crs, all_dates, nodata_value)


# ---------------------------------------------------------------------
# Single-VI incremental update
# ---------------------------------------------------------------------

def run_tsi_stepwise_archive(archive_dir, out_path, date_range, doy_range=(1, 365), int_day=5,
                              wait_days=DEFAULT_WAIT_DAYS,
                              rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                              above_noise=3.0, below_noise=1.0, chunk_size=None,
                              blas_threads=1, despike=True, verbose_chunks=False,
                              nodata_value=np.nan):
    """Incrementally updates one VI's stepwise output at out_path from a
    swisseo per-VI archive directory -- same eligibility/merge logic as
    force_tsi_stepwise.run_tsi_stepwise(), just reading archive_dir (see
    load_vi_cube_from_archive()) instead of a premade FORCE TSS TIF.

    :param archive_dir: one VI's swisseo archive directory
        (archive_root/<vi_key>, see stac_pull.archive_vi_raster()).
    :param wait_days: see force_tsi_stepwise.py's module docstring --
        an output date isn't eligible until archive_dir has real data at
        least this many days past it. 0 disables the gate.
    :param chunk_size: see force_tsi_batch.run_tsi_chunked() -- bounds
        the READ+COMPUTE memory for the dates being computed THIS call.
    :return: (new_dates, status) -- same status strings as
        force_tsi_stepwise.run_tsi_stepwise(): "no output dates in range",
        "no data", "not eligible yet", "up to date", or "written".
    """
    output_dates = output_dates_from_range(date_range, int_day, doy_range)
    if not output_dates:
        return [], "no output dates in range"

    all_dates = _dates_from_archive(archive_dir)
    if not all_dates:
        return [], "no data"
    last_raw_date = max(all_dates)

    eligible = eligible_output_dates(output_dates, last_raw_date, wait_days)
    if not eligible:
        return [], "not eligible yet"

    existing = _existing_output_dates(out_path) or set()
    new_dates = [d for d in eligible if d not in existing]
    if not new_dates:
        return [], "up to date"

    new_stack, transform, crs = interpolate_archive_dates(
        archive_dir, new_dates, rbf_sigma=rbf_sigma, rbf_cutoff=rbf_cutoff,
        above_noise=above_noise, below_noise=below_noise, chunk_size=chunk_size,
        blas_threads=blas_threads, despike=despike, verbose_chunks=verbose_chunks,
    )

    _merge_and_write_output(out_path, new_dates, new_stack, transform, crs, nodata_value)
    return new_dates, "written"


def interpolate_archive_dates(archive_dir, output_dates, rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                               above_noise=3.0, below_noise=1.0, chunk_size=None,
                               blas_threads=1, despike=True, verbose_chunks=False):
    """Computes despike() + the RBF ensemble for EXACTLY the given
    `output_dates` against a swisseo per-VI archive directory -- no
    eligibility/merge logic at all (see run_tsi_stepwise_archive(), which
    is just this plus that bookkeeping on top).

    For callers that track their own "what's new" bookkeeping instead of
    inspecting an output file's existing bands -- e.g.
    stac_pull.update_vi_min_max_interpolated(), which tracks it via its
    own processed_dates metadata (used for min/max + the disco model) --
    so the exact same computed values can also be handed to
    _merge_and_write_output() to persist them, without computing twice.

    :return: (stack, transform, crs) -- stack is (len(output_dates), h, w)
        float32, one 2D array per requested output date, in the same
        order as `output_dates`.
    """
    all_dates = _dates_from_archive(archive_dir)
    if not all_dates:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")

    precomputed = _precompute_rbf_weights(all_dates, output_dates, rbf_sigma, rbf_cutoff)

    if chunk_size is None:
        with _blas_limiter(blas_threads):
            dates, cube, transform, crs = load_vi_cube_from_archive(archive_dir)
            if despike:
                cleaned, _removed = despike_fn(dates, cube, above_noise, below_noise)
            else:
                cleaned = cube
            stack = _apply_rbf_weights(cleaned, precomputed)
        return stack, transform, crs

    width, height, transform, crs = archive_grid_meta(archive_dir)
    windows = list(_iter_chunk_windows(width, height, chunk_size))
    total_chunks = len(windows)
    label = os.path.basename(os.path.normpath(archive_dir))
    chunk_start = time.monotonic()

    stack = np.full((len(output_dates), height, width), np.nan, dtype="float32")
    for chunk_i, window in enumerate(windows, start=1):
        chunk_dates, cube = load_vi_cube_from_archive_window(archive_dir, window)
        with _blas_limiter(blas_threads):
            if despike:
                cleaned, _removed = despike_fn(chunk_dates, cube, above_noise, below_noise)
            else:
                cleaned = cube
            sub_stack = _apply_rbf_weights(cleaned, precomputed)
        row0, col0 = window.row_off, window.col_off
        stack[:, row0:row0 + window.height, col0:col0 + window.width] = sub_stack

        if verbose_chunks:
            elapsed = time.monotonic() - chunk_start
            eta = (f", ~{_format_duration(elapsed / chunk_i * (total_chunks - chunk_i))} remaining"
                   if chunk_i < total_chunks else "")
            print(f"    [{label}] chunk {chunk_i}/{total_chunks}, {_format_duration(elapsed)} elapsed{eta}", flush=True)

    return stack, transform, crs


# ---------------------------------------------------------------------
# Multi-VI batch: every VI in the swisseo archive, incrementally
# ---------------------------------------------------------------------

def _run_one_swisseo_stepwise(archive_dir, out_path, date_range, doy_range, int_day, wait_days,
                               rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                               blas_threads, despike, verbose_chunks, nodata_value):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor.
    Catches its own exceptions -- a bad/missing/corrupt archive becomes a
    "failed: ..." status rather than aborting the whole batch."""
    try:
        new_dates, status = run_tsi_stepwise_archive(
            archive_dir, out_path, date_range, doy_range, int_day, wait_days,
            rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
            blas_threads, despike, verbose_chunks, nodata_value,
        )
    except Exception as e:
        return archive_dir, out_path, f"failed: {type(e).__name__}: {e}"
    if status == "written":
        return archive_dir, out_path, f"written ({len(new_dates)} date(s))"
    return archive_dir, out_path, status


def run_swisseo_stepwise(archive_root, vi_keys, out_dir, date_range, doy_range=(1, 365), int_day=5,
                          wait_days=DEFAULT_WAIT_DAYS,
                          rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                          above_noise=3.0, below_noise=1.0, chunk_size=None,
                          max_workers=1, suffix="_TSI",
                          blas_threads=1, despike=True, verbose_chunks=False,
                          nodata_value=np.nan):
    """Runs run_tsi_stepwise_archive() over every VI in a swisseo archive
    -- archive_root/<vi_key>/<date>.tif in, out_dir/<vi_key><suffix>.tif
    out. No tile-directory structure here (see module docstring) -- one
    output file per VI, directly in out_dir.

    Call this again later (e.g. on a schedule, as more scenes get
    archived) and only newly-eligible dates get computed and merged into
    each existing output file -- same incremental design as
    force_tsi_stepwise.run_tsi_tiles_stepwise().

    :param vi_keys: plain list of VI names, e.g.
        ["NDVI", "EVI", "NDMI", "CIRE", "CCI"] -- matching
        stac_pull.py's own vi_keys convention (unlike FORCE's
        {canonical_name: file_token} mapping, swisseo's archive
        directories are already named by canonical VI key).
    :param wait_days, chunk_size, despike: see run_tsi_stepwise_archive().
    :param max_workers: VIs are independent of each other; >1 processes
        them concurrently via ProcessPoolExecutor.
    :return: {vi_key: (archive_dir, out_path, status)} -- status is
        "up to date", "not eligible yet", "no data", "written (N date(s))",
        or "failed: ...".
    """
    call_params = dict(locals())  # snapshot of every argument -- see force_tsi_batch._write_run_params()

    os.makedirs(out_dir, exist_ok=True)

    job_specs = []
    for vi_key in vi_keys:
        archive_dir = os.path.join(archive_root, vi_key)
        out_path = os.path.join(out_dir, f"{vi_key}{suffix}.tif")
        job_specs.append((archive_dir, out_path, vi_key))

    total = len(job_specs)
    if total == 0:
        print("Nothing to do (0 VIs).")
        return {}

    start = time.monotonic()

    def progress(done):
        elapsed = time.monotonic() - start
        eta = f", ~{_format_duration(elapsed / done * (total - done))} remaining" if 0 < done < total else ""
        return f"[{done}/{total}, {_format_duration(elapsed)} elapsed{eta}]"

    results_by_key = {}
    if max_workers <= 1:
        for i, (archive_dir, out_path, vi_key) in enumerate(job_specs, start=1):
            result = _run_one_swisseo_stepwise(
                archive_dir, out_path, date_range, doy_range, int_day, wait_days,
                rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                blas_threads, despike, verbose_chunks, nodata_value,
            )
            print(f"  {progress(i)} {vi_key}: {result[2]}")
            results_by_key[vi_key] = result
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_swisseo_stepwise, archive_dir, out_path, date_range, doy_range, int_day,
                    wait_days, rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                    blas_threads, despike, verbose_chunks, nodata_value,
                ): vi_key
                for archive_dir, out_path, vi_key in job_specs
            }
            for fut in as_completed(futures):
                vi_key = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    archive_dir = os.path.join(archive_root, vi_key)
                    out_path = os.path.join(out_dir, f"{vi_key}{suffix}.tif")
                    result = (archive_dir, out_path, f"failed: {type(e).__name__}: {e}")
                done += 1
                print(f"  {progress(done)} {vi_key}: {result[2]}")
                results_by_key[vi_key] = result

    written = sum(1 for _, _, s in results_by_key.values() if s.startswith("written"))
    up_to_date = sum(1 for _, _, s in results_by_key.values() if s == "up to date")
    pending = sum(1 for _, _, s in results_by_key.values() if s == "not eligible yet")
    no_data = sum(1 for _, _, s in results_by_key.values() if s == "no data")
    failed = sum(1 for _, _, s in results_by_key.values() if s.startswith("failed"))
    elapsed = time.monotonic() - start
    print(f"Done: {written} updated, {up_to_date} up to date, {pending} not eligible yet, "
          f"{no_data} no data, {failed} failed (of {total}) in {_format_duration(elapsed)}")

    summary = {"updated": written, "up_to_date": up_to_date, "not_eligible_yet": pending,
               "no_data": no_data, "failed": failed, "total": total, "elapsed_seconds": elapsed}
    _write_run_params(out_dir, "swisseo_stepwise.run_swisseo_stepwise", call_params, summary)

    return {vi_key: result for vi_key, result in results_by_key.items()}

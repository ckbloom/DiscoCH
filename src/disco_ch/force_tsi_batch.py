"""
force_tsi_batch.py

Applies force_tsi.py's despike() + rbf_interpolate() to real FORCE data
efficiently. Three things make a real run different from calling
run_tsi() once on a small test file:

  1. Memory: a real FORCE tile (e.g. 3000x3000 px, a full growing
     season of Sentinel-2 dates) can be tens of GB once you're holding
     the raw cube, the despiked cube, and several despike() working
     arrays at once. run_tsi_chunked() processes the raster in spatial
     blocks instead, writing each block straight to disk as it's
     computed -- this is exactly what the param file's own CHUNK_SIZE
     parameter (under PARALLEL PROCESSING) exists for.

  2. Volume/layout: a FORCE archive is laid out as one directory per
     tile (e.g. root_dir/X0068_Y0043/), each holding one TSS file per
     VI. run_tsi_tiles() discovers those tile directories, finds each
     VI's file within each tile (using a {canonical_name: file_token}
     map, since different years abbreviate VI names differently -- see
     find_tile_vi_file()), and runs run_tsi_chunked() over every
     (tile, VI) pair -- optionally in parallel, since they're all
     independent of each other -- mirroring the original codebase's
     discover_tiles()/find_vi_file()/_process_one_tile().

  3. Selection: run_tsi_tiles() optionally takes a CSV restricting which
     tiles get processed (tiles_csv). Default mode is an ALLOW list
     (matching the original codebase's grid_csv, e.g. CH_FORCE_Grids.csv)
     -- only tiles named in the CSV are processed. tiles_csv_mode="exclude"
     inverts that into a skip list instead.

  4. Threading: rbf_interpolate()'s np.tensordot calls are BLAS-backed,
     and BLAS defaults to spinning up threads across every core on the
     machine for EACH call, with no awareness of how many worker
     processes (max_workers) are already running. With chunk_size
     splitting one (tile, VI) job into many small tensordot calls (one
     per output date, per chunk), that per-call thread spin-up/teardown
     overhead -- repeated hundreds or thousands of times -- can dominate
     runtime and produce exactly "CPU pegged, almost no progress."
     run_tsi_chunked()'s blas_threads parameter (default 1, via
     threadpoolctl if installed) pins each call to a single BLAS thread,
     so max_workers is the only source of parallelism and workers don't
     fight each other or themselves.

  5. Print visibility under multiprocessing: verbose_chunks's per-chunk
     line, and _blas_limiter()'s threadpoolctl warning, run inside a
     WORKER process when max_workers > 1. A worker's stdout is a pipe,
     not a real terminal, so Python defaults to full block buffering
     there (unlike the parent process's stdout, e.g. a notebook cell,
     which flushes normally) -- without an explicit flush, those lines
     can sit unseen for a long time (or until the worker exits) even
     though the run is progressing normally. Both prints pass
     flush=True specifically to avoid this.

  6. Redundant per-chunk weight computation: force_tsi.rbf_interpolate()'s
     per-output-date candidate-index/weight arrays depend only on the
     acquisition calendar and output_dates -- NOT on which spatial chunk
     is being processed -- but calling rbf_interpolate() itself recomputes
     them from scratch inside every chunk of every file (e.g. 36x over
     for a 3000x3000 tile at chunk_size=512). _precompute_rbf_weights()/
     _apply_rbf_weights() factor that out: computed ONCE per file, then
     just applied per chunk. This is arithmetically IDENTICAL to calling
     rbf_interpolate() each time (same formula, same float64 precision,
     bit-for-bit the same result) -- deliberately kept float64 rather
     than also switching to float32, so this change in isolation shows
     whether the redundant recomputation itself was the actual
     bottleneck.

  7. Output size: the source TSS files are int16 with no GDAL scale/
     offset tag set -- the x10000-ish scaling spectral indices use is
     pure convention, not something GDAL records. force_tsi.rbf_interpolate()
     (and this module) never divides that back out, so the values being
     interpolated already ARE that same scaled integer range -- just
     carried in float32/float64. Writing the result back out as float32
     doubles the raw bytes/pixel over the source AND, more importantly,
     destroys deflate's compressibility: a weighted average's continuous
     mantissa noise barely compresses at all, vs. the source's heavily
     redundant quantized integers. _quantize_int16()/_write_tsi_int16()
     round back to the nearest integer and write int16 (with a -9999
     nodata sentinel for NaN, matching the source's own convention)
     instead -- same information content (a x10000-scaled index only
     ever carried ~4-5 significant digits anyway), a fraction of the
     disk space, and less write/compression time.

despike() is still called exactly as force_tsi.run_tsi() calls it. This
only changes how the compute is scheduled, how memory is used, how the
output is stored on disk, and which files get discovered.

  8. Run-parameters record: run_tsi_tiles() (and
     force_tsi_stepwise.run_tsi_tiles_stepwise()) writes a timestamped
     JSON file -- see _write_run_params() -- into `out_dir` itself (the
     directory holding the <tile_id> subfolders, not any one tile's own
     folder) recording every parameter that call was made with, plus a
     short result summary. Timestamped rather than a fixed name so a
     batch backfill followed later by one or more stepwise runs each
     leave their own record instead of overwriting the last one --
     answering "what parameters actually produced the data in this
     folder?" without having to remember or dig through notebook history.
"""

import os
import csv
import glob
import json
import time
import contextlib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from src.disco_ch.force_tsi import (
    load_tss,
    load_tss_window,
    tss_grid_meta,
    despike as despike_fn,
    rbf_cutoff_radius,
    output_dates_from_range,
    DATE_RE,
)


# ---------------------------------------------------------------------
# Single-file, memory-bounded processing
# ---------------------------------------------------------------------

def _iter_chunk_windows(width, height, chunk_size):
    """Yield Windows tiling a (height, width) raster into chunk_size x
    chunk_size spatial blocks (edge blocks are smaller), row-major."""
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


_warned_missing_threadpoolctl = False


def _blas_limiter(blas_threads):
    """Context manager capping BLAS (np.tensordot, in _apply_rbf_weights() --
    same math as rbf_interpolate()) to `blas_threads` threads per call --
    see the module docstring's point 4. blas_threads=None leaves BLAS's own default behavior alone
    (only useful for comparison/debugging). Falls back to a no-op (with
    a one-time warning per process) if threadpoolctl isn't installed --
    the fix still works, just needs the pip install to take effect."""
    global _warned_missing_threadpoolctl
    if blas_threads is None:
        return contextlib.nullcontext()
    if threadpool_limits is None:
        if not _warned_missing_threadpoolctl:
            print("  (threadpoolctl not installed -- BLAS thread capping is disabled; "
                  "`pip install threadpoolctl` to fix multi-worker/chunked slowdowns)", flush=True)
            _warned_missing_threadpoolctl = True
        return contextlib.nullcontext()
    return threadpool_limits(limits=blas_threads)


# ---------------------------------------------------------------------
# RBF weight precomputation -- see module docstring point 6
# ---------------------------------------------------------------------

def _dates_from_tss(tss_path):
    """The acquisition-date calendar for a TSS file, parsed from band
    descriptions only (matching force_tsi._collapse_bands_to_daily()'s
    date parsing) -- no pixel data is read. Every window of a TSS file
    shares this exact same calendar (dates come from band metadata, not
    which pixels happen to be valid), so this only needs to run once per
    file, before any chunk is touched, to precompute that file's RBF
    weights (see _precompute_rbf_weights())."""
    with rasterio.open(tss_path) as src:
        descriptions = src.descriptions
    dates = []
    for desc in descriptions:
        m = DATE_RE.match(desc) if desc else None
        if not m:
            raise ValueError(f"Could not parse a YYYYMMDD date from band description: {desc!r}")
        dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
    return sorted(set(dates))


def _precompute_rbf_weights(dates, output_dates, rbf_sigma, rbf_cutoff):
    """Same math as force_tsi.rbf_interpolate()'s per-output-date weight
    construction (same float64 precision too -- deliberately not switched
    to float32, see module docstring point 6), computed ONCE and reused
    across every spatial chunk of a file instead of being recomputed
    inside each one.

    :return: list of (idx, weight) pairs, one per output date -- idx: int
        array of candidate acquisition-date positions; weight: (n_sigma,
        len(idx)) float64 array, or (idx, None) if nothing falls within
        any sigma's cutoff radius of that output date.
    """
    ordinals = np.array([d.toordinal() for d in dates], dtype="float64")
    sigmas = np.asarray(rbf_sigma, dtype="float64")
    radii = np.array([rbf_cutoff_radius(s, rbf_cutoff) for s in sigmas])
    max_radius = radii.max() if radii.size else 0.0

    precomputed = []
    for od in output_dates:
        target = od.toordinal()
        idx = np.where(np.abs(ordinals - target) <= max_radius)[0]
        if idx.size == 0:
            precomputed.append((idx, None))
            continue

        deltas = ordinals[idx] - target
        weight = np.exp(-0.5 * (deltas[None, :] / sigmas[:, None]) ** 2)
        within = np.abs(deltas)[None, :] <= radii[:, None]
        weight = np.where(within, weight, 0.0)
        precomputed.append((idx, weight))
    return precomputed


def _apply_rbf_weights(cube, precomputed):
    """Applies a file's already-precomputed (idx, weight) pairs (see
    _precompute_rbf_weights()) to one chunk's raw (n_dates, h, w) cube --
    identical data-density-weighted combination, identical float64
    precision, to force_tsi.rbf_interpolate()."""
    valid = ~np.isnan(cube)
    valid_f = valid.astype("float64")
    filled = np.where(valid, cube, 0.0).astype("float64")

    h, w = cube.shape[1:]
    out = np.full((len(precomputed), h, w), np.nan, dtype="float32")
    for oi, (idx, weight) in enumerate(precomputed):
        if weight is None:
            continue
        mass = np.tensordot(weight, valid_f[idx], axes=([1], [0]))
        wsum = np.tensordot(weight, filled[idx], axes=([1], [0]))
        total_mass = mass.sum(axis=0)
        total_wsum = wsum.sum(axis=0)
        has_data = total_mass > 0
        out[oi] = np.where(has_data, total_wsum / np.where(has_data, total_mass, 1.0), np.nan)
    return out


# ---------------------------------------------------------------------
# int16 output -- see module docstring point 7
# ---------------------------------------------------------------------

DEFAULT_NODATA_INT16 = -9999


def _quantize_int16(arr, nodata_value=DEFAULT_NODATA_INT16):
    """Rounds a float (n, h, w) array to the nearest integer and casts
    to int16 -- np.round(), not a bare cast, so values round to the
    nearest integer instead of truncating toward zero (which would bias
    every result slightly low). NaN is mapped to `nodata_value` BEFORE
    the cast (NaN has no int16 representation -- casting it directly is
    undefined/platform-dependent). Finite values are clipped to int16's
    range as a safety net: an RBF-ensemble value is a weighted AVERAGE of
    the source's own already-int16-safe values, so it's mathematically
    bounded by whatever contributed to it and should never actually need
    clipping -- this just guards against floating-point boundary noise.
    """
    rounded = np.round(arr)
    quantized = np.where(np.isnan(rounded), nodata_value, np.clip(rounded, -32768, 32767))
    return quantized.astype("int16")


def _write_tsi_int16(path, stack, transform, crs, dates, nodata_value=DEFAULT_NODATA_INT16):
    """Writes an RBF ensemble stack (float, (n, h, w)) as scaled int16
    instead of force_tsi.write_tsi()'s float32 -- see _quantize_int16()
    and module docstring point 7 for why this is lossless for this data
    and dramatically smaller on disk."""
    n, h, w = stack.shape
    quantized = _quantize_int16(stack, nodata_value)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=n,
                        dtype="int16", crs=crs, transform=transform,
                        nodata=nodata_value, compress="deflate") as dst:
        for i in range(n):
            dst.write(quantized[i], i + 1)
            dst.set_band_description(i + 1, dates[i].isoformat())


def run_tsi_chunked(tss_path, out_path, date_range, doy_range=(1, 365), int_day=5,
                     rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                     above_noise=3.0, below_noise=1.0, chunk_size=None,
                     blas_threads=1, verbose_chunks=False, despike=True):
    """Same computation as force_tsi.run_tsi(), but bounds memory and
    avoids BLAS thread oversubscription.

    chunk_size=None processes the whole tile at once, identical to
    run_tsi(). An integer pixel size (e.g. 512) instead reads, despikes,
    and interpolates one chunk_size x chunk_size spatial block at a
    time, writing each block straight to the output file as soon as
    it's done -- the full (n_dates, height, width) output is never
    resident in memory, only one block's worth at a time.

    :param blas_threads: threads allowed per BLAS call (rbf_interpolate()'s
        np.tensordot) -- see _blas_limiter()/module docstring point 4.
        Default 1 avoids thread oversubscription; None leaves BLAS's
        default (usually "use every core") in place.
    :param verbose_chunks: if True and chunk_size is set, print a line
        per chunk as it completes (count, elapsed, ETA) -- useful for
        confirming a long-running job is actually progressing, since
        otherwise nothing prints until the whole file is done. Off by
        default since it's one line per chunk, per (tile, VI) job.
    :param despike: if True (the default), run force_tsi.despike() on
        each chunk before interpolating, exactly as force_tsi.run_tsi()
        does. despike()'s own above_noise/below_noise already support
        skipping it (above_noise<=0), but that's the same as leaving
        despike=True and passing above_noise=0 -- this flag skips the
        despike_fn() call entirely instead, for when you want to rule
        despiking in or out of a slow run without touching those
        thresholds, or avoid its extra per-chunk compute/memory outright.
    :return: the output_dates that were written (one band each). Written
        as int16 (nodata=DEFAULT_NODATA_INT16), not force_tsi.run_tsi()'s
        float32 -- see _write_tsi_int16()/module docstring point 7.
    """
    output_dates = output_dates_from_range(date_range, int_day, doy_range)

    if chunk_size is None:
        with _blas_limiter(blas_threads):
            dates, cube, transform, crs = load_tss(tss_path)
            if despike:
                cleaned, _removed = despike_fn(dates, cube, above_noise, below_noise)
            else:
                cleaned = cube
            precomputed = _precompute_rbf_weights(dates, output_dates, rbf_sigma, rbf_cutoff)
            stack = _apply_rbf_weights(cleaned, precomputed)
        _write_tsi_int16(out_path, stack, transform, crs, output_dates)
        return output_dates

    width, height, transform, crs = tss_grid_meta(tss_path)
    dates = _dates_from_tss(tss_path)
    precomputed = _precompute_rbf_weights(dates, output_dates, rbf_sigma, rbf_cutoff)

    n = len(output_dates)
    windows = list(_iter_chunk_windows(width, height, chunk_size))
    total_chunks = len(windows)
    label = os.path.basename(tss_path)
    start = time.monotonic()

    with rasterio.open(out_path, "w", driver="GTiff", height=height, width=width, count=n,
                        dtype="int16", crs=crs, transform=transform, nodata=DEFAULT_NODATA_INT16,
                        compress="deflate") as dst:
        for i, d in enumerate(output_dates):
            dst.set_band_description(i + 1, d.isoformat())

        for chunk_i, window in enumerate(windows, start=1):
            chunk_dates, cube = load_tss_window(tss_path, window)
            with _blas_limiter(blas_threads):
                if despike:
                    cleaned, _removed = despike_fn(chunk_dates, cube, above_noise, below_noise)
                else:
                    cleaned = cube
                sub_stack = _apply_rbf_weights(cleaned, precomputed)
            quantized_stack = _quantize_int16(sub_stack)
            for i in range(n):
                dst.write(quantized_stack[i], i + 1, window=window)

            if verbose_chunks:
                elapsed = time.monotonic() - start
                eta = (f", ~{_format_duration(elapsed / chunk_i * (total_chunks - chunk_i))} remaining"
                       if chunk_i < total_chunks else "")
                print(f"    [{label}] chunk {chunk_i}/{total_chunks}, {_format_duration(elapsed)} elapsed{eta}", flush=True)

    return output_dates


# ---------------------------------------------------------------------
# Shared job execution (sequential or parallel), one file at a time
# ---------------------------------------------------------------------

def _run_one(tss_path, out_path, date_range, doy_range, int_day,
             rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size, skip_existing,
             blas_threads, verbose_chunks, despike=True):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor.

    Catches its own exceptions -- a bad/missing/corrupt file becomes a
    "failed: ..." status rather than aborting the whole batch, whether
    running sequentially or in parallel.
    """
    if skip_existing and os.path.exists(out_path):
        return tss_path, out_path, "skipped"
    try:
        run_tsi_chunked(tss_path, out_path, date_range, doy_range, int_day,
                         rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                         blas_threads, verbose_chunks, despike)
    except Exception as e:
        return tss_path, out_path, f"failed: {type(e).__name__}: {e}"
    return tss_path, out_path, "written"


def _execute_jobs(job_specs, date_range, doy_range, int_day, rbf_sigma, rbf_cutoff,
                   above_noise, below_noise, chunk_size, skip_existing, max_workers, label_fn,
                   blas_threads=1, verbose_chunks=False, despike=True):
    """Sequential-or-parallel execution core shared by run_tsi_batch() and
    run_tsi_tiles(). job_specs is a list of (tss_path, out_path, key) --
    `key` is whatever the caller wants back to identify the job (a path
    for run_tsi_batch(), a (tile_id, canonical_vi) pair for
    run_tsi_tiles()). label_fn(key) formats a key for progress printing.

    Prints a running "[done/total, elapsed, ~remaining]" prefix on every
    line, in both sequential and parallel modes (in parallel, "done"
    counts completions, which may arrive out of job_specs order).

    :return: {key: (tss_path, out_path, status)}
    """
    results = {}
    total = len(job_specs)
    if total == 0:
        print("Nothing to do (0 jobs).")
        return results

    start = time.monotonic()

    def progress(done):
        elapsed = time.monotonic() - start
        eta = f", ~{_format_duration(elapsed / done * (total - done))} remaining" if 0 < done < total else ""
        return f"[{done}/{total}, {_format_duration(elapsed)} elapsed{eta}]"

    if max_workers <= 1:
        for i, (tss_path, out_path, key) in enumerate(job_specs, start=1):
            result = _run_one(tss_path, out_path, date_range, doy_range, int_day,
                               rbf_sigma, rbf_cutoff, above_noise, below_noise,
                               chunk_size, skip_existing, blas_threads, verbose_chunks, despike)
            print(f"  {progress(i)} {label_fn(key)}: {result[2]}")
            results[key] = result
        return results

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, tss_path, out_path, date_range, doy_range, int_day,
                             rbf_sigma, rbf_cutoff, above_noise, below_noise,
                             chunk_size, skip_existing, blas_threads, verbose_chunks, despike): (tss_path, out_path, key)
            for tss_path, out_path, key in job_specs
        }
        for fut in as_completed(futures):
            tss_path, out_path, key = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = (tss_path, out_path, f"failed: {type(e).__name__}: {e}")
            done += 1
            print(f"  {progress(done)} {label_fn(key)}: {result[2]}")
            results[key] = result

    return results


def _format_duration(seconds):
    """Human-readable H:MM:SS (or M:SS under an hour), for progress()."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _print_summary(statuses, elapsed):
    """One-line rollup of a batch of "written"/"skipped"/"failed: ..."
    statuses, plus total wall time -- printed at the end of run_tsi_batch()
    and run_tsi_tiles().

    :return: {"written": int, "skipped": int, "failed": int, "total": int,
        "elapsed_seconds": float} -- for callers that also want to record
        this (see _write_run_params()); existing callers that only
        wanted the print can ignore the return value.
    """
    written = sum(1 for s in statuses if s == "written")
    skipped = sum(1 for s in statuses if s == "skipped")
    failed = sum(1 for s in statuses if s.startswith("failed"))
    print(f"Done: {written} written, {skipped} skipped, {failed} failed "
          f"(of {len(statuses)}) in {_format_duration(elapsed)}")
    return {"written": written, "skipped": skipped, "failed": failed,
            "total": len(statuses), "elapsed_seconds": elapsed}


def _write_run_params(out_dir, script, params, summary=None):
    """Writes a timestamped JSON record of one run_tsi_tiles()-style
    call's own parameters (and, if given, a brief result summary) into
    `out_dir` -- the directory that HOLDS the <tile_id> subfolders, not
    any one tile's own folder -- see module docstring point 8.

    Timestamped (not a fixed filename) so successive runs against the
    same out_dir -- e.g. a batch backfill followed later by several
    stepwise updates -- each leave their own record rather than
    overwriting the previous one.

    :param script: which function produced this run, e.g.
        "force_tsi_batch.run_tsi_tiles" -- also used to tag the filename.
    :param params: {name: value} of the call's own arguments (a plain
        dict(locals()) snapshot taken at the top of the caller, before
        any other local variables exist, is the intended way to build
        this -- see run_tsi_tiles()). Values that aren't natively JSON
        types (dates, Path-like objects, etc.) are stringified via
        `default=str` rather than requiring the caller to pre-serialize
        them.
    :return: the path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {"script": script, "timestamp": timestamp, "params": params}
    if summary is not None:
        record["summary"] = summary

    short_name = script.rsplit(".", 1)[-1]
    path = os.path.join(out_dir, f"{short_name}_params_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"Wrote run parameters to {path}")
    return path


# ---------------------------------------------------------------------
# Flat file-list processing (no tile-directory structure assumed)
# ---------------------------------------------------------------------

def find_tss_files(root_dir, pattern="*_TSS.tif"):
    """Glob every TSS tif under root_dir (recursively) matching pattern.
    A flat, structure-agnostic finder -- if your archive follows FORCE's
    usual one-directory-per-tile layout, prefer run_tsi_tiles(), which
    understands tile boundaries and can find each VI's file per tile
    rather than just pattern-matching filenames anywhere in the tree."""
    return sorted(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))


def run_tsi_batch(tss_paths, out_dir, date_range, doy_range=(1, 365), int_day=5,
                   rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                   above_noise=3.0, below_noise=1.0, chunk_size=None,
                   max_workers=1, skip_existing=True, suffix="_TSI",
                   blas_threads=1, verbose_chunks=False, despike=True):
    """Run run_tsi_chunked() over an arbitrary flat list of TSS tifs,
    writing each to out_dir with `suffix` appended before the extension.

    Files are independent of each other, so max_workers > 1 processes
    several concurrently (one process per file; chunk_size still bounds
    memory *within* each file, independent of max_workers).

    :param tss_paths: list of input TSS tif paths, e.g. from find_tss_files().
    :param skip_existing: if True, a file whose output already exists is
        left untouched instead of being recomputed -- makes re-running
        this over a growing archive cheap.
    :param blas_threads, verbose_chunks: see run_tsi_chunked() -- forwarded
        to every job. blas_threads=1 (default) avoids BLAS thread
        oversubscription across workers; verbose_chunks=True prints a
        line per spatial chunk (only meaningful if chunk_size is set).
    :param despike: see run_tsi_chunked() -- forwarded to every job.
        despike=False skips force_tsi.despike() entirely (no removal/
        restoration passes, no despike-related compute or memory) --
        useful for isolating whether despiking (rather than chunking/
        multiprocessing itself) is behind a slow run.
    :return: list of (tss_path, out_path, status) -- status is "written",
        "skipped", or "failed: <message>", in the same order as tss_paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    job_specs = []
    for tss_path in tss_paths:
        base, ext = os.path.splitext(os.path.basename(tss_path))
        out_path = os.path.join(out_dir, f"{base}{suffix}{ext}")
        job_specs.append((tss_path, out_path, tss_path))  # key = tss_path (unique)

    start = time.monotonic()
    results_by_key = _execute_jobs(
        job_specs, date_range, doy_range, int_day, rbf_sigma, rbf_cutoff,
        above_noise, below_noise, chunk_size, skip_existing, max_workers,
        label_fn=lambda key: key, blas_threads=blas_threads, verbose_chunks=verbose_chunks,
        despike=despike,
    )
    ordered = [results_by_key[tss_path] for tss_path in tss_paths]
    _print_summary([status for _, _, status in ordered], time.monotonic() - start)
    return ordered


# ---------------------------------------------------------------------
# Tile-directory discovery, matching FORCE's one-folder-per-tile layout
# ---------------------------------------------------------------------

DEFAULT_VI_PATTERN_TEMPLATE = "*_{vi}_TSS.tif"


def _read_tile_id_csv(path, column="Tile_ID"):
    """Read one column of tile IDs out of a CSV -- used as an exclude
    list by discover_tile_dirs(). Tile IDs are matched against tile
    *folder names* exactly (e.g. "X0068_Y0043").

    Opened with utf-8-sig rather than plain utf-8: CSVs saved from Excel
    on Windows commonly start with a UTF-8 BOM, which plain utf-8 leaves
    attached to the first header (so "Tile_ID" reads as "\ufeffTile_ID"
    and never matches `column`). utf-8-sig strips it if present and is a
    harmless no-op if it isn't.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Column {column!r} not found in {path} (columns: {reader.fieldnames})")
        return {row[column].strip() for row in reader if row.get(column, "").strip()}


def discover_tile_dirs(root_dir, tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include"):
    """List tile subdirectories directly under root_dir -- the usual
    FORCE tiled layout is one folder per tile (e.g. root_dir/X0068_Y0043/),
    each holding one TSS file per VI.

    :param tiles_csv: optional path to a CSV with a `tiles_column` column
        of tile IDs (folder names). Default mode ("include") treats this
        as an ALLOW list, matching the original codebase's grid_csv
        (e.g. CH_FORCE_Grids.csv): only tiles named in the CSV are
        processed, everything else under root_dir is ignored. Pass
        tiles_csv_mode="exclude" to invert that instead -- every tile is
        processed EXCEPT the ones named here.
    :param tiles_column: column name in tiles_csv holding tile IDs.
    :param tiles_csv_mode: "include" (default) or "exclude".
    :return: sorted list of tile directory paths.
    """
    tile_dirs = sorted(p for p in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(p))

    if tiles_csv:
        csv_ids = _read_tile_id_csv(tiles_csv, tiles_column)
        before = len(tile_dirs)
        if tiles_csv_mode == "include":
            tile_dirs = [p for p in tile_dirs if os.path.basename(p) in csv_ids]
            print(f"  tiles_csv ({tiles_csv}, include): kept {len(tile_dirs)} of {before} tile(s)")
        elif tiles_csv_mode == "exclude":
            tile_dirs = [p for p in tile_dirs if os.path.basename(p) not in csv_ids]
            print(f"  tiles_csv ({tiles_csv}, exclude): excluded {before - len(tile_dirs)} of {before} tile(s)")
        else:
            raise ValueError(f"tiles_csv_mode must be 'include' or 'exclude', got {tiles_csv_mode!r}")

    return tile_dirs


def find_tile_vi_file(tile_dir, file_token, vi_pattern_template=DEFAULT_VI_PATTERN_TEMPLATE, year=None):
    """Find the one TSS file for a single VI within a single tile
    directory. Raises rather than guessing if zero or multiple files
    match -- a tile should have exactly one TSS file per VI.

    :param file_token: the token this archive's filenames actually use
        for the VI (see find_tile_vi_file()'s callers / vi_keys in
        run_tsi_tiles() for the canonical-name -> file_token mapping).
    :param vi_pattern_template: glob pattern with a `{vi}` placeholder
        (and optionally a `{year}` placeholder -- pass `year=` to fill it
        in, for archives where a tile folder holds more than one year's
        files).
    """
    fmt_kwargs = {"vi": file_token}
    if year is not None:
        fmt_kwargs["year"] = year
    pattern = vi_pattern_template.format(**fmt_kwargs)

    matches = sorted(glob.glob(os.path.join(tile_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {tile_dir}")
    if len(matches) > 1:
        raise ValueError(f"multiple files matching {pattern!r} in {tile_dir}: {matches}")
    return matches[0]


def run_tsi_tiles(root_dir, vi_keys, out_dir, date_range, doy_range=(1, 365), int_day=5,
                   rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                   above_noise=3.0, below_noise=1.0, chunk_size=None,
                   max_workers=1, skip_existing=True, suffix="_TSI",
                   vi_pattern_template=DEFAULT_VI_PATTERN_TEMPLATE, year=None,
                   tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include",
                   blas_threads=1, verbose_chunks=False, despike=True):
    """Run run_tsi_chunked() over every (tile, VI) pair in a FORCE-tiled
    archive: discover_tile_dirs() finds the tile folders, find_tile_vi_file()
    locates each VI's file within each one, and every job runs through
    the same sequential-or-parallel path as run_tsi_batch(). Output
    mirrors the input layout: out_dir/<tile_id>/<basename><suffix>.tif.

    A tile missing one of the requested VIs is skipped for that VI only
    (printed as "not found", not treated as a fatal error) -- the rest
    of that tile's VIs, and every other tile, still run.

    :param vi_keys: {canonical_name: file_token}, e.g.
        {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}   # 2026 naming
        {"NDVI": "NDV",  "EVI": "EVI", "NDMI": "NDM",  "CIRE": "CRE",  "CCI": "CCI"}   # 2018 naming
    :param tiles_csv, tiles_column, tiles_csv_mode: optional CSV
        restricting which tiles get processed -- see discover_tile_dirs().
        Default mode "include" is an allow list (matching the original
        codebase's grid_csv, e.g. CH_FORCE_Grids.csv); pass
        tiles_csv_mode="exclude" for a skip list instead.
    :param blas_threads, verbose_chunks: see run_tsi_chunked() -- forwarded
        to every job. blas_threads=1 (default) avoids BLAS thread
        oversubscription across workers -- important here, since each
        (tile, VI) job's per-chunk, per-output-date tensordot calls can
        number in the thousands over a whole archive. verbose_chunks=True
        prints a line per spatial chunk (only meaningful if chunk_size
        is set) -- useful for confirming a slow-looking run is actually
        progressing.
    :param despike: see run_tsi_chunked() -- forwarded to every job.
        despike=False skips force_tsi.despike() entirely (no removal/
        restoration passes, no despike-related compute or memory) --
        useful for isolating whether despiking (rather than chunking/
        multiprocessing itself) is behind a slow run.
    :return: {tile_id: {canonical_vi: (tss_path, out_path, status)}}
    """
    call_params = dict(locals())  # snapshot of every argument -- see _write_run_params()

    tile_dirs = discover_tile_dirs(root_dir, tiles_csv, tiles_column, tiles_csv_mode)
    print(f"Found {len(tile_dirs)} tile(s) under {root_dir}")

    job_specs = []
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        tile_out_dir = os.path.join(out_dir, tile_id)
        os.makedirs(tile_out_dir, exist_ok=True)

        for canonical, file_token in vi_keys.items():
            try:
                tss_path = find_tile_vi_file(tile_dir, file_token, vi_pattern_template, year)
            except FileNotFoundError as e:
                print(f"  {tile_id}/{canonical}: not found -- {e}")
                continue
            base, ext = os.path.splitext(os.path.basename(tss_path))
            out_path = os.path.join(tile_out_dir, f"{base}{suffix}{ext}")
            job_specs.append((tss_path, out_path, (tile_id, canonical)))

    start = time.monotonic()
    results_by_key = _execute_jobs(
        job_specs, date_range, doy_range, int_day, rbf_sigma, rbf_cutoff,
        above_noise, below_noise, chunk_size, skip_existing, max_workers,
        label_fn=lambda key: f"{key[0]}/{key[1]}",
        blas_threads=blas_threads, verbose_chunks=verbose_chunks, despike=despike,
    )
    summary = _print_summary([status for _, _, status in results_by_key.values()], time.monotonic() - start)
    _write_run_params(out_dir, "force_tsi_batch.run_tsi_tiles", call_params, summary)

    results = {}
    for (tile_id, canonical), result in results_by_key.items():
        results.setdefault(tile_id, {})[canonical] = result
    return results


# Example (from a notebook):
#
#   from datetime import date
#   from force_tsi_batch import find_tss_files, run_tsi_batch
#
#   tss_paths = find_tss_files(r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss", "*_NDVI_TSS.tif")
#   print(f"Found {len(tss_paths)} NDVI TSS files")
#
#   results = run_tsi_batch(
#       tss_paths,
#       out_dir=r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level4_rbf\2026",
#       date_range=(date(2026, 3, 1), date(2026, 8, 16)),
#       doy_range=(1, 365), int_day=5,
#       rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
#       above_noise=3.0, below_noise=1.0,
#       chunk_size=512,      # bounds memory per file; None = whole tile at once
#       max_workers=4,       # process this many files concurrently
#       skip_existing=True,  # rerun-safe: only computes files without output yet
#   )
#
#   written = sum(1 for _, _, status in results if status == "written")
#   print(f"Wrote {written} of {len(results)} file(s)")
#
# Whole tiled archive, every VI, restricted to a tile allow-list (the
# common case -- e.g. CH_FORCE_Grids.csv, a Tile_ID column of tiles to
# actually process):
#
#   from force_tsi_batch import run_tsi_tiles
#
#   vi_keys_2026 = {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}
#   vi_keys_2018 = {"NDVI": "NDV",  "EVI": "EVI", "NDMI": "NDM",  "CIRE": "CRE",  "CCI": "CCI"}
#
#   results = run_tsi_tiles(
#       r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss",
#       vi_keys_2026,
#       out_dir=r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level4_rbf\2026",
#       date_range=(date(2026, 3, 1), date(2026, 8, 16)),
#       doy_range=(1, 365), int_day=5,
#       rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
#       above_noise=3.0, below_noise=1.0,
#       chunk_size=512, max_workers=4, skip_existing=True,
#       tiles_csv=r"B:\bloomc\DiscoCH_2026_08_03\FORCE\data\CH_FORCE_Grids.csv",  # column "Tile_ID"; omit/None to process every tile
#       # tiles_csv_mode="exclude",  # uncomment to treat tiles_csv as a skip list instead
#   )
#
#   for tile_id, by_vi in results.items():
#       for vi, (tss_path, out_path, status) in by_vi.items():
#           if status != "written":
#               print(f"{tile_id}/{vi}: {status}")
"""
force_tsi_stepwise.py

Incremental ("stepwise") counterpart to force_tsi.py/force_tsi_batch.py:
run it over and over as a tile's raw TSS archive grows (e.g. on a cron
schedule), and each call only computes whatever output dates have newly
become eligible since the last call, merging them into the existing
output file instead of recomputing the whole season from scratch every
time. This is the same incremental design rbf_interp.py already had
(_last_raw_band_date() / eligible_output_dates() / _process_one_tile()'s
skip-what's-already-written logic) -- ported onto force_tsi.py's minimal,
param-file-matching foundation and force_tsi_batch.py's int16 output
(see force_tsi_batch.py's module docstring point 7) instead of
rbf_interp.py's multi-width ensemble/isolated-point-fallback machinery.

NO REFORMATTING OF force_tsi.py / force_tsi_batch.py WAS NEEDED. Every
low-level piece this needs (load_tss/load_tss_window, despike(),
_precompute_rbf_weights()/_apply_rbf_weights(), _quantize_int16(),
_iter_chunk_windows(), _blas_limiter(), tile discovery) already existed
there and is reused as-is (imported, not copied) -- this file only adds
the NEW orchestration layer: figuring out which output dates are
actually new, and merging them into an existing file rather than
overwriting it. force_tsi.py and force_tsi_batch.py are untouched, so
their current one-shot/full-season behavior remains available for
reference exactly as it was.

WHAT MAKES A DATE "ELIGIBLE"
------------------------------
An output date is eligible once the source TSS file has real data at
least `wait_days` past it -- giving the ~5-day satellite revisit a
chance to deliver at least one forward observation before a date is
finalized (see eligible_output_dates()). Unlike force_tsi.rbf_interpolate()
itself (which has no such concept -- it just looks as far as each sigma's
own cutoff radius reaches, in both directions, whenever it's called),
this is purely a SCHEDULING gate layered on top: which dates get PASSED
to rbf_interpolate() at all, not a change to what rbf_interpolate()
computes for a date once it's asked to. The gate uses the tile's OWN
most recent real acquisition date (read from band descriptions only, no
pixel data -- see _last_raw_date()) as the "as of" point, not wall-clock
time, so re-running this against a not-yet-updated archive is always a
safe, cheap no-op ("up to date") rather than depending on when you
happen to run it.

Once a date has been written, it is NEVER recomputed on a later call
(mirroring rbf_interp.py's own behavior) -- eligibility only depends on
the source file's own dates and output_dates_from_range(), never on
what's already been written, so re-running never redoes existing work.

IMPORTANT CAVEAT -- wait_days vs. each sigma's own cutoff radius: unlike
rbf_interp.py's delayed_smooth_one_date() (which bounds its OWN forward
compute window to wait_days, so "wait_days elapsed" and "the whole
symmetric window is available" are the same thing), force_tsi.rbf_interpolate()
is left completely unchanged here -- it looks as far as each configured
sigma's own cutoff radius reaches, in BOTH directions, with no wait_days
awareness at all. That means a date becoming "eligible" (wait_days past
it) does NOT guarantee every configured sigma's full window is already
available -- rbf_sigma=(10, 20, 30, 50) at rbf_cutoff=0.95 has cutoff
radii of ~20/39/59/98 days, all bigger than the default wait_days=10. A
date computed the moment it's merely "eligible" is therefore computed
with whatever forward context exists AT THAT TIME, which can genuinely
be less than a full archival run (done much later, once far more future
data exists) would have used for the wide sigmas -- and since a written
date is never recomputed, that difference is permanent for that date.
If you want stepwise output to always match what an eventual full-archive
run would produce for the same date, set wait_days to at least the
widest configured sigma's own cutoff radius -- see
recommended_wait_days(). The smaller default trades a bit of that
convergence guarantee for lower latency (dates become final sooner);
raise wait_days if exact convergence matters more than latency for your
use case.

MERGING INTO AN EXISTING FILE
-------------------------------
GeoTIFF has no true incremental band-append, so a merge always rewrites
the WHOLE output file: existing bands (already int16) are read back
unchanged, the newly computed bands are quantized to int16
(_quantize_int16()) and combined with them, and the union is written out
sorted by date -- but only the NEW dates were ever actually computed.
Unlike rbf_interp.py's _merge_and_write_rbf_output(), the rewrite goes to
a temporary path first and is only swapped into place (os.replace(), atomic
on the same filesystem) once it succeeds -- since this is meant to run
unattended and repeatedly, a crash mid-rewrite should never corrupt or
truncate the previous, already-good output.

chunk_size (optional, same meaning as force_tsi_batch.run_tsi_chunked())
only bounds the READ+COMPUTE side -- it does not change how the merge
happens. In steady-state operation only a handful of dates become newly
eligible per call, so this is rarely needed at all; it exists mainly for
a large first backfill run against an archive that already has a full
season of data.

run_tsi_tiles_stepwise() now defaults to the SAME `suffix` ("_TSI") as
force_tsi_batch.run_tsi_tiles() -- it used to default to "_TSI_stepwise",
which meant continuing an existing batch-produced archive with stepwise
required explicitly overriding suffix to line the two up. Now the common
case (batch a solid historical backfill, then hand the same out_dir off
to run_tsi_tiles_stepwise() for the still-arriving tail of the season)
just works without having to pass a matching suffix by hand -- the
default naming already targets the same files. Also see
force_tsi_batch.py's module docstring point 8 (_write_run_params()):
both run_tsi_tiles() and run_tsi_tiles_stepwise() write a timestamped
JSON record of that call's own parameters into out_dir, so if you DO mix
batch and stepwise runs against the same archive over time, it's always
possible to tell what parameters produced which dates later.

Usage (from a notebook, e.g. on a schedule):

    from datetime import date
    from src.disco_ch.force_tsi_stepwise import run_tsi_tiles_stepwise

    vi_keys = {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}

    rbf_sigma = (10, 20, 30, 50)
    rbf_cutoff = 0.95
    # wait_days=10 finalizes dates fast but can "lock in" the widest
    # sigma(s) with less forward context than a later full-archive run
    # would use for the same date (see module docstring's IMPORTANT
    # CAVEAT) -- pass recommended_wait_days(rbf_sigma, rbf_cutoff) instead
    # if you want stepwise output to always match an eventual full run.
    results = run_tsi_tiles_stepwise(
        r"B:\\bloomc\\DiscoCH_2026_08_03\\FORCE\\level3_tss",
        vi_keys,
        # Same out_dir a run_tsi_tiles() batch backfill already wrote to --
        # suffix defaults to "_TSI" for both, so this continues those files.
        out_dir=r"B:\\bloomc\\DiscoCH_2026_08_03\\FORCE\\level4_rbf\\2026",
        date_range=(date(2026, 3, 1), date(2026, 10, 31)),
        doy_range=(1, 365), int_day=5, wait_days=10,
        rbf_sigma=rbf_sigma, rbf_cutoff=rbf_cutoff,
        above_noise=3.0, below_noise=1.0, despike=True,
        chunk_size=None, max_workers=4,
    )
    # Run again later (e.g. tomorrow, once more raw scenes have arrived):
    # only the newly-eligible dates get computed and merged in.
"""

import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio

from src.disco_ch.force_tsi import (
    load_tss,
    load_tss_window,
    tss_grid_meta,
    despike as despike_fn,
    output_dates_from_range,
    rbf_cutoff_radius,
)
from src.disco_ch.force_tsi_batch import (
    _iter_chunk_windows,
    _blas_limiter,
    _dates_from_tss,
    _precompute_rbf_weights,
    _apply_rbf_weights,
    _quantize_int16,
    _format_duration,
    _write_run_params,
    _year_from_date_range,
    DEFAULT_NODATA_INT16,
    discover_tile_dirs,
    find_tile_vi_file,
    DEFAULT_VI_PATTERN_TEMPLATE,
)

# How long to wait, past an output date, before treating it as eligible
# to compute at all -- see module docstring. 0 disables the gate (every
# date within the season is eligible the moment the raw archive has ANY
# data at or past it).
DEFAULT_WAIT_DAYS = 10


# ---------------------------------------------------------------------
# Eligibility -- which output dates are new since the last run
# ---------------------------------------------------------------------

def eligible_output_dates(output_dates, as_of_date, wait_days=DEFAULT_WAIT_DAYS):
    """Of a full season's output dates, returns those eligible as of
    `as_of_date` (target_date + wait_days <= as_of_date) -- same
    semantics as rbf_interp.eligible_output_dates(). `as_of_date` is
    normally the source file's own last real acquisition date (see
    _last_raw_date()), not wall-clock time -- see module docstring."""
    return [d for d in output_dates if d + timedelta(days=wait_days) <= as_of_date]


def recommended_wait_days(rbf_sigma, rbf_cutoff=0.95):
    """The smallest wait_days that guarantees every configured sigma's
    own cutoff radius is already fully available (in both directions)
    by the time a date is treated as eligible -- see module docstring's
    "IMPORTANT CAVEAT" section. Below this, stepwise output for the
    widest sigma(s) can be "locked in" using less forward context than a
    later full-archive run would have used for the same date (and, since
    a written date is never recomputed, that difference is permanent).
    Pass the result as `wait_days` if convergence to an eventual
    full-archive result matters more than getting dates finalized
    quickly.
    """
    return max(rbf_cutoff_radius(s, rbf_cutoff) for s in rbf_sigma)


def _last_raw_date(tss_path):
    """Most recent acquisition date actually present in a TSS file,
    read from band descriptions only (see force_tsi_batch._dates_from_tss(),
    reused directly) -- the ceiling on which output dates can be computed
    at all right now. None if the file has no dated bands."""
    dates = _dates_from_tss(tss_path)
    return max(dates) if dates else None


# ---------------------------------------------------------------------
# Existing output introspection / merge-write
# ---------------------------------------------------------------------

def _existing_output_dates(out_path):
    """Band dates already present in an existing stepwise output file,
    or None if out_path doesn't exist or can't be read as one (missing,
    corrupt, or not a dated multiband raster this module wrote) -- in
    which case every eligible date must be (re)computed rather than
    skipped. Descriptions-only read, no pixel data touched."""
    if not os.path.exists(out_path):
        return None
    try:
        with rasterio.open(out_path) as src:
            descriptions = src.descriptions
    except rasterio.errors.RasterioIOError:
        return None

    dates = set()
    for desc in descriptions:
        if not desc:
            continue
        try:
            dates.add(datetime.strptime(desc, "%Y-%m-%d").date())
        except ValueError:
            continue
    return dates


def _read_existing_output(out_path):
    """Full existing stepwise output -- (dates, int16 array, transform,
    crs, nodata) -- for merging newly computed bands into it. The array
    is read as-is (already int16 -- no requantization needed for bands
    that aren't changing)."""
    with rasterio.open(out_path) as src:
        arr = src.read()
        transform = src.transform
        crs = src.crs
        nodata_value = src.nodata
        descriptions = src.descriptions
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in descriptions]
    return dates, arr, transform, crs, nodata_value


def _write_int16_stack(path, stack_int16, transform, crs, dates, nodata_value):
    """Writes an ALREADY-int16 (n, h, w) stack (see
    force_tsi_batch._quantize_int16() for the float -> int16 step) to a
    temporary path, then atomically swaps it into place (os.replace()) --
    a crash mid-write leaves the previous, already-good `path` untouched
    instead of a truncated/corrupt file (see module docstring)."""
    n, h, w = stack_int16.shape
    tmp_path = path + ".tmp"
    with rasterio.open(tmp_path, "w", driver="GTiff", height=h, width=w, count=n,
                        dtype="int16", crs=crs, transform=transform,
                        nodata=nodata_value, compress="deflate") as dst:
        for i in range(n):
            dst.write(stack_int16[i].astype("int16"), i + 1)
            dst.set_band_description(i + 1, dates[i].isoformat())
    os.replace(tmp_path, path)


def _merge_and_write_output(out_path, new_dates, new_stack_float, transform, crs,
                             nodata_value=DEFAULT_NODATA_INT16):
    """Combines whatever's already in out_path (read back unchanged) with
    newly computed bands (quantized to int16 here -- see
    force_tsi_batch._quantize_int16()), and (re)writes the union sorted
    by date -- see module docstring's MERGING section for why this has
    to rewrite the whole file, and _write_int16_stack() for the
    crash-safety of that rewrite."""
    new_quantized = _quantize_int16(new_stack_float, nodata_value)

    combined = {}
    if os.path.exists(out_path):
        existing_dates, existing_arr, transform, crs, existing_nodata = _read_existing_output(out_path)
        if existing_nodata is not None:
            nodata_value = existing_nodata
        combined.update(zip(existing_dates, existing_arr))
    combined.update(zip(new_dates, new_quantized))

    all_dates = sorted(combined)
    stack = np.stack([combined[d] for d in all_dates])
    _write_int16_stack(out_path, stack, transform, crs, all_dates, nodata_value)


# ---------------------------------------------------------------------
# Single-file incremental update
# ---------------------------------------------------------------------

def run_tsi_stepwise(tss_path, out_path, date_range, doy_range=(1, 365), int_day=5,
                      wait_days=DEFAULT_WAIT_DAYS,
                      rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                      above_noise=3.0, below_noise=1.0, chunk_size=None,
                      blas_threads=1, despike=True, verbose_chunks=False,
                      nodata_value=DEFAULT_NODATA_INT16):
    """Incrementally updates one VI's stepwise output at out_path: of
    output_dates_from_range()'s full theoretical season calendar, only
    the dates that are BOTH eligible (see eligible_output_dates(), using
    tss_path's own most recent real date as the "as of" point) AND not
    already present in out_path are computed and merged in -- everything
    already written is left untouched.

    Safe to call repeatedly (e.g. on a schedule) as tss_path grows: each
    call only pays for whatever's newly eligible since the last one, and
    a full from-scratch run over an already-complete archive produces
    the same final file a long series of incremental calls would have.

    :param wait_days: an output date isn't eligible until tss_path has
        real data at least this many days past it -- see module
        docstring. 0 disables the gate entirely.
    :param chunk_size: see force_tsi_batch.run_tsi_chunked() -- bounds
        the READ+COMPUTE memory for the dates being computed THIS call;
        rarely needed in steady state (see module docstring), useful for
        a large first backfill.
    :return: (new_dates, status) -- new_dates is the list of dates
        actually computed and merged in this call (empty unless status
        == "written"). status is one of:
          "no output dates in range" -- date_range/doy_range/int_day
              produce an empty season calendar.
          "no data" -- tss_path has no dated bands at all.
          "not eligible yet" -- tss_path's own raw archive doesn't reach
              wait_days past any not-yet-written output date yet.
          "up to date" -- every currently-eligible date is already
              written.
          "written" -- new_dates were computed and merged in.
    """
    output_dates = output_dates_from_range(date_range, int_day, doy_range)
    if not output_dates:
        return [], "no output dates in range"

    all_dates = _dates_from_tss(tss_path)
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

    precomputed = _precompute_rbf_weights(all_dates, new_dates, rbf_sigma, rbf_cutoff)

    if chunk_size is None:
        with _blas_limiter(blas_threads):
            dates, cube, transform, crs = load_tss(tss_path)
            if despike:
                cleaned, _removed = despike_fn(dates, cube, above_noise, below_noise)
            else:
                cleaned = cube
            new_stack = _apply_rbf_weights(cleaned, precomputed)
    else:
        width, height, transform, crs = tss_grid_meta(tss_path)
        windows = list(_iter_chunk_windows(width, height, chunk_size))
        total_chunks = len(windows)
        label = os.path.basename(tss_path)
        chunk_start = time.monotonic()

        new_stack = np.full((len(new_dates), height, width), np.nan, dtype="float32")
        for chunk_i, window in enumerate(windows, start=1):
            chunk_dates, cube = load_tss_window(tss_path, window)
            with _blas_limiter(blas_threads):
                if despike:
                    cleaned, _removed = despike_fn(chunk_dates, cube, above_noise, below_noise)
                else:
                    cleaned = cube
                sub_stack = _apply_rbf_weights(cleaned, precomputed)
            row0, col0 = window.row_off, window.col_off
            new_stack[:, row0:row0 + window.height, col0:col0 + window.width] = sub_stack

            if verbose_chunks:
                elapsed = time.monotonic() - chunk_start
                eta = (f", ~{_format_duration(elapsed / chunk_i * (total_chunks - chunk_i))} remaining"
                       if chunk_i < total_chunks else "")
                print(f"    [{label}] chunk {chunk_i}/{total_chunks}, {_format_duration(elapsed)} elapsed{eta}", flush=True)

    _merge_and_write_output(out_path, new_dates, new_stack, transform, crs, nodata_value)
    return new_dates, "written"


# ---------------------------------------------------------------------
# Tile-directory batch: every (tile, VI) pair, incrementally
# ---------------------------------------------------------------------

def _run_one_stepwise(tss_path, out_path, date_range, doy_range, int_day, wait_days,
                       rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                       blas_threads, despike, verbose_chunks, nodata_value):
    """Module-level (not nested) so it's picklable for ProcessPoolExecutor.
    Catches its own exceptions -- a bad/missing/corrupt file becomes a
    "failed: ..." status rather than aborting the whole batch."""
    try:
        new_dates, status = run_tsi_stepwise(
            tss_path, out_path, date_range, doy_range, int_day, wait_days,
            rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
            blas_threads, despike, verbose_chunks, nodata_value,
        )
    except Exception as e:
        return tss_path, out_path, f"failed: {type(e).__name__}: {e}"
    if status == "written":
        return tss_path, out_path, f"written ({len(new_dates)} date(s))"
    return tss_path, out_path, status


def run_tsi_tiles_stepwise(root_dir, vi_keys, out_dir, date_range, doy_range=(1, 365), int_day=5,
                            wait_days=DEFAULT_WAIT_DAYS,
                            rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                            above_noise=3.0, below_noise=1.0, chunk_size=None,
                            max_workers=1, suffix="_TSI",
                            vi_pattern_template=DEFAULT_VI_PATTERN_TEMPLATE, year=None,
                            tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include",
                            blas_threads=1, despike=True, verbose_chunks=False,
                            nodata_value=DEFAULT_NODATA_INT16):
    """Runs run_tsi_stepwise() over every (tile, VI) pair in a FORCE-tiled
    archive -- same discovery (discover_tile_dirs()/find_tile_vi_file(),
    reused from force_tsi_batch.py) and SAME default output layout/naming
    (out_dir/<tile_id>/<basename><suffix>.tif, suffix="_TSI") as
    force_tsi_batch.run_tsi_tiles(), so pointing this at a batch run's own
    out_dir continues those same files by default -- but incremental:
    call this again later (e.g. on a schedule) and only newly-eligible
    dates get computed and merged into each existing output file.

    :param vi_keys: {canonical_name: file_token}, e.g.
        {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}
    :param wait_days, chunk_size, despike: see run_tsi_stepwise().
    :param year: see force_tsi_batch.run_tsi_tiles() -- fills
        DEFAULT_VI_PATTERN_TEMPLATE's {year} placeholder; None (the
        default) derives it from date_range[0].year.
    :param tiles_csv, tiles_column, tiles_csv_mode: see
        force_tsi_batch.discover_tile_dirs().
    :return: {tile_id: {canonical_vi: (tss_path, out_path, status)}} --
        status is "up to date", "not eligible yet", "no data",
        "written (N date(s))", or "failed: ...".
    """
    if year is None:
        year = _year_from_date_range(date_range)
    call_params = dict(locals())  # snapshot of every argument (year already resolved) -- see force_tsi_batch._write_run_params()

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

    total = len(job_specs)
    if total == 0:
        print("Nothing to do (0 jobs).")
        return {}

    start = time.monotonic()

    def progress(done):
        elapsed = time.monotonic() - start
        eta = f", ~{_format_duration(elapsed / done * (total - done))} remaining" if 0 < done < total else ""
        return f"[{done}/{total}, {_format_duration(elapsed)} elapsed{eta}]"

    results_by_key = {}
    if max_workers <= 1:
        for i, (tss_path, out_path, key) in enumerate(job_specs, start=1):
            result = _run_one_stepwise(
                tss_path, out_path, date_range, doy_range, int_day, wait_days,
                rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                blas_threads, despike, verbose_chunks, nodata_value,
            )
            print(f"  {progress(i)} {key[0]}/{key[1]}: {result[2]}")
            results_by_key[key] = result
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_stepwise, tss_path, out_path, date_range, doy_range, int_day,
                    wait_days, rbf_sigma, rbf_cutoff, above_noise, below_noise, chunk_size,
                    blas_threads, despike, verbose_chunks, nodata_value,
                ): (tss_path, out_path, key)
                for tss_path, out_path, key in job_specs
            }
            for fut in as_completed(futures):
                tss_path, out_path, key = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = (tss_path, out_path, f"failed: {type(e).__name__}: {e}")
                done += 1
                print(f"  {progress(done)} {key[0]}/{key[1]}: {result[2]}")
                results_by_key[key] = result

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
    _write_run_params(out_dir, "force_tsi_stepwise.run_tsi_tiles_stepwise", call_params, summary)

    results = {}
    for (tile_id, canonical), result in results_by_key.items():
        results.setdefault(tile_id, {})[canonical] = result
    return results

"""
force_tsi.py

A minimal, direct match for what the FORCE TSA parameter file specifies
for outlier removal and interpolation -- nothing else.

Maps directly onto these param-file settings:

    ABOVE_NOISE / BELOW_NOISE   -> despike()
    RBF_SIGMA / RBF_CUTOFF      -> rbf_interpolate()
    INT_DAY / DATE_RANGE /
    DOY_RANGE                   -> output_dates_from_range()
    STANDARDIZE_TSI = NONE      -> not implemented (NONE means "do nothing")

Everything else in the param file -- masking (DIR_MASK), sensor merging
and spectral adjustment (SENSORS/SPECTRAL_ADJUST), the spectral index
itself (INDEX), STM/FBY/FBQ/POL/trend outputs, harmonic interpolation --
is computed by FORCE itself (force-higher-level) *before* this script
ever runs. This script only reproduces the interpolation FORCE applies
to an already-produced multiband TSS time series (OUTPUT_TSS = TRUE),
i.e. the OUTPUT_TSI step, for INTERPOLATE = RBF.

Deliberately NOT included, because the parameter file doesn't ask for
it: operational "wait_days" delay, isolated-point fallback, spatial
chunking, multiprocessing, or an alternate archive-directory input
format. If you need those for a production/recurring pipeline, add
them back on top of this -- but this file is the part that should
match the param file, on its own.

Meant to be imported and called directly (e.g. from a notebook) --
there's no command-line entry point here. For single-pixel testing
without running a whole tile, see force_tsi_plot.py.
"""

import re
import math
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.windows import Window

# A band description is expected to start with an 8-digit YYYYMMDD date
# (FORCE's usual naming for TSS bands), e.g. "20230405_NDVI".
DATE_RE = re.compile(r"(\d{8})")


# ---------------------------------------------------------------------
# Load a FORCE TSS multiband TIF (OUTPUT_TSS = TRUE)
# ---------------------------------------------------------------------

def _collapse_bands_to_daily(arr, descriptions):
    """Parse each band's date from its description; average same-day
    duplicates (e.g. two sensors acquiring the same date) into one layer
    per calendar date, ascending."""
    band_dates = []
    for desc in descriptions:
        m = DATE_RE.match(desc) if desc else None
        if not m:
            raise ValueError(f"Could not parse a YYYYMMDD date from band description: {desc!r}")
        band_dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())

    dates = sorted(set(band_dates))
    cube = np.full((len(dates), arr.shape[1], arr.shape[2]), np.nan, dtype="float32")
    for i, d in enumerate(dates):
        same_day = [j for j, dd in enumerate(band_dates) if dd == d]
        with np.errstate(invalid="ignore"):
            cube[i] = np.nanmean(arr[same_day], axis=0)

    return dates, cube


def load_tss(path):
    """Read an entire multiband TSS TIF into (dates, cube, transform, crs)."""
    with rasterio.open(path) as src:
        arr = src.read(masked=True).astype("float32").filled(np.nan)
        transform = src.transform
        crs = src.crs
        descriptions = src.descriptions

    dates, cube = _collapse_bands_to_daily(arr, descriptions)
    return dates, cube, transform, crs


def load_tss_window(path, window):
    """Like load_tss(), but only reads the given rasterio Window --
    cheap way to pull a small spatial subset (e.g. for testing on one
    pixel/patch) without reading the full tile. Returns (dates, cube)."""
    with rasterio.open(path) as src:
        arr = src.read(window=window, masked=True).astype("float32").filled(np.nan)
        descriptions = src.descriptions
    return _collapse_bands_to_daily(arr, descriptions)


def load_tss_pixel(path, row, col, buffer=0):
    """Like load_tss_window(), sized to a single pixel (or a small
    buffer x buffer neighborhood around it). Returns (dates, cube,
    local_row, local_col) -- local_row/local_col locate (row, col)
    within the returned (small) cube, since it's clipped to the raster
    edges when the buffer would otherwise run off the side."""
    with rasterio.open(path) as src:
        height, width = src.height, src.width
    row_off = max(row - buffer, 0)
    col_off = max(col - buffer, 0)
    row_end = min(row + buffer + 1, height)
    col_end = min(col + buffer + 1, width)
    window = Window(col_off=col_off, row_off=row_off,
                     width=col_end - col_off, height=row_end - row_off)
    dates, cube = load_tss_window(path, window)
    return dates, cube, row - row_off, col - col_off


def tss_grid_meta(path):
    """(width, height, transform, crs) for a TSS TIF, without reading any
    pixel data -- used to size chunked/windowed processing (see
    force_tsi_batch.py) without loading the whole tile first."""
    with rasterio.open(path) as src:
        return src.width, src.height, src.transform, src.crs


# ---------------------------------------------------------------------
# Outlier removal: ABOVE_NOISE / BELOW_NOISE
# ---------------------------------------------------------------------

def _triplet_interp(values, ordinals, idx_arr, n, h, w):
    """Linear interpolation, per pixel, between each position's nearest
    real neighbors on either side in time -- the "triplet" FORCE's
    ABOVE_NOISE/BELOW_NOISE description refers to."""
    active = ~np.isnan(values)

    idx_if_active = np.where(active, idx_arr, -1)
    last_idx = np.maximum.accumulate(idx_if_active, axis=0)
    last_idx = np.concatenate([np.full((1, h, w), -1, dtype=int), last_idx[:-1]], axis=0)

    idx_if_active_rev = np.where(active, idx_arr, n)
    next_idx = np.minimum.accumulate(idx_if_active_rev[::-1], axis=0)[::-1]
    next_idx = np.concatenate([next_idx[1:], np.full((1, h, w), n, dtype=int)], axis=0)

    has_prev, has_next = last_idx >= 0, next_idx < n
    safe_last, safe_next = np.clip(last_idx, 0, n - 1), np.clip(next_idx, 0, n - 1)

    prev_val = np.take_along_axis(values, safe_last, axis=0)
    next_val = np.take_along_axis(values, safe_next, axis=0)
    prev_ord, next_ord = ordinals[safe_last], ordinals[safe_next]
    cur_ord = np.broadcast_to(ordinals[:, None, None], (n, h, w))

    span = (next_ord - prev_ord).astype("float64")
    has_neighbors = has_prev & has_next & (span > 0)
    frac = (cur_ord - prev_ord) / np.where(span > 0, span, 1.0)

    with np.errstate(invalid="ignore"):
        interp = prev_val + (next_val - prev_val) * frac
    return interp, has_neighbors, active


def despike(dates, cube, above_noise=3.0, below_noise=1.0, max_iter=20):
    """FORCE's ABOVE_NOISE / BELOW_NOISE outlier filter.

    Per pixel, the RMSE of triplet residuals (real value vs. linear
    interpolation of its real neighbors) is the noise level. The single
    worst-residual observation exceeding `above_noise` times that noise
    is removed per pass, one at a time (removing it changes its
    neighbors' triplets and tightens the noise estimate), repeated up to
    `max_iter` passes. `above_noise = 0` disables filtering entirely.

    Once that converges, a removed point is restored if its ORIGINAL
    value now falls within `below_noise` times the noise estimated from
    its neighbors' current state -- again one best-fit point at a time,
    up to `max_iter` passes. `below_noise = 0` disables restoration.
    """
    if above_noise <= 0:
        return cube.copy(), np.zeros(cube.shape, dtype=bool)

    ordinals = np.array([d.toordinal() for d in dates], dtype="int64")
    n, h, w = cube.shape
    idx_arr = np.broadcast_to(np.arange(n)[:, None, None], (n, h, w))

    cleaned = cube.copy()
    removed = np.zeros((n, h, w), dtype=bool)

    for _ in range(max_iter):
        interp, has_neighbors, active = _triplet_interp(cleaned, ordinals, idx_arr, n, h, w)
        has_triplet = active & has_neighbors
        with np.errstate(invalid="ignore"):
            residual = np.where(has_triplet, cleaned - interp, np.nan)
            noise = np.sqrt(np.nanmean(residual ** 2, axis=0, keepdims=True))
        is_candidate = has_triplet & (np.abs(residual) > above_noise * noise)
        any_candidate = is_candidate.any(axis=0, keepdims=True)
        if not any_candidate.any():
            break
        worst = np.argmax(np.where(is_candidate, np.abs(residual), -np.inf), axis=0, keepdims=True)
        this_pass = (idx_arr == worst) & any_candidate
        cleaned = np.where(this_pass, np.nan, cleaned).astype("float32")
        removed |= this_pass

    if below_noise > 0:
        for _ in range(max_iter):
            if not removed.any():
                break
            interp, has_neighbors, active = _triplet_interp(cleaned, ordinals, idx_arr, n, h, w)
            has_triplet = active & has_neighbors
            with np.errstate(invalid="ignore"):
                residual = np.where(has_triplet, cleaned - interp, np.nan)
                noise = np.sqrt(np.nanmean(residual ** 2, axis=0, keepdims=True))
                candidates = removed & has_neighbors
                restore_residual = np.where(candidates, cube - interp, np.nan)  # original value
            is_candidate = candidates & (np.abs(restore_residual) <= below_noise * noise)
            any_candidate = is_candidate.any(axis=0, keepdims=True)
            if not any_candidate.any():
                break
            best = np.argmin(np.where(is_candidate, np.abs(restore_residual), np.inf), axis=0, keepdims=True)
            this_pass = (idx_arr == best) & any_candidate
            cleaned = np.where(this_pass, cube, cleaned).astype("float32")
            removed &= ~this_pass

    return cleaned, removed


# ---------------------------------------------------------------------
# Interpolation: RBF_SIGMA / RBF_CUTOFF
# ---------------------------------------------------------------------

def _erfinv(x, tol=1e-12, max_iter=100):
    lo, hi = -6.0, 6.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if math.erf(mid) < x:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def rbf_cutoff_radius(sigma_days, cutoff):
    """One-sided truncation radius (days) keeping `cutoff` fraction of a
    Gaussian kernel's area, i.e. FORCE's RBF_CUTOFF: cutoff = erf(radius
    / (sigma * sqrt(2))), solved for radius."""
    return sigma_days * math.sqrt(2.0) * _erfinv(cutoff)


def rbf_interpolate(dates, cube, output_dates, rbf_sigma, rbf_cutoff=0.95):
    """FORCE's RBF interpolation: for each output date, each sigma in
    RBF_SIGMA is convolved independently (weight = Gaussian, truncated at
    its own RBF_CUTOFF-derived radius), then combined by data-density
    weighting -- sigmas that found more real nearby data dominate the
    result automatically. If no sigma finds any real data within its
    cutoff radius, the output is NaN: that's the only nodata rule: no
    separate gap/delay gating on top of it.
    """
    ordinals = np.array([d.toordinal() for d in dates], dtype="float64")
    valid = ~np.isnan(cube)
    valid_f = valid.astype("float64")
    filled = np.where(valid, cube, 0.0).astype("float64")

    sigmas = np.asarray(rbf_sigma, dtype="float64")
    radii = np.array([rbf_cutoff_radius(s, rbf_cutoff) for s in sigmas])
    max_radius = radii.max()

    h, w = cube.shape[1:]
    out = np.full((len(output_dates), h, w), np.nan, dtype="float32")

    for oi, od in enumerate(output_dates):
        target = od.toordinal()
        idx = np.where(np.abs(ordinals - target) <= max_radius)[0]
        if idx.size == 0:
            continue

        deltas = ordinals[idx] - target                      # (window,)
        weight = np.exp(-0.5 * (deltas[None, :] / sigmas[:, None]) ** 2)
        within = np.abs(deltas)[None, :] <= radii[:, None]
        weight = np.where(within, weight, 0.0)                # (n_sigma, window)

        mass = np.tensordot(weight, valid_f[idx], axes=([1], [0]))   # (n_sigma, h, w)
        wsum = np.tensordot(weight, filled[idx], axes=([1], [0]))    # (n_sigma, h, w)

        total_mass = mass.sum(axis=0)
        total_wsum = wsum.sum(axis=0)
        has_data = total_mass > 0
        out[oi] = np.where(has_data, total_wsum / np.where(has_data, total_mass, 1.0), np.nan)

    return out


# ---------------------------------------------------------------------
# Output calendar: INT_DAY / DATE_RANGE / DOY_RANGE
# ---------------------------------------------------------------------

def output_dates_from_range(date_range, int_day, doy_range=(1, 365)):
    """Every int_day days across date_range, kept only where day-of-year
    falls in doy_range. doy_range supports FORCE's wrap-around convention
    (e.g. (274, 90) means "Oct-Mar", spanning the year boundary)."""
    start, end = date_range
    doy_lo, doy_hi = doy_range
    dates = []
    d = start
    while d <= end:
        doy = d.timetuple().tm_yday
        keep = (doy_lo <= doy <= doy_hi) if doy_lo <= doy_hi else (doy >= doy_lo or doy <= doy_hi)
        if keep:
            dates.append(d)
        d += timedelta(days=int_day)
    return dates


# ---------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------

def write_tsi(path, stack, transform, crs, dates, nodata=np.nan):
    n, h, w = stack.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=n,
                        dtype="float32", crs=crs, transform=transform,
                        nodata=nodata, compress="deflate") as dst:
        for i in range(n):
            dst.write(stack[i], i + 1)
            dst.set_band_description(i + 1, dates[i].isoformat())


# ---------------------------------------------------------------------
# Orchestration: one TSS tif -> one TSI tif
# ---------------------------------------------------------------------

def run_tsi(tss_path, out_path, date_range, doy_range=(1, 365), int_day=5,
            rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
            above_noise=3.0, below_noise=1.0):
    dates, cube, transform, crs = load_tss(tss_path)
    cleaned, _removed = despike(dates, cube, above_noise, below_noise)
    output_dates = output_dates_from_range(date_range, int_day, doy_range)
    stack = rbf_interpolate(dates, cleaned, output_dates, rbf_sigma, rbf_cutoff)
    write_tsi(out_path, stack, transform, crs, output_dates)
    return output_dates

# Example (from a notebook):
#
#   from force_tsi import run_tsi
#   from datetime import date
#
#   output_dates = run_tsi(
#       "NDVI_TSS.tif", "NDVI_TSI.tif",
#       date_range=(date(2017, 1, 1), date(2023, 12, 31)),
#       doy_range=(1, 365), int_day=5,
#       rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
#       above_noise=3.0, below_noise=1.0,
#   )
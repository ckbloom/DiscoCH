"""
rbf_interp.py

Builds a 5-day, interpolated VI time series using an
ensemble of symmetric Gaussian-kernel convolutions (Schwieder et al. 2016;
Hemmerling et al. 2021). Kernels stay symmetric (look both forward and
backward); operational delay is handled via `wait_days`, which postpones
computing each output date until enough real data exists on both sides.

Ensemble weighting: each kernel width is convolved independently to get
value_w (weighted mean) and mass_w (realized weight mass captured). Final
value = sum(mass_w * value_w) / sum(mass_w) -- kernels that found more real
nearby data dominate automatically. If all mass_w == 0, output is NaN.

Isolated-point fallback: normally a target date needs real data on BOTH
sides (backward_ok AND forward_ok, see delayed_smooth_one_date()) or it's
NaN -- this is what makes a real data gap read as a gap. But a genuinely
isolated point (real data on only one side, with nothing reasonably
nearby on the other side either) would otherwise never produce a value
anywhere, even right next to itself. So a one-sided value IS emitted when
the succeeding side passes its own gate AND the failing side's nearest
real observation is beyond `isolation_radius_days` -- far enough that
this isn't just an ordinary gap edge (where the far side is only
modestly beyond its own tight gate, and should stay NaN).

Truncation: each kernel's own per-side truncation radius is derived from
its width via `DEFAULT_RBF_CUTOFF`, matching FORCE's RBF_CUTOFF semantics
(see _rbf_cutoff_radius_days()) -- a 10d-sigma kernel is truncated much
tighter than a 50d-sigma one, not to one flat radius shared by every width.
`DEFAULT_MAX_RADIUS_DAYS` remains as an outer safety cap (and sets how much
raw data is even fetched for a target date): the radius actually used per
kernel is min(DEFAULT_MAX_RADIUS_DAYS, cutoff-derived radius).
"""

import os
import glob
import math
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from src.disco_ch.force_pull import (
    DATE_RE_COMPACT as DATE_RE,
    find_vi_file,
    discover_tiles,
    VI_KEYS,
    _resolve_vi_keys,
    DEFAULT_FORCE_GRID_CSV,
)

# Kernel widths (days) matching the original RBF-smoothed training data.
DEFAULT_WIDTHS_DAYS = (10, )

# Days to wait after a target date before computing it operationally
# (gives the ~5-day satellite revisit a chance to deliver a forward obs).
DEFAULT_WAIT_DAYS = 10

# Outer safety cap (per side) on how far any kernel may draw observations
# from, and on how much raw data is fetched for a target date at all. The
# radius actually applied per kernel is min(this, its own RBF_CUTOFF-derived
# radius -- see _rbf_cutoff_radius_days()), so this constant only binds for
# unusually wide kernel widths; for FORCE's default widths (<=50d) it never
# does (see DEFAULT_RBF_CUTOFF).
DEFAULT_MAX_RADIUS_DAYS = 100

# Fraction of a Gaussian kernel's area to keep before truncating its tails,
# matching FORCE's RBF_CUTOFF: each width's own truncation radius is
# width_days * sqrt(2) * erfinv(cutoff) (see _rbf_cutoff_radius_days()) --
# e.g. a 10d kernel is truncated at ~19.6d, a 50d kernel at ~98d, not both
# capped identically.
DEFAULT_RBF_CUTOFF = 0.95

# Max allowed gap to the most recent real observation before target_date.
DEFAULT_MAX_BACKWARD_GAP_DAYS = 20

# How far away (days) the OTHER side must be -- when one side already
# fails its own gate above -- before a target date is treated as
# genuinely isolated (see delayed_smooth_one_date()'s isolated-point
# fallback) rather than just an ordinary gap. Deliberately much larger
# than both max_backward_gap_days and wait_days, so a normal gap edge
# (where the far side is still only modestly beyond its own tight gate)
# is left as NaN, and only a point with a long empty stretch on the
# other side qualifies for the one-sided fallback.
DEFAULT_ISOLATION_RADIUS_DAYS = 45

# Triplet-residual despiking (see despike_daily_cube()): an observation is
# removed once its residual exceeds this many times the per-pixel noise
# (RMSE of triplet residuals). Only the single worst-offending point per
# pixel is removed per pass, so max_iter should cover the largest number
# of real outliers expected in one pixel's series.
DEFAULT_DESPIKE_THRESHOLD_FACTOR = 3.0
DEFAULT_DESPIKE_MAX_ITER = 20

# After removal converges, a previously-removed point is restored if its
# ORIGINAL value now fits within this many multiples of the (tightened)
# noise estimate -- matching FORCE's BELOW_NOISE, which restores a
# QAI-masked observation that turns out to fit the local trend well. This
# codebase has no separate QAI mask to draw restoration candidates from
# (masked-out observations are simply absent from daily_cube upstream), so
# the candidate pool here is despike_daily_cube()'s own removed points
# instead -- the closest available analog, not a literal port. 0 disables
# restoration entirely (nothing satisfies a <= 0 residual bound in practice).
DEFAULT_DESPIKE_BELOW_NOISE_FACTOR = 1.0


# ---------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------

def _collapse_bands_to_daily(arr, descriptions):
    """Parse each band's date from its description; average same-day duplicates."""
    band_dates = []
    keep_idx = []
    for i, desc in enumerate(descriptions):
        if not desc:
            continue
        m = DATE_RE.match(desc)
        if not m:
            continue
        band_dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
        keep_idx.append(i)

    if not band_dates:
        raise ValueError("No dated bands found in raster")

    cube = arr[keep_idx]
    dates = sorted(set(band_dates))
    daily_cube = np.full((len(dates), cube.shape[1], cube.shape[2]), np.nan, dtype="float32")
    for out_i, d in enumerate(dates):
        same_day_idx = [i for i, dd in enumerate(band_dates) if dd == d]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices expected
            daily_cube[out_i] = np.nanmean(cube[same_day_idx], axis=0)

    return dates, daily_cube


def load_vi_cube(tif_path):
    """Read an entire VI multiband TIF, collapsed to one layer per calendar date.

    Returns: dates, daily_cube (n_dates, h, w), transform, crs
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(masked=True).astype("float32").filled(np.nan)
        transform = src.transform
        crs = src.crs
        descriptions = src.descriptions

    dates, daily_cube = _collapse_bands_to_daily(arr, descriptions)
    return dates, daily_cube, transform, crs


def load_vi_pixel_series(tif_path, row, col, buffer=0):
    """Like load_vi_cube(), but only reads a small window around (row, col).

    Returns dates, daily_cube (windowed), and pixel row/col within that window.
    """
    with rasterio.open(tif_path) as src:
        row_off = max(row - buffer, 0)
        col_off = max(col - buffer, 0)
        row_end = min(row + buffer + 1, src.height)
        col_end = min(col + buffer + 1, src.width)
        window = Window(col_off=col_off, row_off=row_off,
                         width=col_end - col_off, height=row_end - row_off)
        arr = src.read(window=window, masked=True).astype("float32").filled(np.nan)
        descriptions = src.descriptions

    dates, daily_cube = _collapse_bands_to_daily(arr, descriptions)
    return dates, daily_cube, row - row_off, col - col_off


def load_vi_cube_window(tif_path, window):
    """Like load_vi_cube(), but only reads the given spatial `window`
    (rasterio.windows.Window) -- the read side of chunked/spatial-block
    processing (see chunk_size in delayed_interpolate_series() etc.)."""
    with rasterio.open(tif_path) as src:
        arr = src.read(window=window, masked=True).astype("float32").filled(np.nan)
        descriptions = src.descriptions

    return _collapse_bands_to_daily(arr, descriptions)


def tile_grid_meta(tif_path):
    """(width, height, transform, crs) for a FORCE multiband TIF, without
    reading any pixel data -- used to size/tile chunk_size windows."""
    with rasterio.open(tif_path) as src:
        return src.width, src.height, src.transform, src.crs


# ---------------------------------------------------------------------
# Outlier filtering (despiking)
# ---------------------------------------------------------------------

def _triplet_interp(cleaned, ordinals, idx_arr, n, h, w):
    """Per-position linear interpolation between each position's nearest
    ACTIVE (non-NaN in `cleaned`) neighbors on either side in time, plus
    where such a neighbor pair even exists (`has_neighbors`) -- computed
    independent of whether the position itself is active, so the same
    interpolation serves both a removal candidate (itself active,
    despike_daily_cube()'s main loop) and a restoration candidate (itself
    NOT active -- despike_daily_cube()'s below-noise pass). Also returns
    `active` since both callers need it.
    """
    active = ~np.isnan(cleaned)

    # Nearest active index strictly before / after each position i, via
    # the same forward/backward-fill trick _precompute_daily_stats() uses
    # for ordinals -- here we track the source *index* itself (via
    # take_along_axis below) so a removed point's old neighbors correctly
    # become adjacent to each other once it's gone.
    idx_if_active = np.where(active, idx_arr, -1)
    last_idx_incl = np.maximum.accumulate(idx_if_active, axis=0)
    last_idx = np.concatenate([np.full((1, h, w), -1, dtype=int), last_idx_incl[:-1]], axis=0)

    idx_if_active_rev = np.where(active, idx_arr, n)
    next_idx_incl = np.minimum.accumulate(idx_if_active_rev[::-1], axis=0)[::-1]
    next_idx = np.concatenate([next_idx_incl[1:], np.full((1, h, w), n, dtype=int)], axis=0)

    has_prev = last_idx >= 0
    has_next = next_idx < n

    safe_last = np.clip(last_idx, 0, n - 1)
    safe_next = np.clip(next_idx, 0, n - 1)

    prev_val = np.take_along_axis(cleaned, safe_last, axis=0)
    next_val = np.take_along_axis(cleaned, safe_next, axis=0)
    prev_ord = ordinals[safe_last]
    next_ord = ordinals[safe_next]
    cur_ord = np.broadcast_to(ordinals[:, None, None], (n, h, w))

    span = (next_ord - prev_ord).astype("float64")
    has_neighbors = has_prev & has_next & (span > 0)
    safe_span = np.where(span > 0, span, 1.0)
    frac = (cur_ord - prev_ord) / safe_span

    with warnings.catch_warnings():
        # Pixels with zero real observations anywhere (e.g. outside the
        # forest mask) gather NaN for both prev_val/next_val here --
        # harmless (has_neighbors is already False there, so callers
        # discard it), but NaN arithmetic still trips numpy's "invalid
        # value" warning on the add/multiply.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        interp = prev_val + (next_val - prev_val) * frac

    return interp, has_neighbors, active


def despike_daily_cube(dates, daily_cube, threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                        max_iter=DEFAULT_DESPIKE_MAX_ITER,
                        below_noise_factor=DEFAULT_DESPIKE_BELOW_NOISE_FACTOR):
    """
    Iterative triplet-residual despiking, matching the QC method used on
    the original training data: for every real observation that has a
    real neighbor on both sides, compare it to a linear interpolation
    between those two neighbors (in time). The RMSE of these residuals,
    computed per pixel across its whole series, is that pixel's noise
    level; the single observation whose residual exceeds
    `threshold_factor` times the noise level by the widest margin is
    removed. Removing it changes its old neighbors' triplets (they become
    each other's new neighbors) and shrinks the noise estimate (that
    point no longer contaminates it, and neither does the artificially
    inflated residual it was causing in its neighbors), so residuals/
    noise/threshold are recomputed and the check repeats -- up to
    `max_iter` passes, or until nothing exceeds the threshold.

    Removing only the single worst point per pixel per pass (rather than
    every currently-exceeding point at once) matters: an outlier inflates
    not just its own residual but its immediate neighbors' too (they each
    interpolate against it), so batch-removing every flagged point in one
    pass lets multiple/adjacent outliers inflate the shared noise estimate
    enough that the threshold stops catching any of them. One-at-a-time
    removal always clears the true outlier first (its own residual is the
    full deviation from the smooth curve; a contaminated neighbor's is
    only ever a fraction of that, from averaging one clean side against
    the outlier), and the noise estimate tightens on every subsequent pass
    as contamination is progressively cleared.

    A pixel's first/last real observation, or any point whose only real
    neighbor is on one side, is never flagged -- there's no triplet to
    test it against. Pixels without at least one full triplet (fewer
    than 3 real observations, or all remaining ones share a date/ordinal)
    are left untouched -- their noise estimate would be undefined (NaN),
    which naturally fails every ">" comparison below.

    Once removal converges, a below-noise restoration pass runs (mirrors
    FORCE's BELOW_NOISE): a removed point is restored -- using its
    ORIGINAL value, not recomputed -- if that value now falls within
    `below_noise_factor` times the (by-then-tightened) noise estimate of
    a fresh triplet built from its neighbors' *current* state. Only the
    single best-fitting (smallest residual) restoration candidate per
    pixel is applied per pass, then noise/candidates are recomputed and
    the check repeats, mirroring the removal loop's one-at-a-time
    rationale: restoring a point changes its old neighbors' triplets too,
    so resolving one at a time keeps the result independent of iteration
    order. Removal always fully converges first, then restoration runs
    against that final state, rather than interleaving pass-by-pass --
    simpler and deterministic, at the cost of not exactly reproducing
    FORCE's per-pass interleaving (undocumented in the param file anyway).
    Unlike FORCE's BELOW_NOISE (which restores QAI-masked observations
    that were never actually contamination -- just flagged by a
    conservative cloud/shadow/etc. mask), this has no separate QAI-masked
    pool to draw from: upstream masking already discards those values
    before they ever reach daily_cube. The candidate pool here is
    despike_daily_cube()'s own removed points instead -- the closest
    available analog, not a literal port of FORCE's mechanism.

    :param dates: ascending list of `date` objects, one per daily_cube slice.
    :param daily_cube: (n_dates, h, w) float array, NaN where no real
        observation exists (see load_vi_cube() / load_vi_cube_from_archive()).
    :param threshold_factor: residual-to-noise ratio above which an
        observation is removed.
    :param max_iter: safety cap on despiking passes (removal and
        restoration each get their own budget of up to `max_iter` passes)
        -- since at most one point per pixel is removed/restored per pass,
        this should be at least the largest number of real outliers you'd
        expect in a single pixel's series.
    :param below_noise_factor: residual-to-noise ratio within which a
        previously-removed observation is restored -- see above. 0
        effectively disables restoration (no real-valued residual
        satisfies a <= 0 bound in practice).
    :return: (cleaned_cube, removed_mask) -- `daily_cube` is not modified
        in place; `removed_mask` (same shape, bool) marks every entry
        still excluded after both passes.
    """
    ordinals = np.array([d.toordinal() for d in dates], dtype="int64")
    n, h, w = daily_cube.shape

    cleaned = daily_cube.copy()
    removed = np.zeros((n, h, w), dtype=bool)
    idx_arr = np.broadcast_to(np.arange(n)[:, None, None], (n, h, w))

    for _ in range(max_iter):
        interp, has_neighbors, active = _triplet_interp(cleaned, ordinals, idx_arr, n, h, w)
        has_triplet = active & has_neighbors

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            residual = np.where(has_triplet, cleaned - interp, np.nan)
            noise = np.sqrt(np.nanmean(residual ** 2, axis=0, keepdims=True))  # (1, h, w)

        is_candidate = has_triplet & (np.abs(residual) > threshold_factor * noise)
        has_any_candidate = is_candidate.any(axis=0, keepdims=True)  # (1, h, w)
        if not has_any_candidate.any():
            break

        # Per pixel, remove only the single largest-|residual| candidate.
        residual_for_argmax = np.where(is_candidate, np.abs(residual), -np.inf)
        worst_idx = np.argmax(residual_for_argmax, axis=0, keepdims=True)  # (1, h, w)
        this_pass = (idx_arr == worst_idx) & has_any_candidate

        cleaned = np.where(this_pass, np.nan, cleaned).astype("float32")
        removed |= this_pass

    if below_noise_factor > 0:
        for _ in range(max_iter):
            if not removed.any():
                break

            interp, has_neighbors, active = _triplet_interp(cleaned, ordinals, idx_arr, n, h, w)
            has_triplet = active & has_neighbors

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                residual = np.where(has_triplet, cleaned - interp, np.nan)
                noise = np.sqrt(np.nanmean(residual ** 2, axis=0, keepdims=True))  # (1, h, w)

                # Candidates: currently-removed points, tested against
                # their ORIGINAL value (not `cleaned`, which is NaN there)
                # against the triplet formed by whatever's active right now.
                restore_candidates = removed & has_neighbors
                restore_residual = np.where(restore_candidates, daily_cube - interp, np.nan)

            is_candidate = restore_candidates & (np.abs(restore_residual) <= below_noise_factor * noise)
            has_any_candidate = is_candidate.any(axis=0, keepdims=True)  # (1, h, w)
            if not has_any_candidate.any():
                break

            # Per pixel, restore only the single best-fitting (smallest
            # |residual|) candidate.
            residual_for_argmin = np.where(is_candidate, np.abs(restore_residual), np.inf)
            best_idx = np.argmin(residual_for_argmin, axis=0, keepdims=True)  # (1, h, w)
            this_pass = (idx_arr == best_idx) & has_any_candidate

            cleaned = np.where(this_pass, daily_cube, cleaned).astype("float32")
            removed &= ~this_pass

    return cleaned, removed


def debug_despike_pixel(dates, values, threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                         max_iter=DEFAULT_DESPIKE_MAX_ITER, top_n=5):
    """
    Single-pixel, verbose trace of despike_daily_cube()'s removal pass --
    prints each pass's noise estimate/threshold, the top `top_n` residuals
    by magnitude (not just the single worst), and, if anything exceeds
    threshold, which date got removed and why. Same logic as
    despike_daily_cube()'s removal loop, just unvectorized and talkative,
    for understanding why a specific pixel's outlier is/isn't being
    flagged (e.g. via plot_pixel_comparison(..., debug_pixel=True)) --
    seeing the next-largest residuals alongside the threshold tells you
    how close a borderline point is, and whether other suspect points
    exist at all. Does NOT trace despike_daily_cube()'s below-noise
    restoration pass -- a point this prints as removed may still end up
    restored in the vectorized function's actual output.

    :param dates: ascending list of `date` objects.
    :param values: 1D array/list of VI values for one pixel, NaN where no
        real observation.
    :param top_n: how many of the largest-|residual| points to print per pass.
    :return: (cleaned_values, removed_mask), same convention as
        despike_daily_cube() but for this one pixel.
    """
    values = np.asarray(values, dtype="float64").copy()
    ordinals = np.array([d.toordinal() for d in dates], dtype="int64")
    n = len(values)
    removed = np.zeros(n, dtype=bool)

    for it in range(max_iter):
        active_idx = np.where(~np.isnan(values))[0]
        if len(active_idx) < 3:
            print(f"  pass {it}: fewer than 3 real observations remain -- stopping.")
            break

        residuals = np.full(n, np.nan)
        for k in range(1, len(active_idx) - 1):
            i, i_prev, i_next = active_idx[k], active_idx[k - 1], active_idx[k + 1]
            span = ordinals[i_next] - ordinals[i_prev]
            if span <= 0:
                continue
            frac = (ordinals[i] - ordinals[i_prev]) / span
            interp = values[i_prev] + (values[i_next] - values[i_prev]) * frac
            residuals[i] = values[i] - interp

        if np.all(np.isnan(residuals)):
            print(f"  pass {it}: no eligible triplets -- stopping.")
            break

        noise = np.sqrt(np.nanmean(residuals ** 2))
        threshold = threshold_factor * noise
        eligible_idx = np.where(~np.isnan(residuals))[0]
        ranked = eligible_idx[np.argsort(-np.abs(residuals[eligible_idx]))][:top_n]

        print(f"  pass {it}: noise(RMSE)={noise:.4f}, threshold({threshold_factor}x)={threshold:.4f}, "
              f"top {len(ranked)} residuals:")
        for i in ranked:
            flag = " <-- exceeds threshold" if abs(residuals[i]) > threshold else ""
            print(f"      {dates[i]}: value={values[i]:.4f}, residual={residuals[i]:+.4f}{flag}")

        worst_i = int(ranked[0])
        if abs(residuals[worst_i]) <= threshold:
            print("  -> largest residual does not exceed threshold, stopping.")
            break

        print(f"  -> removing {dates[worst_i]} (residual {residuals[worst_i]:+.4f} > threshold {threshold:.4f})")
        values[worst_i] = np.nan
        removed[worst_i] = True

    return values, removed


def despike_daily_cubes_union(dates, daily_cubes_by_vi, threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                               max_iter=DEFAULT_DESPIKE_MAX_ITER):
    """
    Cross-VI despiking. Residual cloud/shadow/atmospheric contamination
    corrupts the shared raw spectral bands, not one VI's formula in
    isolation -- a contaminated date can distort NDVI/CIRE/etc. enough to
    clear their own noise threshold while barely nudging a VI like CCI
    that happens to be less sensitive to it at that pixel, or vice versa.
    Despiking each VI purely against its own residuals (despike_daily_cube())
    misses that: the "clean-looking" VI still feeds the contaminated raw
    value into interpolation and the model.

    This runs despike_daily_cube() once per VI to get each one's
    independent verdict, takes the union of every (pixel, date) any VI
    flagged, and applies that same exclusion to every VI's cube -- so one
    VI's detection protects all of them.

    :param dates: shared ascending list of `date` objects. All VIs in
        `daily_cubes_by_vi` are assumed to already share this exact date
        axis and grid -- true for both pipelines here, since FORCE's
        per-tile TSA product and stac_pull's archive both compute/archive
        every VI together from the same acquisition.
    :param daily_cubes_by_vi: {vi_key: (n_dates,h,w) daily_cube}.
    :param threshold_factor, max_iter: see despike_daily_cube(). Each VI's
        below-noise restoration pass runs with despike_daily_cube()'s own
        default (DEFAULT_DESPIKE_BELOW_NOISE_FACTOR) -- not exposed here --
        before the union mask below is taken, so a point one VI's own
        residual test flags and then restores never reaches the union.
    :return: ({vi_key: cleaned_cube}, union_removed_mask (n_dates,h,w)) --
        none of the input cubes are modified in place.
    """
    removed_masks = {
        vi_key: despike_daily_cube(dates, cube, threshold_factor, max_iter)[1]
        for vi_key, cube in daily_cubes_by_vi.items()
    }

    union_removed = np.zeros_like(next(iter(removed_masks.values())))
    for mask in removed_masks.values():
        union_removed = union_removed | mask

    cleaned = {
        vi_key: np.where(union_removed, np.nan, cube).astype("float32")
        for vi_key, cube in daily_cubes_by_vi.items()
    }
    return cleaned, union_removed


# ---------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------

def gaussian_weight(delta_days, width_days):
    """Symmetric Gaussian weight for an observation delta_days from the output date."""
    return np.exp(-0.5 * (delta_days / width_days) ** 2)


def _erfinv(x, tol=1e-12, max_iter=100):
    """Inverse error function via bisection. Only used to turn RBF_CUTOFF
    into a truncation radius (see _rbf_cutoff_radius_days()) -- once per
    (width, cutoff) pair, never in a per-pixel hot loop -- so a bisection
    is plenty accurate without adding scipy as a dependency for this one
    call.
    """
    if not (-1.0 < x < 1.0):
        raise ValueError(f"erfinv domain is (-1, 1), got {x}")
    lo, hi = -6.0, 6.0  # math.erf saturates to +-1 well within +-6
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if math.erf(mid) < x:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def _rbf_cutoff_radius_days(width_days, cutoff=DEFAULT_RBF_CUTOFF):
    """One-sided truncation radius (days) for a Gaussian kernel of the
    given width (sigma), reproducing FORCE's RBF_CUTOFF: `cutoff` is the
    fraction of the kernel's area kept within +-radius, i.e.
    cutoff = erf(radius / (width_days * sqrt(2))), solved for radius. E.g.
    cutoff=0.95 -> radius ~= 1.96 * width_days (the familiar two-sided 95%
    bound), vs. a wider cutoff=0.99 -> ~2.576 * width_days.
    """
    return width_days * math.sqrt(2.0) * _erfinv(cutoff)


# ---------------------------------------------------------------------
# Precomputation and per-width smoothing
# ---------------------------------------------------------------------

def _precompute_daily_stats(dates, daily_cube):
    """Computed once per (tile, VI), reused across every output date/width.

    Returns:
        valid_mask: bool cube, True where a real observation exists
        ordinals: date ordinals (n_dates,), ascending
        last_valid_cum: per-pixel running most-recent valid ordinal seen so far (forward-filled)
        next_valid_cum: per-pixel running soonest valid ordinal from here on (backward-filled)
    """
    valid_mask = ~np.isnan(daily_cube)
    ordinals = np.array([d.toordinal() for d in dates], dtype="int64")

    ordinal_if_valid = np.where(valid_mask, ordinals[:, None, None], -1)
    last_valid_cum = np.maximum.accumulate(ordinal_if_valid, axis=0)

    big = np.iinfo("int64").max
    ordinal_if_valid_rev = np.where(valid_mask, ordinals[:, None, None], big)
    next_valid_cum = np.minimum.accumulate(ordinal_if_valid_rev[::-1], axis=0)[::-1]

    return valid_mask, ordinals, last_valid_cum, next_valid_cum


def _ensemble_weighted_sums(daily_cube, valid_mask, ordinals, target_ordinal,
                             widths, start_idx, end_idx, rbf_cutoff=DEFAULT_RBF_CUTOFF,
                             max_radius_days=DEFAULT_MAX_RADIUS_DAYS):
    """Vectorized pass over window [start_idx, end_idx], computing weighted
    sums/masses for ALL kernel widths at once via two tensordot contractions.

    Each width's own tails are truncated at min(max_radius_days,
    _rbf_cutoff_radius_days(width, rbf_cutoff)) -- a hard zero-weight cutoff
    on top of the Gaussian's natural decay, so a narrow kernel doesn't draw
    (numerically negligible but non-zero) weight from observations far
    beyond where FORCE's RBF_CUTOFF would have excluded them.

    Returns:
        values_stack: (n_widths, h, w) weighted mean per width (NaN if zero mass)
        masses_stack: (n_widths, h, w) realized weight mass per width
    """
    sub_ordinals = ordinals[start_idx:end_idx + 1].astype("float64")
    deltas = sub_ordinals - float(target_ordinal)  # (window_size,)
    widths_arr = np.asarray(widths, dtype="float64")  # (n_widths,)

    # (n_widths, window_size) Gaussian weight matrix, one row per width.
    weight_matrix = np.exp(-0.5 * (deltas[None, :] / widths_arr[:, None]) ** 2).astype("float32")

    # Per-width hard truncation (see docstring) -- zero out any weight
    # beyond that width's own cutoff-derived radius.
    cutoff_radius = np.minimum(
        max_radius_days,
        np.array([_rbf_cutoff_radius_days(w, rbf_cutoff) for w in widths], dtype="float64"),
    )
    within_radius = np.abs(deltas)[None, :] <= cutoff_radius[:, None]
    weight_matrix = np.where(within_radius, weight_matrix, 0.0).astype("float32")

    sub_valid = valid_mask[start_idx:end_idx + 1]  # (window_size, h, w)
    sub_valid_f = sub_valid.astype("float32")
    sub_cube = daily_cube[start_idx:end_idx + 1]
    sub_cube_filled = np.where(sub_valid, sub_cube, 0.0).astype("float32")

    # Contract time axis: (n_widths, window_size) . (window_size, h, w) -> (n_widths, h, w)
    mass_stack = np.tensordot(weight_matrix, sub_valid_f, axes=([1], [0]))
    weighted_value_stack = np.tensordot(weight_matrix, sub_cube_filled, axes=([1], [0]))

    has_data = mass_stack > 0
    safe_denom = np.where(has_data, mass_stack, 1.0)
    values_stack = np.where(has_data, weighted_value_stack / safe_denom, np.nan)
    masses_stack = np.where(has_data, mass_stack, 0.0)

    return values_stack.astype("float32"), masses_stack.astype("float32")


def delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                             target_date, widths=DEFAULT_WIDTHS_DAYS,
                             wait_days=DEFAULT_WAIT_DAYS,
                             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                             max_radius_days=DEFAULT_MAX_RADIUS_DAYS,
                             rbf_cutoff=DEFAULT_RBF_CUTOFF,
                             isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS,
                             return_fallback_mask=False):
    """Compute one ensemble-smoothed 2D raster for target_date, using only
    observations available wait_days after that date.

    Normally a pixel is NaN unless it has a real observation within
    max_backward_gap_days before target_date AND within wait_days after it,
    even if the ensemble weighting alone would produce a value -- this is
    what makes a real data gap read as a gap.

    Isolated-point fallback: if exactly one side passes its own gate, the
    pixel is still eligible when the OTHER (failing) side's nearest real
    observation -- checked against the full loaded series, not just this
    call's wait_days/max_radius_days window -- is farther than
    `isolation_radius_days` away (or doesn't exist at all). That's the
    signal this is a genuinely isolated point (nothing reasonably nearby
    on the other side at all) rather than an ordinary gap edge, where the
    far side is still only modestly beyond its own tight gate and should
    stay NaN. `isolation_radius_days` should be set well above both
    max_backward_gap_days and wait_days so it only fires for real
    isolation, not routine gaps.

    max_radius_days/rbf_cutoff together set each kernel's own truncation
    radius -- see _ensemble_weighted_sums()/_rbf_cutoff_radius_days().
    max_radius_days alone still bounds how much raw data is fetched here
    (window_start_ordinal below), so it must be at least as large as the
    widest width's cutoff-derived radius for that width to reach its full
    intended extent.

    :param return_fallback_mask: if True, also returns a bool (h, w) mask
        marking pixels eligible ONLY via the isolated-point fallback (not
        the normal both-sided path) -- for diagnostics (see
        plot_pixel_comparison()); production callers leave this False.
    """
    target_ordinal = target_date.toordinal()
    window_start_ordinal = target_ordinal - max_radius_days
    window_end_ordinal = target_ordinal + min(wait_days, max_radius_days)

    start_idx = int(np.searchsorted(ordinals, window_start_ordinal, side="left"))
    end_idx = int(np.searchsorted(ordinals, window_end_ordinal, side="right")) - 1

    h, w = daily_cube.shape[1:]
    if end_idx < start_idx:
        empty = np.full((h, w), np.nan, dtype="float32")
        if return_fallback_mask:
            return empty, np.zeros((h, w), dtype=bool)
        return empty

    values_stack, masses_stack = _ensemble_weighted_sums(
        daily_cube, valid_mask, ordinals, target_ordinal, widths, start_idx, end_idx,
        rbf_cutoff=rbf_cutoff, max_radius_days=max_radius_days,
    )

    total_mass = masses_stack.sum(axis=0)
    has_any = total_mass > 0
    safe_total = np.where(has_any, total_mass, 1.0)

    # Zero out NaN values before summing so NaN*0 doesn't poison the total.
    values_for_sum = np.where(masses_stack > 0, values_stack, 0.0)
    ensemble_value = np.sum(masses_stack * values_for_sum, axis=0) / safe_total

    # Backward coverage: most recent real obs up to target must be within gap limit.
    backward_idx = min(int(np.searchsorted(ordinals, target_ordinal, side="right")) - 1, end_idx)
    if backward_idx < 0:
        last_valid_ordinal = np.full((h, w), -1, dtype="int64")
    else:
        last_valid_ordinal = last_valid_cum[backward_idx]
    backward_gap = np.where(last_valid_ordinal >= 0, target_ordinal - last_valid_ordinal, np.inf)
    backward_ok = backward_gap <= max_backward_gap_days

    # Forward coverage: soonest real obs from target onward must exist in window.
    forward_idx = max(int(np.searchsorted(ordinals, target_ordinal, side="left")), start_idx)
    if forward_idx > end_idx:
        next_valid_ordinal = np.full((h, w), np.iinfo("int64").max, dtype="int64")
    else:
        next_valid_ordinal = next_valid_cum[forward_idx]
    forward_ok = next_valid_ordinal <= window_end_ordinal

    eligible_normal = has_any & backward_ok & forward_ok

    # Isolated-point fallback -- re-checks whichever side failed above,
    # but against the FULL loaded series (not the wait_days/max_radius_days
    # window), to tell a genuinely isolated point apart from an ordinary
    # gap edge (see docstring).
    big = np.iinfo("int64").max
    n = len(ordinals)

    true_forward_idx = int(np.searchsorted(ordinals, target_ordinal, side="left"))
    true_next_valid_ordinal = next_valid_cum[true_forward_idx] if true_forward_idx < n else np.full((h, w), big, dtype="int64")
    true_forward_gap = np.where(true_next_valid_ordinal < big, true_next_valid_ordinal - target_ordinal, np.inf)

    backward_isolated = has_any & backward_ok & ~forward_ok & (true_forward_gap > isolation_radius_days)
    forward_isolated = has_any & forward_ok & ~backward_ok & (backward_gap > isolation_radius_days)

    isolated_fallback = backward_isolated | forward_isolated
    eligible = eligible_normal | isolated_fallback

    result = np.where(eligible, ensemble_value, np.nan).astype("float32")
    if return_fallback_mask:
        return result, isolated_fallback
    return result


# ---------------------------------------------------------------------
# Growing-season output calendar
# ---------------------------------------------------------------------

def growing_season_dates(year, season_start="05-01", season_end="07-24", step_days=5):
    """Every step_days days from season_start to season_end (inclusive), for year."""
    start = datetime.strptime(f"{year}-{season_start}", "%Y-%m-%d").date()
    end = datetime.strptime(f"{year}-{season_end}", "%Y-%m-%d").date()
    n_steps = (end - start).days // step_days + 1
    return [start + timedelta(days=step_days * i) for i in range(n_steps)]


def eligible_output_dates(all_output_dates, as_of_date, wait_days=DEFAULT_WAIT_DAYS):
    """Of a full season's output dates, return those eligible as of as_of_date
    (target_date + wait_days <= as_of_date). Enables recurring/rolling runs."""
    return [d for d in all_output_dates if d + timedelta(days=wait_days) <= as_of_date]


# ---------------------------------------------------------------------
# Full series / tile / year orchestration
# ---------------------------------------------------------------------

def _delayed_interpolate_stack(dates, daily_cube, output_dates, widths, wait_days,
                                max_backward_gap_days, max_radius_days, despike=True,
                                despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
                                isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Shared core of delayed_interpolate_series() / delayed_interpolate_series_from_archive():
    given an already-loaded (dates, daily_cube), optionally despikes it
    (see despike_daily_cube()) and runs delayed_smooth_one_date() over
    every output date."""
    if despike:
        daily_cube, _removed = despike_daily_cube(dates, daily_cube, despike_threshold_factor, despike_max_iter)

    valid_mask, ordinals, last_valid_cum, next_valid_cum = _precompute_daily_stats(dates, daily_cube)
    return np.stack([
        delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                 d, widths, wait_days, max_backward_gap_days, max_radius_days,
                                 isolation_radius_days=isolation_radius_days)
        for d in output_dates
    ])


def _iter_chunk_windows(width, height, chunk_size):
    """Yield rasterio Windows tiling a (height, width) raster into
    chunk_size x chunk_size spatial blocks (edge blocks are smaller),
    row-major."""
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _delayed_interpolate_series_generic(path, output_dates, load_full_fn, load_window_fn, meta_fn,
                                         widths, wait_days, max_backward_gap_days, max_radius_days,
                                         despike, despike_threshold_factor, despike_max_iter,
                                         chunk_size, isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Shared chunked-or-not core of delayed_interpolate_series() (FORCE
    TIF) and delayed_interpolate_series_from_archive() (STAC archive dir).

    chunk_size=None (default) loads and processes the whole tile at once,
    exactly as before. An integer chunk_size instead processes the tile in
    chunk_size x chunk_size spatial windows, one at a time: every array
    despike_daily_cube()/_precompute_daily_stats() build is shaped
    (n_dates, h, w), so shrinking h/w from the full tile down to a chunk
    (rather than reducing anything about despike=True/False or n_dates) is
    what actually caps peak memory -- despiking and the ensemble smoothing
    are otherwise untouched, just run once per chunk instead of once for
    the whole tile.
    """
    if chunk_size is None:
        dates, daily_cube, transform, crs = load_full_fn(path)
        stack = _delayed_interpolate_stack(
            dates, daily_cube, output_dates, widths, wait_days,
            max_backward_gap_days, max_radius_days, despike,
            despike_threshold_factor, despike_max_iter, isolation_radius_days,
        )
        return stack, transform, crs

    width, height, transform, crs = meta_fn(path)
    stack = np.full((len(output_dates), height, width), np.nan, dtype="float32")
    for window in _iter_chunk_windows(width, height, chunk_size):
        dates, cube = load_window_fn(path, window)
        sub_stack = _delayed_interpolate_stack(
            dates, cube, output_dates, widths, wait_days,
            max_backward_gap_days, max_radius_days, despike,
            despike_threshold_factor, despike_max_iter, isolation_radius_days,
        )
        row0, col0 = window.row_off, window.col_off
        stack[:, row0:row0 + window.height, col0:col0 + window.width] = sub_stack
    return stack, transform, crs


def delayed_interpolate_series(tif_path, output_dates, widths=DEFAULT_WIDTHS_DAYS,
                                wait_days=DEFAULT_WAIT_DAYS,
                                max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                max_radius_days=DEFAULT_MAX_RADIUS_DAYS, despike=True,
                                despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
                                chunk_size=None,
                                isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Run delayed_smooth_one_date() over every output date for one VI TIF.
    See despike_daily_cube() for the despike_* params -- despike=True (the
    default) removes triplet-residual outliers before smoothing; despike=False
    skips it entirely (no despike_daily_cube() call, no despike-related
    memory cost).
    :param chunk_size: see _delayed_interpolate_series_generic() -- None
        (default) processes the whole tile at once; an integer pixel size
        processes it in chunk_size x chunk_size spatial windows to cap peak
        memory, independent of the despike setting.
    :param isolation_radius_days: see delayed_smooth_one_date()'s
        isolated-point fallback.
    """
    return _delayed_interpolate_series_generic(
        tif_path, output_dates, load_vi_cube, load_vi_cube_window, tile_grid_meta,
        widths, wait_days, max_backward_gap_days, max_radius_days, despike,
        despike_threshold_factor, despike_max_iter, chunk_size, isolation_radius_days,
    )


def load_vi_cube_from_archive(archive_dir):
    """Directory-archive equivalent of load_vi_cube().

    Reads one directory of single-band, per-date VI TIFs (filenames
    '<ISO date>.tif' -- e.g. one file appended per newly arrived STAC
    scene, see stac_pull.archive_vi_raster()) into the same (dates,
    daily_cube, transform, crs) shape load_vi_cube() returns from a
    premade multiband FORCE TSA TIF, so every downstream RBF function
    works identically regardless of which pipeline produced the source
    data.
    """
    paths = sorted(glob.glob(os.path.join(archive_dir, "*.tif")))

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
    around (row, col) from each per-date file.

    Returns dates, daily_cube (windowed), and pixel row/col within that window.
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
    `window` (rasterio.windows.Window) from each per-date file -- the read
    side of chunked/spatial-block processing (see chunk_size in
    delayed_interpolate_series_from_archive() etc.)."""
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
    """(width, height, transform, crs) for a STAC/swisseo per-date archive
    directory, from its first dated file, without reading any pixel data --
    used to size/tile chunk_size windows."""
    paths = sorted(glob.glob(os.path.join(archive_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No dated VI rasters found in {archive_dir}")
    with rasterio.open(paths[0]) as src:
        return src.width, src.height, src.transform, src.crs


def delayed_interpolate_series_from_archive(archive_dir, output_dates, widths=DEFAULT_WIDTHS_DAYS,
                                             wait_days=DEFAULT_WAIT_DAYS,
                                             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                             max_radius_days=DEFAULT_MAX_RADIUS_DAYS, despike=True,
                                             despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                             despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
                                             chunk_size=None,
                                             isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Archive-directory equivalent of delayed_interpolate_series(), for the
    STAC pipeline where VI dates arrive one scene at a time instead of as a
    premade FORCE TSA product. See despike_daily_cube() for the despike_*
    params -- despike=True (the default) removes triplet-residual outliers
    (e.g. residual cloud/shadow contamination not caught by the STAC cloud
    mask) before smoothing; despike=False skips it entirely.
    :param chunk_size: see _delayed_interpolate_series_generic().
    :param isolation_radius_days: see delayed_smooth_one_date()'s
        isolated-point fallback."""
    return _delayed_interpolate_series_generic(
        archive_dir, output_dates, load_vi_cube_from_archive, load_vi_cube_from_archive_window,
        archive_grid_meta, widths, wait_days, max_backward_gap_days, max_radius_days, despike,
        despike_threshold_factor, despike_max_iter, chunk_size, isolation_radius_days,
    )


def delayed_interpolate_series_multi_vi(vi_paths, output_dates, load_fn, load_window_fn=None, meta_fn=None,
                                         widths=DEFAULT_WIDTHS_DAYS,
                                         wait_days=DEFAULT_WAIT_DAYS,
                                         max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                         max_radius_days=DEFAULT_MAX_RADIUS_DAYS, despike=True,
                                         despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                         despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
                                         cross_vi_despike=False, chunk_size=None,
                                         isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """
    Generic multi-VI core shared by delayed_interpolate_series_for_tile()
    (FORCE) and delayed_interpolate_series_from_archive_multi_vi() (STAC):
    loads every VI via `load_fn` and runs delayed_smooth_one_date() over
    every output date for each VI.

    :param vi_paths: {vi_key: path}, one per VI -- a TIF path for
        load_vi_cube, or an archive directory for load_vi_cube_from_archive.
    :param load_fn: load_vi_cube or load_vi_cube_from_archive.
    :param load_window_fn, meta_fn: the windowed-read/grid-metadata
        counterparts of `load_fn` (load_vi_cube_window/tile_grid_meta, or
        load_vi_cube_from_archive_window/archive_grid_meta) -- only used
        when chunk_size is not None.
    :param cross_vi_despike: if True, despike every VI jointly (see
        despike_daily_cubes_union() -- a date flagged by any one VI's own
        residual test is excluded from every VI, since contamination
        corrupts the shared raw bands, not one VI's formula in isolation).
        This requires every VI's raw daily cube resident in memory at once
        (all 5 VIs' full tile-sized cubes), which is what makes despiking a
        3000x3000 FORCE tile with a full year of scenes memory-heavy.
        Default False: each VI is despiked independently via
        despike_daily_cube() and processed fully one at a time (load,
        despike, interpolate, discard) before moving to the next VI, so
        only one VI's cube is ever resident -- the same footprint as
        delayed_interpolate_series() had before cross-VI despiking existed.
    :param chunk_size: see _delayed_interpolate_series_generic() -- None
        (default) processes each VI's whole tile at once; an integer pixel
        size processes chunk_size x chunk_size spatial windows at a time,
        for both the per-VI and cross_vi_despike branches.
    :param isolation_radius_days: see delayed_smooth_one_date()'s
        isolated-point fallback.
    :return: {vi_key: (stack, transform, crs)}
    """
    if not cross_vi_despike:
        results = {}
        for vi_key, path in vi_paths.items():
            stack, transform, crs = _delayed_interpolate_series_generic(
                path, output_dates, load_fn, load_window_fn, meta_fn,
                widths, wait_days, max_backward_gap_days, max_radius_days, despike,
                despike_threshold_factor, despike_max_iter, chunk_size, isolation_radius_days,
            )
            results[vi_key] = (stack, transform, crs)
        return results

    if chunk_size is None:
        dates_ref = None
        cubes = {}
        transforms = {}
        crss = {}
        for vi_key, path in vi_paths.items():
            dates, cube, transform, crs = load_fn(path)
            if dates_ref is None:
                dates_ref = dates
            elif dates != dates_ref:
                raise ValueError(
                    f"delayed_interpolate_series_multi_vi: VI '{vi_key}' has a different date "
                    f"axis than the other VIs -- cross-VI despiking requires every VI to share "
                    f"the same dates/grid (true for both FORCE's per-tile TSA product and "
                    f"stac_pull's archive, which archive every VI together per acquisition)."
                )
            cubes[vi_key] = cube
            transforms[vi_key] = transform
            crss[vi_key] = crs

        if despike:
            cubes, _union_removed = despike_daily_cubes_union(
                dates_ref, cubes, despike_threshold_factor, despike_max_iter,
            )

        results = {}
        for vi_key, cube in cubes.items():
            stack = _delayed_interpolate_stack(
                dates_ref, cube, output_dates, widths, wait_days,
                max_backward_gap_days, max_radius_days, despike=False,
                isolation_radius_days=isolation_radius_days,
            )
            results[vi_key] = (stack, transforms[vi_key], crss[vi_key])
        return results

    # cross_vi_despike=True, chunked: the union mask has to be computed
    # per-window (it needs every VI's cube for that window at once), then
    # each VI is interpolated for that same window before moving on --
    # never holding more than one window's worth of any VI's cube.
    vi_keys = list(vi_paths.keys())
    width, height, transform, crs = meta_fn(vi_paths[vi_keys[0]])
    stacks = {vi_key: np.full((len(output_dates), height, width), np.nan, dtype="float32") for vi_key in vi_keys}

    for window in _iter_chunk_windows(width, height, chunk_size):
        dates_ref = None
        cubes = {}
        for vi_key in vi_keys:
            dates, cube = load_window_fn(vi_paths[vi_key], window)
            if dates_ref is None:
                dates_ref = dates
            elif dates != dates_ref:
                raise ValueError(
                    f"delayed_interpolate_series_multi_vi: VI '{vi_key}' has a different date "
                    f"axis than the other VIs -- cross-VI despiking requires every VI to share "
                    f"the same dates/grid (true for both FORCE's per-tile TSA product and "
                    f"stac_pull's archive, which archive every VI together per acquisition)."
                )
            cubes[vi_key] = cube

        if despike:
            cubes, _union_removed = despike_daily_cubes_union(
                dates_ref, cubes, despike_threshold_factor, despike_max_iter,
            )

        row0, col0 = window.row_off, window.col_off
        for vi_key, cube in cubes.items():
            sub_stack = _delayed_interpolate_stack(
                dates_ref, cube, output_dates, widths, wait_days,
                max_backward_gap_days, max_radius_days, despike=False,
                isolation_radius_days=isolation_radius_days,
            )
            stacks[vi_key][:, row0:row0 + window.height, col0:col0 + window.width] = sub_stack

    return {vi_key: (stacks[vi_key], transform, crs) for vi_key in vi_keys}


def delayed_interpolate_series_for_tile(tif_paths, output_dates, widths=DEFAULT_WIDTHS_DAYS,
                                         wait_days=DEFAULT_WAIT_DAYS,
                                         max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                         max_radius_days=DEFAULT_MAX_RADIUS_DAYS, despike=True,
                                         despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                         despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
                                         cross_vi_despike=False, chunk_size=None,
                                         isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """FORCE multiband-TIF equivalent of delayed_interpolate_series(), for
    every VI of one tile at once. See delayed_interpolate_series_multi_vi()
    for cross_vi_despike (default False: each VI is despiked and processed
    independently, one at a time), chunk_size (default None: whole tile
    at once; an integer pixel size caps peak memory by processing spatial
    windows one at a time), and isolation_radius_days (delayed_smooth_one_date()'s
    isolated-point fallback).

    :param tif_paths: {vi_key: tif_path}, one per VI, from the same tile/year.
    :return: {vi_key: (stack, transform, crs)}
    """
    return delayed_interpolate_series_multi_vi(
        tif_paths, output_dates, load_vi_cube, load_vi_cube_window, tile_grid_meta,
        widths=widths, wait_days=wait_days,
        max_backward_gap_days=max_backward_gap_days, max_radius_days=max_radius_days,
        despike=despike, despike_threshold_factor=despike_threshold_factor,
        despike_max_iter=despike_max_iter, cross_vi_despike=cross_vi_despike,
        chunk_size=chunk_size, isolation_radius_days=isolation_radius_days,
    )


def delayed_interpolate_series_from_archive_multi_vi(archive_dirs, output_dates, widths=DEFAULT_WIDTHS_DAYS,
                                                       wait_days=DEFAULT_WAIT_DAYS,
                                                       max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                                       max_radius_days=DEFAULT_MAX_RADIUS_DAYS, despike=True,
                                                       despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                                       despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
                                                       cross_vi_despike=False, chunk_size=None,
                                                       isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """STAC archive equivalent of delayed_interpolate_series_for_tile(),
    for every VI's archive directory at once. See
    delayed_interpolate_series_multi_vi() for cross_vi_despike (default
    False: each VI is despiked and processed independently, one at a time),
    chunk_size (default None: whole extent at once; an integer pixel
    size caps peak memory by processing spatial windows one at a time),
    and isolation_radius_days (delayed_smooth_one_date()'s isolated-point
    fallback).

    :param archive_dirs: {vi_key: archive_dir}, one per VI (e.g.
        os.path.join(archive_root, vi_key) for each of stac_pull's
        canonical VI keys).
    :return: {vi_key: (stack, transform, crs)}
    """
    return delayed_interpolate_series_multi_vi(
        archive_dirs, output_dates, load_vi_cube_from_archive, load_vi_cube_from_archive_window,
        archive_grid_meta, widths=widths, wait_days=wait_days,
        max_backward_gap_days=max_backward_gap_days, max_radius_days=max_radius_days,
        despike=despike, despike_threshold_factor=despike_threshold_factor,
        despike_max_iter=despike_max_iter, cross_vi_despike=cross_vi_despike,
        chunk_size=chunk_size, isolation_radius_days=isolation_radius_days,
    )


def write_multiband_tif(path, stack, transform, crs, band_dates, nodata=np.nan):
    n_bands, h, w = stack.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=n_bands,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
        compress="deflate",
    ) as dst:
        for i in range(n_bands):
            dst.write(stack[i], i + 1)
            dst.set_band_description(i + 1, band_dates[i].isoformat())


def _existing_rbf_output_dates(out_path):
    """Band dates already present in an existing RBF multiband output, or
    None if out_path doesn't exist or can't be read as such (e.g. missing,
    corrupt, or not actually a multiband raster written by
    write_multiband_tif()) -- in which case it must be (re)computed rather
    than skipped. Descriptions-only read (see write_multiband_tif() --
    band descriptions are the ISO date, not the compact acquisition-date
    format raw VI TIFs use), no pixel data touched."""
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


def _last_raw_band_date(tif_path):
    """Most recent acquisition date actually present in a raw FORCE VI
    TIF, read from band descriptions only (no pixel data) -- the ceiling
    on which output dates can be computed at all right now: any target
    date beyond it has zero forward observations anywhere in the archive,
    so it would come out fully NaN. None if no dated bands are found."""
    with rasterio.open(tif_path) as src:
        descriptions = src.descriptions
    dates = []
    for desc in descriptions:
        if not desc:
            continue
        m = DATE_RE.match(desc)
        if m:
            dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
    return max(dates) if dates else None


def _read_rbf_output_full(path):
    """Full existing RBF output -- (dates, per-band arrays, transform, crs)
    -- for merging newly computed bands into it (see
    _merge_and_write_rbf_output())."""
    with rasterio.open(path) as src:
        arr = src.read(masked=True).astype("float32").filled(np.nan)
        transform = src.transform
        crs = src.crs
        descriptions = src.descriptions
    dates = [datetime.strptime(desc, "%Y-%m-%d").date() for desc in descriptions]
    return dates, arr, transform, crs


def _merge_and_write_rbf_output(out_path, new_dates, new_stack, transform, crs):
    """Write out_path's multiband RBF output, combining whatever bands are
    already on disk with newly computed ones: existing bands are carried
    over untouched, new dates are added, and the result is written sorted
    by date. GeoTIFF has no true incremental band-append, so the merged
    result is always (re)written as a whole file -- but only the newly
    computed bands were actually (re)computed, not the existing ones."""
    combined = {}
    if os.path.exists(out_path):
        existing_dates, existing_arr, transform, crs = _read_rbf_output_full(out_path)
        combined.update(zip(existing_dates, existing_arr))
    combined.update(zip(new_dates, new_stack))

    all_dates = sorted(combined)
    stack = np.stack([combined[d] for d in all_dates])
    write_multiband_tif(out_path, stack, transform, crs, all_dates)


def _process_one_tile(tile_path, year, output_root, vi_key_map, widths,
                       output_dates, wait_days, max_backward_gap_days, max_radius_days,
                       vi_pattern_template, skip_existing=True, despike=True,
                       despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                       despike_max_iter=DEFAULT_DESPIKE_MAX_ITER, cross_vi_despike=False,
                       chunk_size=None, isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Run the delayed ensemble interpolation for every VI in one tile,
    incrementally.

    Module-level (not nested) so it's picklable for ProcessPoolExecutor.

    Two things gate what actually gets (re)computed, on top of
    `output_dates` (the nominal season calendar, already filtered by any
    caller-supplied as_of_date -- see run_year()):

    1. Data ceiling: a target date can only ever be computed once the raw
       archive actually reaches far enough forward -- see
       _last_raw_band_date()/eligible_output_dates(). This tile's raw
       archive's own most recent band (not a wall-clock as_of_date) is
       used as the "as of" point, so requesting a season that runs past
       what's actually been ingested (e.g. season_end in October when the
       raw TSA product only has data through July) naturally stops at the
       last supportable date instead of writing fully-NaN trailing bands.
       Every VI of one tile/year shares the same raw acquisition-date axis
       (see delayed_interpolate_series_multi_vi()'s cross-VI date-axis
       check), so this only needs checking once per tile.
    2. Already written: of the currently-eligible dates, only the ones NOT
       already present as a band in that VI's output file (see
       _existing_rbf_output_dates()) are (re)computed -- freshly computed
       bands are merged into the existing file (see
       _merge_and_write_rbf_output()) rather than recomputing/overwriting
       every date from scratch. skip_existing=False forces every eligible
       date to be recomputed (existing bands for those dates are
       overwritten with fresh values; nothing else is touched).

    cross_vi_despike (see delayed_interpolate_series_multi_vi(), default
    False) changes what "needs computing" means here: with
    cross_vi_despike=True, a VI with nothing new to add still has its raw
    series loaded (though not rewritten) alongside whichever VIs DO need
    new dates, since a date flagged as an outlier by any one VI's own
    residual test is excluded from every VI at that pixel. With the
    default False, a VI with nothing new to add is skipped entirely -- not
    loaded at all.

    :return: (tile_id, written_paths, skipped_paths, pending_paths) --
        `pending_paths` are outputs where the raw archive doesn't yet
        reach far enough forward to compute any new eligible date at all
        (e.g. this tile's TSA product hasn't been updated since the last
        run).
    """
    tile_id = os.path.basename(tile_path)
    out_dir = os.path.join(output_root, tile_id)
    os.makedirs(out_dir, exist_ok=True)

    tif_paths = {}
    out_paths = {}
    for canonical, file_token in vi_key_map.items():
        tif_path = find_vi_file(tile_path, file_token, year, vi_pattern_template)
        base, ext = os.path.splitext(os.path.basename(tif_path))
        tif_paths[canonical] = tif_path
        out_paths[canonical] = os.path.join(out_dir, f"{base}_rbf{ext}")

    last_real_date = _last_raw_band_date(next(iter(tif_paths.values())))
    eligible_dates = (
        eligible_output_dates(output_dates, last_real_date, wait_days)
        if last_real_date is not None else []
    )

    if not eligible_dates:
        return tile_id, [], [], list(out_paths.values())

    to_write = {}
    already_done = set()
    for canonical, out_path in out_paths.items():
        existing_dates = (_existing_rbf_output_dates(out_path) or set()) if skip_existing else set()
        missing = [d for d in eligible_dates if d not in existing_dates]
        if missing:
            to_write[canonical] = missing
        else:
            already_done.add(canonical)
    skipped = [out_paths[canonical] for canonical in already_done]

    if not to_write:
        return tile_id, [], skipped, []

    # Union of missing dates across every VI that needs something -- keeps
    # cross_vi_despike batched across VIs; each VI's own subset (they may
    # differ if outputs were left in inconsistent states by prior partial
    # runs) is pulled back out below before merging/writing.
    union_dates = sorted(set().union(*to_write.values()))

    # cross_vi_despike=True needs every VI's raw series (see docstring);
    # otherwise a VI with nothing new can be skipped without loading it at all.
    tif_paths_needed = tif_paths if cross_vi_despike else {k: tif_paths[k] for k in to_write}
    results = delayed_interpolate_series_for_tile(
        tif_paths_needed, union_dates, widths=widths, wait_days=wait_days,
        max_backward_gap_days=max_backward_gap_days, max_radius_days=max_radius_days,
        despike=despike, despike_threshold_factor=despike_threshold_factor,
        despike_max_iter=despike_max_iter, cross_vi_despike=cross_vi_despike,
        chunk_size=chunk_size, isolation_radius_days=isolation_radius_days,
    )

    written = []
    for canonical, missing_dates in to_write.items():
        stack, transform, crs = results[canonical]
        idx = [union_dates.index(d) for d in missing_dates]
        _merge_and_write_rbf_output(out_paths[canonical], missing_dates, stack[idx], transform, crs)
        written.append(out_paths[canonical])

    return tile_id, written, skipped, []


def run_year(root_dir, year, output_root, vi_keys=None,
             widths=DEFAULT_WIDTHS_DAYS,
             wait_days=DEFAULT_WAIT_DAYS,
             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
             max_radius_days=DEFAULT_MAX_RADIUS_DAYS,
             season_start="05-01", season_end="10-31", step_days=5,
             as_of_date=None,
             vi_pattern_template="{year}*_{vi}_TSS.tif",
             max_workers=1,
             skip_existing=True,
             despike=True,
             despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
             despike_max_iter=DEFAULT_DESPIKE_MAX_ITER,
             cross_vi_despike=False, chunk_size=None, grid_csv=DEFAULT_FORCE_GRID_CSV,
             isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Run delayed ensemble-kernel interpolation for every tile under root_dir,
    for every requested VI, writing one multiband GeoTIFF per (tile, VI) to
    output_root/<tile_id>/<VI>_delayed_<year>.tif.

    :param vi_keys: list or {canonical_name: file_token} dict, defaults to VI_KEYS.
    :param as_of_date: if given, only process dates with (date + wait_days) <=
        as_of_date -- makes recurring/cron runs only compute newly-eligible dates.
        Defaults to processing the whole season (archival/offline mode).
    :param max_workers: tiles are independent; >1 processes them concurrently.
    :param skip_existing: if True, skip a (tile, VI) once its output file
        already has a band for every date that's currently eligible --
        both "in output_dates" AND "within reach of that tile's own raw
        archive right now" (see _process_one_tile()). Any eligible dates
        missing from the file are computed and merged in (existing bands
        are left untouched, not recomputed) instead of skipping or
        rewriting the whole file. Requesting a season that runs past what
        the raw archive currently covers (e.g. season_end in October when
        the TSA product only has data through July) naturally stops
        output at the last date the data actually supports, rather than
        writing NaN-filled bands out to season_end.
    :param despike, despike_threshold_factor, despike_max_iter: see
        despike_daily_cube() -- despike=True (the default) removes
        triplet-residual outliers from each tile/VI's raw series before
        smoothing. despike=False skips despiking entirely (no
        despike_daily_cube() call at all, and none of its memory cost).
    :param cross_vi_despike: see delayed_interpolate_series_multi_vi().
        Default False despikes each VI independently, one at a time --
        much lower peak memory per tile than cross_vi_despike=True, which
        needs every VI's full raw daily cube resident at once.
    :param chunk_size: see _delayed_interpolate_series_generic(). None
        (default) processes each tile at full 3000x3000-ish extent, as
        before. An integer pixel size (e.g. 512) instead processes each
        tile in chunk_size x chunk_size spatial windows, one at a time --
        despike_daily_cube() and the ensemble-smoothing precompute step
        both build several full (n_dates, height, width) arrays, so this
        is what actually caps peak memory per tile, independent of the
        despike/cross_vi_despike settings above.
    :param grid_csv: path to a CSV with a Tile_ID column (defaults to
        data/CH_FORCE_Grids.csv) restricting processing to only the grid
        tiles listed there -- see force_pull.discover_tiles(). Pass None
        to process every tile folder found under `root_dir`, unfiltered.
    :param isolation_radius_days: see delayed_smooth_one_date()'s
        isolated-point fallback.
    """
    vi_key_map = _resolve_vi_keys(vi_keys)

    output_dates = growing_season_dates(year, season_start, season_end, step_days)
    if as_of_date is not None:
        output_dates = eligible_output_dates(output_dates, as_of_date, wait_days)

    if not output_dates:
        print("No output dates are eligible yet -- nothing to do.")
        return

    print(f"Output dates ({len(output_dates)}): "
          f"{output_dates[0]} .. {output_dates[-1]}, every {step_days}d, "
          f"wait_days={wait_days}, max_radius_days={max_radius_days}")

    tiles = discover_tiles(root_dir, grid_csv=grid_csv)
    print(f"Found {len(tiles)} tile folders under {root_dir}")

    if skip_existing:
        print("skip_existing=True: (tile, VI) outputs already covering every currently-eligible date will be skipped.")

    total_written = 0
    total_skipped = 0
    total_pending = 0

    if max_workers <= 1:
        for tile_path in tiles:
            tile_id, written, skipped, pending = _process_one_tile(
                tile_path, year, output_root, vi_key_map, widths,
                output_dates, wait_days, max_backward_gap_days, max_radius_days,
                vi_pattern_template, skip_existing, despike,
                despike_threshold_factor, despike_max_iter, cross_vi_despike,
                chunk_size, isolation_radius_days,
            )
            for p in written:
                print(f"  {tile_id} -> {p}")
            for p in skipped:
                print(f"  {tile_id} -> skipped (already up to date): {p}")
            for p in pending:
                print(f"  {tile_id} -> pending (raw archive not new enough yet): {p}")
            total_written += len(written)
            total_skipped += len(skipped)
            total_pending += len(pending)
    else:
        print(f"Processing tiles with max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_one_tile, tile_path, year, output_root, vi_key_map,
                    widths, output_dates, wait_days, max_backward_gap_days,
                    max_radius_days, vi_pattern_template, skip_existing, despike,
                    despike_threshold_factor, despike_max_iter, cross_vi_despike,
                    chunk_size, isolation_radius_days,
                ): tile_path
                for tile_path in tiles
            }
            for fut in as_completed(futures):
                tile_path = futures[fut]
                fallback_id = os.path.basename(tile_path)
                try:
                    tile_id, written, skipped, pending = fut.result()
                except Exception as e:
                    print(f"  Tile {fallback_id} failed: {type(e).__name__}: {e}")
                    continue
                for p in written:
                    print(f"  {tile_id} -> {p}")
                for p in skipped:
                    print(f"  {tile_id} -> skipped (already up to date): {p}")
                for p in pending:
                    print(f"  {tile_id} -> pending (raw archive not new enough yet): {p}")
                total_written += len(written)
                total_skipped += len(skipped)
                total_pending += len(pending)

    print(f"Done. Wrote {total_written} file(s), skipped {total_skipped} up-to-date file(s), "
          f"{total_pending} pending more raw data.")


# ---------------------------------------------------------------------
# Point comparison plots
# ---------------------------------------------------------------------

def find_sample_pixel(daily_cube, min_obs=10):
    """Locate a pixel with reasonable observation density, for quick plotting."""
    obs_count = np.sum(~np.isnan(daily_cube), axis=0)
    candidates = np.argwhere(obs_count >= min_obs)
    if len(candidates) == 0:
        raise ValueError(f"No pixel found with at least {min_obs} valid observations")
    row, col = candidates[len(candidates) // 2]
    return int(row), int(col)


def _plot_pixel_comparison(dates, daily_cube, local_row, local_col, row, col, year, source_label,
                            widths, wait_days, max_backward_gap_days, max_radius_days,
                            season_start, season_end, step_days, show_components, title, output_path,
                            despike=True, despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                            despike_max_iter=DEFAULT_DESPIKE_MAX_ITER, debug_pixel=False,
                            other_vi_cubes=None, isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Shared plotting core of plot_pixel_comparison() /
    plot_pixel_comparison_from_archive(): given an already-loaded pixel
    series, optionally despikes it (see despike_daily_cube()) and draws
    the raw observations -- with any despiked-out points marked distinctly
    -- each width's independent component (dashed), and the final ensemble
    (bold, what run_year() / stac_pull.update_vi_min_max_interpolated()
    write to disk, computed from the despiked series to match production).
    Ensemble points produced only via delayed_smooth_one_date()'s
    isolated-point fallback (real data on one side only, with nothing
    reasonably nearby on the other -- see isolation_radius_days) are
    outlined distinctly, so it's visible which points a normal both-sided
    gap wouldn't have produced.
    debug_pixel=True additionally prints debug_despike_pixel()'s pass-by-
    pass noise/threshold trace for this exact pixel, to see why a
    particular point is or isn't getting flagged.

    other_vi_cubes: optional {vi_key: cube} for the OTHER VIs at this
    exact same pixel/date grid, same shape as `daily_cube`. When given
    (and despike=True), despiking is cross-checked across all of them
    (see despike_daily_cubes_union()) instead of despiking `daily_cube`
    in isolation -- matching production, where a date flagged by any one
    VI's own residual test is excluded from every VI at that pixel."""
    raw_series = daily_cube[:, local_row, local_col]

    if debug_pixel:
        print(f"Despike trace for pixel (row={row}, col={col}):")
        debug_despike_pixel(dates, raw_series, despike_threshold_factor, despike_max_iter)

    if despike:
        if other_vi_cubes:
            cubes_by_vi = {"__displayed__": daily_cube, **other_vi_cubes}
            cleaned_by_vi, removed_mask = despike_daily_cubes_union(
                dates, cubes_by_vi, despike_threshold_factor, despike_max_iter,
            )
            smoothing_cube = cleaned_by_vi["__displayed__"]
        else:
            smoothing_cube, removed_mask = despike_daily_cube(dates, daily_cube, despike_threshold_factor, despike_max_iter)
        removed_series = removed_mask[:, local_row, local_col]
    else:
        smoothing_cube = daily_cube
        removed_series = np.zeros(len(dates), dtype=bool)

    valid_mask, ordinals, last_valid_cum, next_valid_cum = _precompute_daily_stats(dates, smoothing_cube)
    output_dates = growing_season_dates(year, season_start, season_end, step_days)

    fig, ax = plt.subplots(figsize=(11, 5))
    date_arr = np.array(dates)
    kept = ~removed_series
    ax.scatter(date_arr[kept], raw_series[kept], color="black", s=28, zorder=5, label="Raw observations")
    if removed_series.any():
        ax.scatter(date_arr[removed_series], raw_series[removed_series], color="darkorange", marker="x",
                   s=70, linewidths=2, zorder=6, label="Removed as outlier")

    if show_components:
        for width_days in widths:
            component = np.array([
                delayed_smooth_one_date(smoothing_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                         d, (width_days,), wait_days, max_backward_gap_days,
                                         max_radius_days,
                                         isolation_radius_days=isolation_radius_days)[local_row, local_col]
                for d in output_dates
            ])
            ax.plot(output_dates, component, linestyle="--", linewidth=1, alpha=0.5, label=f"{width_days}d component")

    ensemble = np.empty(len(output_dates), dtype="float32")
    isolated_fallback = np.zeros(len(output_dates), dtype=bool)
    for i, d in enumerate(output_dates):
        value, fallback_mask = delayed_smooth_one_date(
            smoothing_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
            d, widths, wait_days, max_backward_gap_days, max_radius_days,
            isolation_radius_days=isolation_radius_days, return_fallback_mask=True,
        )
        ensemble[i] = value[local_row, local_col]
        isolated_fallback[i] = fallback_mask[local_row, local_col]

    ax.plot(output_dates, ensemble, color="tab:red", linewidth=2.5,
            label="Ensemble (production output)")
    output_date_arr = np.array(output_dates)
    normal_pts = ~isolated_fallback & ~np.isnan(ensemble)
    ax.scatter(output_date_arr[normal_pts], ensemble[normal_pts], color="tab:red", s=20, zorder=7)
    if isolated_fallback.any():
        ax.scatter(output_date_arr[isolated_fallback], ensemble[isolated_fallback],
                   facecolors="none", edgecolors="tab:red", marker="o", s=110, linewidths=2,
                   zorder=8, label="Isolated-point fallback")

    ax.set_title(title or f"Pixel (row={row}, col={col}) -- {source_label}")
    ax.set_ylabel("VI value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    return fig, (row, col)


def plot_pixel_comparison(tif_path, year, row=None, col=None,
                           widths=DEFAULT_WIDTHS_DAYS,
                           wait_days=DEFAULT_WAIT_DAYS,
                           max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                           max_radius_days=DEFAULT_MAX_RADIUS_DAYS,
                           season_start="01-01", season_end="10-31", step_days=5,
                           show_components=True, title=None, output_path=None,
                           despike=True, despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                           despike_max_iter=DEFAULT_DESPIKE_MAX_ITER, debug_pixel=False,
                           other_vi_tif_paths=None,
                           isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Diagnostic plot for one pixel: raw observations (any despiked-out
    outliers marked with an "x" -- see despike_daily_cube()), each width's
    independent component (dashed), and the final ensemble (bold, what
    run_year() writes to disk; points produced only via the isolated-point
    fallback -- see isolation_radius_days -- are outlined). If row/col
    given, only a small window is read from disk; otherwise a full-tile
    read is needed to find a dense pixel. Reads from a premade FORCE-style
    multiband TIF -- for a STAC/swisseo per-date archive directory, see
    plot_pixel_comparison_from_archive().
    debug_pixel=True prints debug_despike_pixel()'s pass-by-pass trace for
    this exact pixel.

    :param other_vi_tif_paths: optional {vi_key: tif_path} for the OTHER
        VIs of this same tile/year -- when given, despiking is cross-
        checked across all of them (see despike_daily_cubes_union()): a
        date flagged as an outlier in any of these VIs is treated as
        removed for the displayed VI too, matching
        force_pull.run_tsa_workflow()'s production behavior (via
        rbf_interp.run_year()). Leave None to despike only the displayed
        VI in isolation.
    :param isolation_radius_days: see delayed_smooth_one_date()'s
        isolated-point fallback -- match whichever pipeline produced
        tif_path.
    """
    if row is not None and col is not None:
        dates, daily_cube, local_row, local_col = load_vi_pixel_series(tif_path, row, col, buffer=0)
    else:
        dates, daily_cube, _, _ = load_vi_cube(tif_path)
        row, col = find_sample_pixel(daily_cube)
        local_row, local_col = row, col

    other_vi_cubes = None
    if other_vi_tif_paths:
        # Cross-VI despiking needs every VI's cube at the exact same
        # single-pixel window -- re-fetch the displayed VI that way too
        # (cheap: a 1x1 window vs. a full-tile read) so shapes line up
        # regardless of whether row/col above came from auto-find or not.
        dates, daily_cube, local_row, local_col = load_vi_pixel_series(tif_path, row, col, buffer=0)
        other_vi_cubes = {}
        for vi_key, other_path in other_vi_tif_paths.items():
            other_dates, other_cube, _, _ = load_vi_pixel_series(other_path, row, col, buffer=0)
            if other_dates != dates:
                raise ValueError(
                    f"plot_pixel_comparison: VI '{vi_key}' has a different date axis than "
                    f"'{tif_path}' -- cross-VI despiking requires matching dates "
                    f"(see delayed_interpolate_series_multi_vi())."
                )
            other_vi_cubes[vi_key] = other_cube

    return _plot_pixel_comparison(
        dates, daily_cube, local_row, local_col, row, col, year, os.path.basename(tif_path),
        widths, wait_days, max_backward_gap_days, max_radius_days,
        season_start, season_end, step_days, show_components, title, output_path,
        despike, despike_threshold_factor, despike_max_iter, debug_pixel, other_vi_cubes,
        isolation_radius_days,
    )


def plot_pixel_comparison_from_archive(archive_dir, year, row=None, col=None,
                                        widths=DEFAULT_WIDTHS_DAYS,
                                        wait_days=DEFAULT_WAIT_DAYS,
                                        max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                        max_radius_days=DEFAULT_MAX_RADIUS_DAYS,
                                        season_start="03-01", season_end="10-31", step_days=5,
                                        show_components=True, title=None, output_path=None,
                                        despike=True, despike_threshold_factor=DEFAULT_DESPIKE_THRESHOLD_FACTOR,
                                        despike_max_iter=DEFAULT_DESPIKE_MAX_ITER, debug_pixel=False,
                                        other_vi_archive_dirs=None,
                                        isolation_radius_days=DEFAULT_ISOLATION_RADIUS_DAYS):
    """Archive-directory equivalent of plot_pixel_comparison(), for a
    per-VI STAC/swisseo archive (archive_root/<VI>/<date>.tif, see
    stac_pull.archive_vi_raster()) instead of a premade FORCE TSA
    multiband TIF. `archive_dir` is the per-VI folder itself, e.g.
    os.path.join(archive_root, "NDVI"). See despike_daily_cube() for the
    despike_* params -- this is the tool for visually checking outliers
    (e.g. residual cloud/shadow contamination) in a swisseo archive and
    tuning the threshold before it's applied operationally. debug_pixel=True
    prints debug_despike_pixel()'s pass-by-pass noise/threshold trace for
    this exact pixel -- use it to see why a specific point is or isn't
    getting flagged.

    :param other_vi_archive_dirs: optional {vi_key: archive_dir} for the
        OTHER VIs under this same archive_root -- when given, despiking is
        cross-checked across all of them (see despike_daily_cubes_union()):
        a date flagged as an outlier in any of these VIs is treated as
        removed for the displayed VI too, matching
        stac_pull.update_vi_min_max_interpolated()'s production behavior.
        Leave None to despike only the displayed VI in isolation (e.g.
        what you'd have seen on CCI alone before this option existed --
        which is why an obvious outlier caught by other VIs might not
        show up as removed here).
    :param isolation_radius_days: see delayed_smooth_one_date()'s
        isolated-point fallback -- match whichever pipeline produced
        archive_dir.
    """
    if row is not None and col is not None:
        dates, daily_cube, local_row, local_col = load_vi_pixel_series_from_archive(archive_dir, row, col, buffer=0)
    else:
        dates, daily_cube, _, _ = load_vi_cube_from_archive(archive_dir)
        row, col = find_sample_pixel(daily_cube)
        local_row, local_col = row, col

    other_vi_cubes = None
    if other_vi_archive_dirs:
        # Cross-VI despiking needs every VI's cube at the exact same
        # single-pixel window -- re-fetch the displayed VI that way too
        # (cheap: a 1x1 window vs. a full-tile read) so shapes line up
        # regardless of whether row/col above came from auto-find or not.
        dates, daily_cube, local_row, local_col = load_vi_pixel_series_from_archive(archive_dir, row, col, buffer=0)
        other_vi_cubes = {}
        for vi_key, other_dir in other_vi_archive_dirs.items():
            other_dates, other_cube, _, _ = load_vi_pixel_series_from_archive(other_dir, row, col, buffer=0)
            if other_dates != dates:
                raise ValueError(
                    f"plot_pixel_comparison_from_archive: VI '{vi_key}' has a different date "
                    f"axis than '{archive_dir}' -- cross-VI despiking requires matching dates "
                    f"(see delayed_interpolate_series_multi_vi())."
                )
            other_vi_cubes[vi_key] = other_cube

    source_label = os.path.basename(os.path.normpath(archive_dir))
    return _plot_pixel_comparison(
        dates, daily_cube, local_row, local_col, row, col, year, source_label,
        widths, wait_days, max_backward_gap_days, max_radius_days,
        season_start, season_end, step_days, show_components, title, output_path,
        despike, despike_threshold_factor, despike_max_iter, debug_pixel, other_vi_cubes,
        isolation_radius_days,
    )


if __name__ == "__main__":
    # Set the VI keys in the dataset
    vi_keys_2026 = {
        "NDVI": "NDVI",
        "EVI": "EVI",
        "NDMI": "NDMI",
        "CIRE": "CIre",
        "CCI": "CCI",
    }
    vi_keys_2018 = {
        "NDVI": "NDV",
        "EVI": "EVI",
        "NDMI": "NDM",
        "CIRE": "CRE",
        "CCI": "CCI",
    }

    # Example usage -- adjust paths/year before running.
    ROOT_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss"
    # r'B:\bloomc\DiscoCH_2026_08_03\FORCE\level_3_2018_tiled'
    # r"\\speedy16-36\Data_23\FORCE\FORCE_Kingslide\level2\tsa\real_values_flagged"
    # r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss_v2"
    OUTPUT_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level4_rbf\2026"
    YEAR = 2026

    run_year(ROOT_DIR, YEAR, OUTPUT_ROOT, season_start="03-01", season_end="08-16",
             wait_days=DEFAULT_WAIT_DAYS, max_radius_days=DEFAULT_MAX_RADIUS_DAYS, vi_keys=vi_keys_2026,
             max_workers=3,
             despike=False,     # toggle: False skips despike_daily_cube() entirely
             chunk_size=None)    # toggle: spatial block size (pixels); None = whole tile at once

    example_tif = find_vi_file(
        os.path.join(ROOT_DIR, discover_tiles(ROOT_DIR)[24]), "NDVI", YEAR
    )
    plot_pixel_comparison(
        example_tif, YEAR, row=700, col=210,
        widths=(10,),  # only the 10d kernel
        output_path="pixel_comparison_10d_only.png",
    )
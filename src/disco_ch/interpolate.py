# """
# causal_interp.py
#
# Builds a single 5-day, growing-season-only interpolated VI time series
# from the same FORCE TSA multiband TIFs used in force_pull.py, using a
# *causal* (one-sided, no-lookahead) composite Gaussian kernel instead of
# the symmetric RBF convolution the discoloration model was originally
# trained with.
#
# WHY THIS EXISTS
# ----------------
# The original model expects 5-day series produced by a symmetric RBF
# smoother that combined four kernel widths (10/20/30/50 days, weighted
# more heavily toward the narrower ones) into a single output value. That
# combination needs observations from *both* before and after each
# smoothed date, which is fine offline but doesn't exist for anything
# close to "today" in an operational setting.
#
# Rather than fabricate a plausible future (which risks smoothing away the
# exact early-season discoloration decline we're trying to detect), this
# module only ever reweights real observations that already happened:
#
#   - For a given output date, only observations from at least `lag_days`
#     before it are used at all (the causal / no-lookahead constraint).
#   - Each usable observation is weighted by a *composite* kernel: the sum
#     of four Gaussians (std = 10/20/30/50 days, same as training), scaled
#     so the narrow kernels dominate and the wide ones only really matter
#     when nothing recent is available -- see default_scale_weights().
#     Weights are truncated to the past and renormalized over whatever
#     real data actually exists for that pixel (a standard boundary
#     correction for kernel smoothing near the edge of a series, see e.g.
#     Gasser & Muller 1979).
#   - If the most recent usable observation for a pixel is older than
#     `max_gap_days`, that pixel is set to NaN rather than smoothed from
#     stale data.
#
# This produces ONE output series per VI (matching what the original RBF
# scheme produced), not one per kernel width.
#
# Because eligibility for a given output date is a pure function of that
# date and `lag_days` -- never of what other output dates exist, and never
# of anything after the cutoff -- this behaves identically whether it's
# run once over a full year of archived data or literally scheduled every
# few days operationally.
#
# GROWING SEASON GATING
# ----------------------
# Output dates only start in May (configurable via `season_start`).
# January-April observations are never turned into their own output rows,
# but they DO remain available as history for early-May outputs whose
# kernel window reaches back that far -- there's no special-case needed
# for this, it falls straight out of the causal weighting above.
#
# PERFORMANCE NOTES
# ------------------
#   - The per-output-date history loop is bounded by `max_radius_days`
#     (default ~4 standard deviations of the widest kernel component),
#     rather than growing across the whole season -- contributions beyond
#     that are numerically negligible.
#   - Per-(tile, VI) bookkeeping (which bands are valid, and the most
#     recent valid date up through each point in the series) is computed
#     ONCE via _precompute_daily_stats() and reused across every output
#     date, rather than being recomputed inside the loop.
#   - plot_pixel_comparison() does a small windowed raster read instead of
#     loading the whole tile, when a specific pixel is requested.
#   - run_year() can process tiles in parallel via `max_workers`, the same
#     ProcessPoolExecutor pattern used in force_pull.py.
#
# Place this file alongside force_pull.py (e.g. src/disco_ch/causal_interp.py)
# so the relative import below resolves.
# """
#
# import os
# import warnings
# from datetime import datetime, timedelta
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# import numpy as np
# import rasterio
# from rasterio.windows import Window
# import matplotlib.pyplot as plt
#
# from src.disco_ch.force_pull import (
#     DATE_RE,
#     find_vi_file,
#     discover_tiles,
#     VI_KEYS,
#     _resolve_vi_keys,
# )
#
# # Same kernel widths (in days) the original RBF-smoothed training data used.
# DEFAULT_WIDTHS_DAYS = (10, 20, 30, 50)
#
# # "A few days" -- how long we wait after a date before we're willing to
# # call it final and compute its interpolated value.
# DEFAULT_LAG_DAYS = 5
#
# # "About 2 weeks" -- beyond this gap since the last real observation, a
# # pixel is reported as NaN rather than smoothed from stale data.
# DEFAULT_MAX_GAP_DAYS = 20
#
#
# # ---------------------------------------------------------------------
# # Raw data loading
# # ---------------------------------------------------------------------
#
# def _collapse_bands_to_daily(arr, descriptions):
#     """
#     Shared by load_vi_cube() and load_vi_pixel_series(): parses each
#     band's date from its description and averages same-day duplicates
#     (e.g. two satellite overpasses on one date), same convention as
#     force_pull.get_vi_band().
#     """
#     band_dates = []
#     keep_idx = []
#     for i, desc in enumerate(descriptions):
#         if not desc:
#             continue
#         m = DATE_RE.match(desc)
#         if not m:
#             continue
#         band_dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
#         keep_idx.append(i)
#
#     if not band_dates:
#         raise ValueError("No dated bands found in raster")
#
#     cube = arr[keep_idx]
#     dates = sorted(set(band_dates))
#     daily_cube = np.full((len(dates), cube.shape[1], cube.shape[2]), np.nan, dtype="float32")
#     for out_i, d in enumerate(dates):
#         same_day_idx = [i for i, dd in enumerate(band_dates) if dd == d]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices are expected
#             daily_cube[out_i] = np.nanmean(cube[same_day_idx], axis=0)
#
#     return dates, daily_cube
#
#
# def load_vi_cube(tif_path):
#     """
#     Reads an entire VI multiband TIF into memory and collapses it to one
#     layer per calendar date.
#
#     Returns:
#         dates: sorted list of date objects, one per distinct acquisition date
#         daily_cube: float32 array (n_dates, height, width), nodata -> NaN
#         transform, crs: from the source raster, for writing outputs later
#     """
#     with rasterio.open(tif_path) as src:
#         arr = src.read(masked=True).astype("float32").filled(np.nan)
#         transform = src.transform
#         crs = src.crs
#         descriptions = src.descriptions
#
#     dates, daily_cube = _collapse_bands_to_daily(arr, descriptions)
#     return dates, daily_cube, transform, crs
#
#
# def load_vi_pixel_series(tif_path, row, col, buffer=0):
#     """
#     Like load_vi_cube(), but only reads a small window around (row, col)
#     from disk -- avoids paying the full-tile I/O cost just to look at one
#     pixel's time series. buffer=0 reads a single pixel column.
#
#     Returns dates, daily_cube (windowed), and the pixel's row/col *within
#     that window* (0, 0 for buffer=0, since the window edge may have been
#     clamped to the raster bounds near an edge/corner pixel).
#     """
#     with rasterio.open(tif_path) as src:
#         row_off = max(row - buffer, 0)
#         col_off = max(col - buffer, 0)
#         row_end = min(row + buffer + 1, src.height)
#         col_end = min(col + buffer + 1, src.width)
#         window = Window(col_off=col_off, row_off=row_off,
#                          width=col_end - col_off, height=row_end - row_off)
#         arr = src.read(window=window, masked=True).astype("float32").filled(np.nan)
#         descriptions = src.descriptions
#
#     dates, daily_cube = _collapse_bands_to_daily(arr, descriptions)
#     return dates, daily_cube, row - row_off, col - col_off
#
#
# # ---------------------------------------------------------------------
# # Causal composite kernel
# # ---------------------------------------------------------------------
#
# def causal_gaussian_weight(delta_days, width_days):
#     """
#     One-sided Gaussian weight for an observation `delta_days` before the
#     output date. Same functional form as one of the four kernels used in
#     training (std = width_days); "one-sided" comes entirely from never
#     calling this with a negative (future) delta.
#     """
#     return np.exp(-0.5 * (delta_days / width_days) ** 2)
#
#
# def default_scale_weights(widths=DEFAULT_WIDTHS_DAYS):
#     """
#     How much each width contributes to the composite kernel, normalized
#     to sum to 1. Defaults to inverse-width weighting, so the narrow (10d)
#     kernel dominates and the wide (50d) one mostly just fills in when
#     nothing recent is available -- matching the "weighted much higher for
#     the 10d vs further out" recollection of the original scheme. Swap
#     this for the exact original weights if/when you confirm them.
#     """
#     raw = np.array([1.0 / w for w in widths], dtype="float64")
#     return raw / raw.sum()
#
#
# def default_max_radius_days(widths=DEFAULT_WIDTHS_DAYS, n_sigma=4):
#     """
#     Contributions beyond n_sigma standard deviations of the widest kernel
#     component are numerically negligible (4 sigma of a 50d kernel is
#     exp(-8) ~ 3e-4). Capping history lookups at this radius bounds the
#     per-output-date loop to a fixed size instead of it growing across the
#     whole season.
#     """
#     return int(n_sigma * max(widths))
#
#
# def make_composite_kernel(widths=DEFAULT_WIDTHS_DAYS, scale_weights=None):
#     """
#     Builds a single kernel_weight_fn(delta_days) -> weight, combining the
#     four Gaussians into one composite curve. This is the causal analogue
#     of the original multi-width RBF combination -- same shape, same
#     relative scale weighting, just never evaluated for delta_days < 0.
#     """
#     if scale_weights is None:
#         scale_weights = default_scale_weights(widths)
#     widths_arr = np.asarray(widths, dtype="float64")
#     scale_weights = np.asarray(scale_weights, dtype="float64")
#     scale_weights = scale_weights / scale_weights.sum()
#
#     def kernel(delta_days):
#         return float(np.sum(scale_weights * np.exp(-0.5 * (delta_days / widths_arr) ** 2)))
#
#     return kernel
#
#
# # ---------------------------------------------------------------------
# # Causal smoothing
# # ---------------------------------------------------------------------
#
# def _precompute_daily_stats(dates, daily_cube):
#     """
#     Computed ONCE per (tile, VI) and reused across every output date:
#       - valid_mask: boolean cube, True where a real observation exists
#       - ordinals: date ordinals, shape (n_dates,), ascending
#       - last_valid_cum: per-pixel running "most recent valid date-ordinal
#         seen up through date index i" (forward-filled). This turns the
#         staleness check for any output date into an O(1) array lookup
#         instead of re-scanning history every time.
#     """
#     valid_mask = ~np.isnan(daily_cube)
#     ordinals = np.array([d.toordinal() for d in dates], dtype="int64")
#     ordinal_if_valid = np.where(valid_mask, ordinals[:, None, None], -1)
#     last_valid_cum = np.maximum.accumulate(ordinal_if_valid, axis=0)
#     return valid_mask, ordinals, last_valid_cum
#
#
# def causal_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum,
#                             target_date, kernel_weight_fn,
#                             lag_days=DEFAULT_LAG_DAYS, max_gap_days=DEFAULT_MAX_GAP_DAYS,
#                             max_radius_days=None):
#     """
#     Computes one causally-interpolated 2D raster representing `target_date`.
#
#     Takes the precomputed outputs of _precompute_daily_stats() rather than
#     recomputing them, so a whole season of output dates can share one pass
#     over the raw data instead of re-scanning it each time.
#
#     Only observations at or before (target_date - lag_days) are used.
#     Nothing is invented: a pixel with one usable observation just gets
#     that observation's value; a pixel with none gets NaN.
#
#     `max_radius_days` bounds how far back the weighted sum looks (see
#     default_max_radius_days()); pass None to consider every eligible date
#     regardless of age (slower, and the extra dates contribute ~nothing).
#     """
#     cutoff_ordinal = (target_date - timedelta(days=lag_days)).toordinal()
#     end_idx = int(np.searchsorted(ordinals, cutoff_ordinal, side="right")) - 1
#
#     h, w = daily_cube.shape[1:]
#     if end_idx < 0:
#         return np.full((h, w), np.nan, dtype="float32")
#
#     start_idx = 0
#     if max_radius_days is not None:
#         start_idx = int(np.searchsorted(ordinals, cutoff_ordinal - max_radius_days, side="left"))
#
#     target_ordinal = target_date.toordinal()
#     weight_sum = np.zeros((h, w), dtype="float32")
#     value_sum = np.zeros((h, w), dtype="float32")
#
#     for i in range(start_idx, end_idx + 1):
#         valid = valid_mask[i]
#         if not valid.any():
#             continue
#         wgt = kernel_weight_fn(target_ordinal - ordinals[i])
#         layer = daily_cube[i]
#         weight_sum[valid] += wgt
#         value_sum[valid] += wgt * layer[valid]
#
#     has_data = weight_sum > 0
#     last_valid_ordinal = last_valid_cum[end_idx]
#     gap_days = np.where(last_valid_ordinal >= 0, target_ordinal - last_valid_ordinal, np.inf)
#     stale = gap_days > max_gap_days
#
#     safe_denom = np.where(has_data, weight_sum, 1.0)
#     result = np.where(has_data & ~stale, value_sum / safe_denom, np.nan)
#     return result.astype("float32")
#
#
# # ---------------------------------------------------------------------
# # Growing-season output calendar
# # ---------------------------------------------------------------------
#
# def growing_season_dates(year, season_start="05-01", season_end="07-24", step_days=5):
#     """Every `step_days` days from season_start to season_end (inclusive), for `year`."""
#     start = datetime.strptime(f"{year}-{season_start}", "%Y-%m-%d").date()
#     end = datetime.strptime(f"{year}-{season_end}", "%Y-%m-%d").date()
#     n_steps = (end - start).days // step_days + 1
#     return [start + timedelta(days=step_days * i) for i in range(n_steps)]
#
#
# # ---------------------------------------------------------------------
# # Full series / tile / year orchestration
# # ---------------------------------------------------------------------
#
# def causal_interpolate_series(tif_path, output_dates, kernel_weight_fn=None,
#                                lag_days=DEFAULT_LAG_DAYS, max_gap_days=DEFAULT_MAX_GAP_DAYS,
#                                max_radius_days=None):
#     """Runs causal_smooth_one_date() over every output date for one VI TIF, producing one series."""
#     if kernel_weight_fn is None:
#         kernel_weight_fn = make_composite_kernel()
#     if max_radius_days is None:
#         max_radius_days = default_max_radius_days()
#
#     dates, daily_cube, transform, crs = load_vi_cube(tif_path)
#     valid_mask, ordinals, last_valid_cum = _precompute_daily_stats(dates, daily_cube)
#
#     stack = np.stack([
#         causal_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum,
#                                 d, kernel_weight_fn, lag_days, max_gap_days, max_radius_days)
#         for d in output_dates
#     ])
#     return stack, transform, crs
#
#
# def write_multiband_tif(path, stack, transform, crs, band_dates, nodata=np.nan):
#     n_bands, h, w = stack.shape
#     with rasterio.open(
#         path, "w", driver="GTiff", height=h, width=w, count=n_bands,
#         dtype="float32", crs=crs, transform=transform, nodata=nodata,
#         compress="deflate",
#     ) as dst:
#         for i in range(n_bands):
#             dst.write(stack[i], i + 1)
#             dst.set_band_description(i + 1, band_dates[i].isoformat())
#
#
# def _process_one_tile(tile_path, year, output_root, vi_key_map, widths, scale_weights,
#                        output_dates, lag_days, max_gap_days, max_radius_days,
#                        vi_pattern_template):
#     """
#     Runs the causal composite interpolation for every VI in one tile.
#     Defined at module level (not nested) so it's picklable for
#     ProcessPoolExecutor -- takes plain data (widths/scale_weights) rather
#     than a kernel closure/function object, and rebuilds the kernel
#     function locally inside the worker process.
#     """
#     kernel_weight_fn = make_composite_kernel(widths, scale_weights)
#     tile_id = os.path.basename(tile_path)
#     out_dir = os.path.join(output_root, tile_id)
#     os.makedirs(out_dir, exist_ok=True)
#
#     written = []
#     for canonical, file_token in vi_key_map.items():
#         tif_path = find_vi_file(tile_path, file_token, year, vi_pattern_template)
#         stack, transform, crs = causal_interpolate_series(
#             tif_path, output_dates, kernel_weight_fn=kernel_weight_fn,
#             lag_days=lag_days, max_gap_days=max_gap_days, max_radius_days=max_radius_days,
#         )
#         out_path = os.path.join(out_dir, f"{canonical}_causal_{year}.tif")
#         write_multiband_tif(out_path, stack, transform, crs, output_dates)
#         written.append(out_path)
#
#     return tile_id, written
#
#
# def run_year(root_dir, year, output_root, vi_keys=None,
#              widths=DEFAULT_WIDTHS_DAYS, scale_weights=None,
#              lag_days=DEFAULT_LAG_DAYS, max_gap_days=DEFAULT_MAX_GAP_DAYS,
#              max_radius_days=None,
#              season_start="05-01", season_end="10-31", step_days=5,
#              vi_pattern_template="{year}*_{vi}_TSS.tif",
#              max_workers=1):
#     """
#     Runs causal composite-kernel interpolation for every tile under
#     `root_dir`, for every requested VI, and writes ONE multiband GeoTIFF
#     per (tile, VI) to output_root/<tile_id>/<VI>_causal_<year>.tif.
#
#     :param vi_keys: plain list or {canonical_name: file_token} dict, same
#         convention as force_pull._resolve_vi_keys(). Defaults to VI_KEYS.
#     :param widths / scale_weights: kernel widths (days) and how much each
#         contributes to the composite -- defaults to the same four values
#         used in training (10/20/30/50), weighted toward the narrow end.
#     :param max_radius_days: bounds the history lookback per output date
#         (see default_max_radius_days()); computed automatically from
#         `widths` when left None.
#     :param max_workers: number of tiles to process concurrently. 1 (the
#         default) runs sequentially; tiles are fully independent, so
#         raising this is a straightforward win when processing many tiles.
#     """
#     vi_key_map = _resolve_vi_keys(vi_keys)
#     if max_radius_days is None:
#         max_radius_days = default_max_radius_days(widths)
#
#     output_dates = growing_season_dates(year, season_start, season_end, step_days)
#     print(f"Growing-season output dates ({len(output_dates)}): "
#           f"{output_dates[0]} .. {output_dates[-1]}, every {step_days}d")
#
#     tiles = discover_tiles(root_dir)
#     print(f"Found {len(tiles)} tile folders under {root_dir}")
#
#     if max_workers <= 1:
#         for tile_path in tiles:
#             tile_id, written = _process_one_tile(
#                 tile_path, year, output_root, vi_key_map, widths, scale_weights,
#                 output_dates, lag_days, max_gap_days, max_radius_days, vi_pattern_template,
#             )
#             for p in written:
#                 print(f"  {tile_id} -> {p}")
#     else:
#         print(f"Processing tiles with max_workers={max_workers}")
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = {
#                 executor.submit(
#                     _process_one_tile, tile_path, year, output_root, vi_key_map,
#                     widths, scale_weights, output_dates, lag_days, max_gap_days,
#                     max_radius_days, vi_pattern_template,
#                 ): tile_path
#                 for tile_path in tiles
#             }
#             for fut in as_completed(futures):
#                 tile_path = futures[fut]
#                 fallback_id = os.path.basename(tile_path)
#                 try:
#                     tile_id, written = fut.result()
#                 except Exception as e:
#                     print(f"  Tile {fallback_id} failed: {type(e).__name__}: {e}")
#                     continue
#                 for p in written:
#                     print(f"  {tile_id} -> {p}")
#
#
# # ---------------------------------------------------------------------
# # Point comparison plots
# # ---------------------------------------------------------------------
#
# def find_sample_pixel(daily_cube, min_obs=10):
#     """Convenience: locate a pixel with reasonable observation density, for quick plotting."""
#     obs_count = np.sum(~np.isnan(daily_cube), axis=0)
#     candidates = np.argwhere(obs_count >= min_obs)
#     if len(candidates) == 0:
#         raise ValueError(f"No pixel found with at least {min_obs} valid observations")
#     row, col = candidates[len(candidates) // 2]
#     return int(row), int(col)
#
#
# def plot_pixel_comparison(tif_path, year, row=None, col=None,
#                            widths=DEFAULT_WIDTHS_DAYS, scale_weights=None,
#                            lag_days=DEFAULT_LAG_DAYS, max_gap_days=DEFAULT_MAX_GAP_DAYS,
#                            max_radius_days=None,
#                            season_start="03-01", season_end="10-31", step_days=5,
#                            show_components=True, title=None, output_path=None):
#     """
#     Diagnostic plot for a single pixel: raw observations (scatter), the
#     final composite causal series (bold line -- this is what run_year()
#     actually writes to disk), and optionally each width's individual
#     causal component (thin dashed lines) so you can see how much each
#     scale is contributing and whether a real decline survives.
#
#     When row/col are given explicitly, only a small window around that
#     pixel is read from disk (fast path for repeated diagnostic plotting
#     on a known pixel). If row/col are omitted, a full-tile read is still
#     needed to search for a pixel with decent observation density.
#     """
#     if max_radius_days is None:
#         max_radius_days = default_max_radius_days(widths)
#
#     if row is not None and col is not None:
#         dates, daily_cube, local_row, local_col = load_vi_pixel_series(tif_path, row, col, buffer=0)
#     else:
#         dates, daily_cube, _, _ = load_vi_cube(tif_path)
#         row, col = find_sample_pixel(daily_cube)
#         local_row, local_col = row, col
#
#     valid_mask, ordinals, last_valid_cum = _precompute_daily_stats(dates, daily_cube)
#     raw_series = daily_cube[:, local_row, local_col]
#     output_dates = growing_season_dates(year, season_start, season_end, step_days)
#
#     fig, ax = plt.subplots(figsize=(11, 5))
#     ax.scatter(dates, raw_series, color="black", s=28, zorder=5, label="Raw observations")
#
#     if show_components:
#         for width in widths:
#             component_fn = lambda delta, w=width: causal_gaussian_weight(delta, w)
#             component = np.array([
#                 causal_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum,
#                                         d, component_fn, lag_days, max_gap_days,
#                                         max_radius_days)[local_row, local_col]
#                 for d in output_dates
#             ])
#             ax.plot(output_dates, component, linestyle="--", linewidth=1, alpha=0.5, label=f"{width}d component")
#
#     composite_fn = make_composite_kernel(widths, scale_weights)
#     composite = np.array([
#         causal_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum,
#                                 d, composite_fn, lag_days, max_gap_days,
#                                 max_radius_days)[local_row, local_col]
#         for d in output_dates
#     ])
#     ax.plot(output_dates, composite, color="tab:red", linewidth=2.5, marker="o", markersize=4,
#             label="Composite (production output)")
#
#     ax.set_title(title or f"Pixel (row={row}, col={col}) -- {os.path.basename(tif_path)}")
#     ax.set_ylabel("VI value")
#     ax.legend()
#     fig.autofmt_xdate()
#     fig.tight_layout()
#
#     if output_path:
#         fig.savefig(output_path, dpi=150)
#         print(f"Saved: {output_path}")
#
#     return fig, (row, col)
#
#
# if __name__ == "__main__":
#     # Example usage -- adjust paths/year before running.
#     ROOT_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss"
#     OUTPUT_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\interpolated_vi"
#     YEAR = 2026
#
#     run_year(ROOT_DIR, YEAR, OUTPUT_ROOT, season_start="03-01", season_end="07-25", max_workers=10)
#
#     # Quick visual sanity check on one tile's NDVI.
#     example_tif = find_vi_file(
#         os.path.join(ROOT_DIR, discover_tiles(ROOT_DIR)[24]), "NDVI", YEAR
#     )
#     plot_pixel_comparison(example_tif, YEAR, row=700, col=210,  output_path="pixel_comparison_example.png")

"""
delayed_interp.py

Builds a 5-day, growing-season-only interpolated VI time series from the
same FORCE TSA multiband TIFs used in force_pull.py, using the *original*
symmetric composite Gaussian kernel the discoloration model was trained
with -- run operationally by delaying each output date's computation
until enough real data exists on both sides of it, rather than by
restricting the kernel to only look backward.

WHY THIS EXISTS (vs. causal_interp.py)
----------------------------------------
causal_interp.py made the kernel itself one-sided (delta_days >= 0 only)
so it could be evaluated the instant a target date passed. That avoids
ever waiting, but it's a different estimator than the one the model was
trained on: a one-sided weighted mean has no far-side observations to
cancel the near-side ones against, so during any local trend (green-up,
or the discoloration decline itself) it's systematically biased toward
the trailing side of the trend. In practice that showed up as VI values
reading low during green-up.

This module keeps the *kernel* exactly as trained -- symmetric, using
both past and future observations, same four widths (10/20/30/50 days),
same relative scale weighting -- and instead makes the *schedule*
operational: an output date is only computed once `wait_days` have
elapsed since that date, so that (given the ~5 day satellite revisit)
there's ordinarily at least one real forward observation to balance the
backward ones, closely reproducing the offline symmetric-RBF behavior.

This is a scheduling constraint, not a kernel constraint, so there is no
causal/symmetric mismatch between training and inference: the same
weighting math runs either way, just over a window that's necessarily
narrower on the forward side than an unbounded offline run because
"the future" only extends up to `wait_days` past the target date at
processing time.

NO-DATA GUARD
--------------
Boundary renormalization (dividing by whatever weight mass is actually
present) is appropriate for genuine edge effects -- e.g. the very start
of the growing season, when there simply isn't 50 days of prior history
yet. But left unchecked, the same renormalization will happily produce a
confident-looking value for a pixel that has one real observation 18
days before the target date and nothing else nearby in either direction.
To avoid that, eligibility for a given output date requires BOTH:
  - at least one real observation within `max_backward_gap_days` before
    the target date, and
  - at least one real observation within `wait_days` after it (i.e.
    somewhere in the forward window that's actually available by the
    time this runs).
A pixel failing either check is set to NaN for that date rather than
interpolated from a single distant point.

GROWING SEASON GATING
----------------------
Output dates only start in May (configurable via `season_start`).
January-April observations are never turned into their own output rows,
but they DO remain available as history for early-May outputs whose
kernel window reaches back that far.

PERFORMANCE NOTES
------------------
  - The per-output-date history loop is bounded to
    [target - max_radius_days, target + wait_days], rather than growing
    across the whole season -- contributions beyond max_radius_days are
    numerically negligible, and nothing past target + wait_days is
    available at processing time anyway.
  - Per-(tile, VI) bookkeeping (which bands are valid, and running
    nearest-valid-date-before / nearest-valid-date-after arrays) is
    computed ONCE via _precompute_daily_stats() and reused across every
    output date.
  - plot_pixel_comparison() does a small windowed raster read instead of
    loading the whole tile, when a specific pixel is requested.
  - run_year() can process tiles in parallel via `max_workers`, the same
    ProcessPoolExecutor pattern used in force_pull.py.

OPERATIONAL USE
-----------------
Because output dates simply aren't eligible until `target_date +
wait_days` has passed, this can be scheduled to run every few days (e.g.
via cron): each run only produces values for dates that have newly
become eligible since the last run, and re-running over a full archived
season produces the same series a rolling operational deployment would
have produced date-by-date, since eligibility never depends on what
other output dates have been computed.

Place this file alongside force_pull.py (e.g. src/disco_ch/delayed_interp.py)
so the relative import below resolves.
"""

import os
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from src.disco_ch.force_pull import (
    DATE_RE,
    find_vi_file,
    discover_tiles,
    VI_KEYS,
    _resolve_vi_keys,
)

# Same kernel widths (in days) the original RBF-smoothed training data used.
DEFAULT_WIDTHS_DAYS = (10, 20, 30, 50)

# How long we wait after a target date before we're willing to compute it
# operationally -- "a week or so" gives the ~5-day satellite revisit a
# real chance to deliver at least one forward observation.
DEFAULT_WAIT_DAYS = 10

# How far back we'll accept the most recent real observation before the
# target date. Beyond this, a pixel is reported as NaN rather than
# smoothed from stale backward-only data.
DEFAULT_MAX_BACKWARD_GAP_DAYS = 20


# ---------------------------------------------------------------------
# Raw data loading (identical convention to force_pull.get_vi_band)
# ---------------------------------------------------------------------

def _collapse_bands_to_daily(arr, descriptions):
    """
    Shared by load_vi_cube() and load_vi_pixel_series(): parses each
    band's date from its description and averages same-day duplicates
    (e.g. two satellite overpasses on one date).
    """
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
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices are expected
            daily_cube[out_i] = np.nanmean(cube[same_day_idx], axis=0)

    return dates, daily_cube


def load_vi_cube(tif_path):
    """
    Reads an entire VI multiband TIF into memory and collapses it to one
    layer per calendar date.

    Returns:
        dates: sorted list of date objects, one per distinct acquisition date
        daily_cube: float32 array (n_dates, height, width), nodata -> NaN
        transform, crs: from the source raster, for writing outputs later
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(masked=True).astype("float32").filled(np.nan)
        transform = src.transform
        crs = src.crs
        descriptions = src.descriptions

    dates, daily_cube = _collapse_bands_to_daily(arr, descriptions)
    return dates, daily_cube, transform, crs


def load_vi_pixel_series(tif_path, row, col, buffer=0):
    """
    Like load_vi_cube(), but only reads a small window around (row, col)
    from disk -- avoids paying the full-tile I/O cost just to look at one
    pixel's time series. buffer=0 reads a single pixel column.

    Returns dates, daily_cube (windowed), and the pixel's row/col *within
    that window* (0, 0 for buffer=0, since the window edge may have been
    clamped to the raster bounds near an edge/corner pixel).
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


# ---------------------------------------------------------------------
# Symmetric composite kernel (identical to the original training kernel)
# ---------------------------------------------------------------------

def gaussian_weight(delta_days, width_days):
    """
    Symmetric Gaussian weight for an observation `delta_days` from the
    output date (can be negative -- before -- or positive -- after).
    Same functional form as one of the four kernels used in training
    (std = width_days).
    """
    return np.exp(-0.5 * (delta_days / width_days) ** 2)


def default_scale_weights(widths=DEFAULT_WIDTHS_DAYS):
    """
    How much each width contributes to the composite kernel, normalized
    to sum to 1. Defaults to inverse-width weighting, so the narrow (10d)
    kernel dominates and the wide (50d) one mostly just fills in when
    nothing recent is available -- matching the "weighted much higher for
    the 10d vs further out" recollection of the original scheme. Swap
    this for the exact original weights if/when you confirm them.
    """
    raw = np.array([1.0 / w for w in widths], dtype="float64")
    return raw / raw.sum()


def default_max_radius_days(widths=DEFAULT_WIDTHS_DAYS, n_sigma=4):
    """
    Contributions beyond n_sigma standard deviations of the widest kernel
    component are numerically negligible (4 sigma of a 50d kernel is
    exp(-8) ~ 3e-4). Bounds the *backward* side of the history lookup;
    the forward side is separately bounded by `wait_days` (see below),
    since nothing past that is available at processing time anyway.
    """
    return int(n_sigma * max(widths))


def make_composite_kernel(widths=DEFAULT_WIDTHS_DAYS, scale_weights=None):
    """
    Builds a single kernel_weight_fn(delta_days) -> weight, combining the
    four Gaussians into one composite curve. This is evaluated for both
    negative (past) and positive (future) delta_days -- i.e. this is the
    same symmetric kernel the model was trained with, not a truncated
    version of it.
    """
    if scale_weights is None:
        scale_weights = default_scale_weights(widths)
    widths_arr = np.asarray(widths, dtype="float64")
    scale_weights = np.asarray(scale_weights, dtype="float64")
    scale_weights = scale_weights / scale_weights.sum()

    def kernel(delta_days):
        return float(np.sum(scale_weights * np.exp(-0.5 * (delta_days / widths_arr) ** 2)))

    return kernel


# ---------------------------------------------------------------------
# Delayed symmetric smoothing
# ---------------------------------------------------------------------

def _precompute_daily_stats(dates, daily_cube):
    """
    Computed ONCE per (tile, VI) and reused across every output date:
      - valid_mask: boolean cube, True where a real observation exists
      - ordinals: date ordinals, shape (n_dates,), ascending
      - last_valid_cum: per-pixel running "most recent valid date-ordinal
        seen up through date index i" (forward-filled from the past).
      - next_valid_cum: per-pixel running "soonest valid date-ordinal
        seen from date index i onward" (backward-filled from the
        future). Together these turn both the backward- and
        forward-coverage checks for any output date into O(1) array
        lookups instead of re-scanning history every time.
    """
    valid_mask = ~np.isnan(daily_cube)
    ordinals = np.array([d.toordinal() for d in dates], dtype="int64")

    ordinal_if_valid = np.where(valid_mask, ordinals[:, None, None], -1)
    last_valid_cum = np.maximum.accumulate(ordinal_if_valid, axis=0)

    big = np.iinfo("int64").max
    ordinal_if_valid_rev = np.where(valid_mask, ordinals[:, None, None], big)
    next_valid_cum = np.minimum.accumulate(ordinal_if_valid_rev[::-1], axis=0)[::-1]

    return valid_mask, ordinals, last_valid_cum, next_valid_cum


def delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                             target_date, kernel_weight_fn,
                             wait_days=DEFAULT_WAIT_DAYS,
                             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                             max_radius_days=None):
    """
    Computes one symmetrically-interpolated 2D raster representing
    `target_date`, using only observations that would actually be on
    hand `wait_days` after that date.

    Takes the precomputed outputs of _precompute_daily_stats() rather
    than recomputing them, so a whole season of output dates can share
    one pass over the raw data.

    History window used: [target_date - max_radius_days, target_date +
    wait_days]. Everything within it is weighted by the symmetric
    composite kernel (negative deltas for the past, positive for the
    future) and renormalized over whatever real observations exist in
    that window -- a standard boundary correction for kernel smoothing
    (Gasser & Muller 1979), applied here for genuine season-edge effects
    rather than to paper over missing data (see the coverage gate below).

    A pixel is set to NaN unless BOTH:
      - it has a real observation within max_backward_gap_days before
        target_date, and
      - it has a real observation within wait_days after target_date
    are satisfied -- i.e. there's real data on both sides to smooth
    between, not just renormalized weight from one distant point.
    """
    if max_radius_days is None:
        max_radius_days = default_max_radius_days()

    target_ordinal = target_date.toordinal()
    window_start_ordinal = target_ordinal - max_radius_days
    window_end_ordinal = target_ordinal + wait_days

    start_idx = int(np.searchsorted(ordinals, window_start_ordinal, side="left"))
    end_idx = int(np.searchsorted(ordinals, window_end_ordinal, side="right")) - 1

    h, w = daily_cube.shape[1:]
    if end_idx < start_idx:
        return np.full((h, w), np.nan, dtype="float32")

    weight_sum = np.zeros((h, w), dtype="float32")
    value_sum = np.zeros((h, w), dtype="float32")

    for i in range(start_idx, end_idx + 1):
        valid = valid_mask[i]
        if not valid.any():
            continue
        wgt = kernel_weight_fn(ordinals[i] - target_ordinal)
        layer = daily_cube[i]
        weight_sum[valid] += wgt
        value_sum[valid] += wgt * layer[valid]

    has_data = weight_sum > 0

    # Backward coverage: most recent real observation up through the end
    # of the window must be within max_backward_gap_days *before* the
    # target date (an observation strictly after the target doesn't
    # count as "backward" coverage).
    backward_idx = min(int(np.searchsorted(ordinals, target_ordinal, side="right")) - 1, end_idx)
    if backward_idx < 0:
        last_valid_ordinal = np.full((h, w), -1, dtype="int64")
    else:
        last_valid_ordinal = last_valid_cum[backward_idx]
    backward_gap = np.where(last_valid_ordinal >= 0, target_ordinal - last_valid_ordinal, np.inf)
    backward_ok = backward_gap <= max_backward_gap_days

    # Forward coverage: soonest real observation from the target date
    # onward, within the window, must exist at all (it's already bounded
    # to <= wait_days by the window itself).
    forward_idx = max(int(np.searchsorted(ordinals, target_ordinal, side="left")), start_idx)
    if forward_idx > end_idx:
        next_valid_ordinal = np.full((h, w), np.iinfo("int64").max, dtype="int64")
    else:
        next_valid_ordinal = next_valid_cum[forward_idx]
    forward_ok = next_valid_ordinal <= window_end_ordinal

    eligible = has_data & backward_ok & forward_ok

    safe_denom = np.where(has_data, weight_sum, 1.0)
    result = np.where(eligible, value_sum / safe_denom, np.nan)
    return result.astype("float32")


# ---------------------------------------------------------------------
# Growing-season output calendar
# ---------------------------------------------------------------------

def growing_season_dates(year, season_start="05-01", season_end="07-24", step_days=5):
    """Every `step_days` days from season_start to season_end (inclusive), for `year`."""
    start = datetime.strptime(f"{year}-{season_start}", "%Y-%m-%d").date()
    end = datetime.strptime(f"{year}-{season_end}", "%Y-%m-%d").date()
    n_steps = (end - start).days // step_days + 1
    return [start + timedelta(days=step_days * i) for i in range(n_steps)]


def eligible_output_dates(all_output_dates, as_of_date, wait_days=DEFAULT_WAIT_DAYS):
    """
    Operational helper: of a full season's output dates, returns only
    those that have become eligible to compute as of `as_of_date` (i.e.
    target_date + wait_days <= as_of_date). Running this on every
    scheduled invocation and only (re)writing newly-eligible dates is
    what makes the module usable as a recurring job -- eligibility never
    depends on what's already been computed, so partial/rolling runs and
    a single full-season run produce identical results.
    """
    return [d for d in all_output_dates if d + timedelta(days=wait_days) <= as_of_date]


# ---------------------------------------------------------------------
# Full series / tile / year orchestration
# ---------------------------------------------------------------------

def delayed_interpolate_series(tif_path, output_dates, kernel_weight_fn=None,
                                wait_days=DEFAULT_WAIT_DAYS,
                                max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                max_radius_days=None):
    """Runs delayed_smooth_one_date() over every output date for one VI TIF, producing one series."""
    if kernel_weight_fn is None:
        kernel_weight_fn = make_composite_kernel()
    if max_radius_days is None:
        max_radius_days = default_max_radius_days()

    dates, daily_cube, transform, crs = load_vi_cube(tif_path)
    valid_mask, ordinals, last_valid_cum, next_valid_cum = _precompute_daily_stats(dates, daily_cube)

    stack = np.stack([
        delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                 d, kernel_weight_fn, wait_days, max_backward_gap_days, max_radius_days)
        for d in output_dates
    ])
    return stack, transform, crs


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


def _process_one_tile(tile_path, year, output_root, vi_key_map, widths, scale_weights,
                       output_dates, wait_days, max_backward_gap_days, max_radius_days,
                       vi_pattern_template):
    """
    Runs the delayed symmetric composite interpolation for every VI in
    one tile. Defined at module level (not nested) so it's picklable for
    ProcessPoolExecutor -- takes plain data (widths/scale_weights) rather
    than a kernel closure/function object, and rebuilds the kernel
    function locally inside the worker process.
    """
    kernel_weight_fn = make_composite_kernel(widths, scale_weights)
    tile_id = os.path.basename(tile_path)
    out_dir = os.path.join(output_root, tile_id)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for canonical, file_token in vi_key_map.items():
        tif_path = find_vi_file(tile_path, file_token, year, vi_pattern_template)
        stack, transform, crs = delayed_interpolate_series(
            tif_path, output_dates, kernel_weight_fn=kernel_weight_fn,
            wait_days=wait_days, max_backward_gap_days=max_backward_gap_days,
            max_radius_days=max_radius_days,
        )
        out_path = os.path.join(out_dir, f"{canonical}_delayed_{year}.tif")
        write_multiband_tif(out_path, stack, transform, crs, output_dates)
        written.append(out_path)

    return tile_id, written


def run_year(root_dir, year, output_root, vi_keys=None,
             widths=DEFAULT_WIDTHS_DAYS, scale_weights=None,
             wait_days=DEFAULT_WAIT_DAYS,
             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
             max_radius_days=None,
             season_start="05-01", season_end="10-31", step_days=5,
             as_of_date=None,
             vi_pattern_template="{year}*_{vi}_TSS.tif",
             max_workers=1):
    """
    Runs delayed symmetric composite-kernel interpolation for every tile
    under `root_dir`, for every requested VI, and writes ONE multiband
    GeoTIFF per (tile, VI) to
    output_root/<tile_id>/<VI>_delayed_<year>.tif.

    :param vi_keys: plain list or {canonical_name: file_token} dict, same
        convention as force_pull._resolve_vi_keys(). Defaults to VI_KEYS.
    :param widths / scale_weights: kernel widths (days) and how much each
        contributes to the composite -- defaults to the same four values
        used in training (10/20/30/50), weighted toward the narrow end.
    :param wait_days: operational latency budget -- an output date isn't
        computed until this many days after it. ~10 ("a week or so" plus
        margin for the ~5-day revisit) is a reasonable default; increase
        for archival/QA reruns if you want maximum forward coverage.
    :param max_radius_days: bounds the *backward* side of history lookup
        per output date; computed automatically from `widths` when left
        None. The forward side is bounded by `wait_days` instead.
    :param as_of_date: if given, only output dates with
        (date + wait_days) <= as_of_date are processed -- this is what
        makes a recurring/cron-style operational run behave correctly
        (only newly-eligible dates get (re)computed). Defaults to
        processing every date in the season regardless of "today" (the
        archival/offline mode).
    :param max_workers: number of tiles to process concurrently. 1 (the
        default) runs sequentially; tiles are fully independent, so
        raising this is a straightforward win when processing many tiles.
    """
    vi_key_map = _resolve_vi_keys(vi_keys)
    if max_radius_days is None:
        max_radius_days = default_max_radius_days(widths)

    output_dates = growing_season_dates(year, season_start, season_end, step_days)
    if as_of_date is not None:
        output_dates = eligible_output_dates(output_dates, as_of_date, wait_days)

    if not output_dates:
        print("No output dates are eligible yet -- nothing to do.")
        return

    print(f"Output dates ({len(output_dates)}): "
          f"{output_dates[0]} .. {output_dates[-1]}, every {step_days}d, wait_days={wait_days}")

    tiles = discover_tiles(root_dir)
    print(f"Found {len(tiles)} tile folders under {root_dir}")

    if max_workers <= 1:
        for tile_path in tiles:
            tile_id, written = _process_one_tile(
                tile_path, year, output_root, vi_key_map, widths, scale_weights,
                output_dates, wait_days, max_backward_gap_days, max_radius_days, vi_pattern_template,
            )
            for p in written:
                print(f"  {tile_id} -> {p}")
    else:
        print(f"Processing tiles with max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_one_tile, tile_path, year, output_root, vi_key_map,
                    widths, scale_weights, output_dates, wait_days, max_backward_gap_days,
                    max_radius_days, vi_pattern_template,
                ): tile_path
                for tile_path in tiles
            }
            for fut in as_completed(futures):
                tile_path = futures[fut]
                fallback_id = os.path.basename(tile_path)
                try:
                    tile_id, written = fut.result()
                except Exception as e:
                    print(f"  Tile {fallback_id} failed: {type(e).__name__}: {e}")
                    continue
                for p in written:
                    print(f"  {tile_id} -> {p}")


# ---------------------------------------------------------------------
# Point comparison plots
# ---------------------------------------------------------------------

def find_sample_pixel(daily_cube, min_obs=10):
    """Convenience: locate a pixel with reasonable observation density, for quick plotting."""
    obs_count = np.sum(~np.isnan(daily_cube), axis=0)
    candidates = np.argwhere(obs_count >= min_obs)
    if len(candidates) == 0:
        raise ValueError(f"No pixel found with at least {min_obs} valid observations")
    row, col = candidates[len(candidates) // 2]
    return int(row), int(col)


def plot_pixel_comparison(tif_path, year, row=None, col=None,
                           widths=DEFAULT_WIDTHS_DAYS, scale_weights=None,
                           wait_days=DEFAULT_WAIT_DAYS,
                           max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                           max_radius_days=None,
                           season_start="03-01", season_end="10-31", step_days=5,
                           show_components=True, title=None, output_path=None):
    """
    Diagnostic plot for a single pixel: raw observations (scatter), the
    final composite delayed-symmetric series (bold line -- this is what
    run_year() actually writes to disk), and optionally each width's
    individual component (thin dashed lines) so you can see how much
    each scale is contributing and whether a real decline survives.

    When row/col are given explicitly, only a small window around that
    pixel is read from disk. If row/col are omitted, a full-tile read is
    still needed to search for a pixel with decent observation density.
    """
    if max_radius_days is None:
        max_radius_days = default_max_radius_days(widths)

    if row is not None and col is not None:
        dates, daily_cube, local_row, local_col = load_vi_pixel_series(tif_path, row, col, buffer=0)
    else:
        dates, daily_cube, _, _ = load_vi_cube(tif_path)
        row, col = find_sample_pixel(daily_cube)
        local_row, local_col = row, col

    valid_mask, ordinals, last_valid_cum, next_valid_cum = _precompute_daily_stats(dates, daily_cube)
    raw_series = daily_cube[:, local_row, local_col]
    output_dates = growing_season_dates(year, season_start, season_end, step_days)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(dates, raw_series, color="black", s=28, zorder=5, label="Raw observations")

    if show_components:
        for width in widths:
            component_fn = lambda delta, w=width: gaussian_weight(delta, w)
            component = np.array([
                delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                         d, component_fn, wait_days, max_backward_gap_days,
                                         max_radius_days)[local_row, local_col]
                for d in output_dates
            ])
            ax.plot(output_dates, component, linestyle="--", linewidth=1, alpha=0.5, label=f"{width}d component")

    composite_fn = make_composite_kernel(widths, scale_weights)
    composite = np.array([
        delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                 d, composite_fn, wait_days, max_backward_gap_days,
                                 max_radius_days)[local_row, local_col]
        for d in output_dates
    ])
    ax.plot(output_dates, composite, color="tab:red", linewidth=2.5, marker="o", markersize=4,
            label="Composite (production output)")

    ax.set_title(title or f"Pixel (row={row}, col={col}) -- {os.path.basename(tif_path)}")
    ax.set_ylabel("VI value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    return fig, (row, col)


if __name__ == "__main__":
    # Example usage -- adjust paths/year before running.
    ROOT_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss"
    OUTPUT_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\interpolated_vi"
    YEAR = 2026

    # Archival/offline mode: process every date in the range regardless
    # of "today". For a recurring operational run, pass as_of_date (e.g.
    # datetime.now().date()) so only newly-eligible dates get computed.
    # run_year(ROOT_DIR, YEAR, OUTPUT_ROOT, season_start="03-01", season_end="07-25",
    #           wait_days=DEFAULT_WAIT_DAYS, max_radius_days=DEFAULT_WAIT_DAYS, max_workers=10)

    # Quick visual sanity check on one tile's NDVI.
    example_tif = find_vi_file(
        os.path.join(ROOT_DIR, discover_tiles(ROOT_DIR)[24]), "NDVI", YEAR
    )
    plot_pixel_comparison(example_tif, YEAR, row=700, col=210, output_path="pixel_comparison_delayed_example.png")
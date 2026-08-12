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

Truncation: all kernels are capped to `max_radius_days` (100d default, per
Hemmerling et al. 2021), applied per side.
"""

import os
import glob
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
)

# Kernel widths (days) matching the original RBF-smoothed training data.
DEFAULT_WIDTHS_DAYS = (10, )

# Days to wait after a target date before computing it operationally
# (gives the ~5-day satellite revisit a chance to deliver a forward obs).
DEFAULT_WAIT_DAYS = 10

# Max temporal distance (per side) any kernel may draw observations from.
DEFAULT_MAX_RADIUS_DAYS = 100

# Max allowed gap to the most recent real observation before target_date.
DEFAULT_MAX_BACKWARD_GAP_DAYS = 20


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


# ---------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------

def gaussian_weight(delta_days, width_days):
    """Symmetric Gaussian weight for an observation delta_days from the output date."""
    return np.exp(-0.5 * (delta_days / width_days) ** 2)


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
                             widths, start_idx, end_idx):
    """Vectorized pass over window [start_idx, end_idx], computing weighted
    sums/masses for ALL kernel widths at once via two tensordot contractions.

    Returns:
        values_stack: (n_widths, h, w) weighted mean per width (NaN if zero mass)
        masses_stack: (n_widths, h, w) realized weight mass per width
    """
    sub_ordinals = ordinals[start_idx:end_idx + 1].astype("float64")
    deltas = sub_ordinals - float(target_ordinal)  # (window_size,)
    widths_arr = np.asarray(widths, dtype="float64")  # (n_widths,)

    # (n_widths, window_size) Gaussian weight matrix, one row per width.
    weight_matrix = np.exp(-0.5 * (deltas[None, :] / widths_arr[:, None]) ** 2).astype("float32")

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
                             max_radius_days=DEFAULT_MAX_RADIUS_DAYS):
    """Compute one ensemble-smoothed 2D raster for target_date, using only
    observations available wait_days after that date.

    A pixel is NaN unless it has a real observation within
    max_backward_gap_days before target_date AND within wait_days after it,
    even if the ensemble weighting alone would produce a value.
    """
    target_ordinal = target_date.toordinal()
    window_start_ordinal = target_ordinal - max_radius_days
    window_end_ordinal = target_ordinal + min(wait_days, max_radius_days)

    start_idx = int(np.searchsorted(ordinals, window_start_ordinal, side="left"))
    end_idx = int(np.searchsorted(ordinals, window_end_ordinal, side="right")) - 1

    h, w = daily_cube.shape[1:]
    if end_idx < start_idx:
        return np.full((h, w), np.nan, dtype="float32")

    values_stack, masses_stack = _ensemble_weighted_sums(
        daily_cube, valid_mask, ordinals, target_ordinal, widths, start_idx, end_idx,
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

    eligible = has_any & backward_ok & forward_ok

    result = np.where(eligible, ensemble_value, np.nan)
    return result.astype("float32")


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
                                max_backward_gap_days, max_radius_days):
    """Shared core of delayed_interpolate_series() / delayed_interpolate_series_from_archive():
    given an already-loaded (dates, daily_cube), runs delayed_smooth_one_date()
    over every output date."""
    valid_mask, ordinals, last_valid_cum, next_valid_cum = _precompute_daily_stats(dates, daily_cube)
    return np.stack([
        delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                 d, widths, wait_days, max_backward_gap_days, max_radius_days)
        for d in output_dates
    ])


def delayed_interpolate_series(tif_path, output_dates, widths=DEFAULT_WIDTHS_DAYS,
                                wait_days=DEFAULT_WAIT_DAYS,
                                max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                max_radius_days=DEFAULT_MAX_RADIUS_DAYS):
    """Run delayed_smooth_one_date() over every output date for one VI TIF."""
    dates, daily_cube, transform, crs = load_vi_cube(tif_path)
    stack = _delayed_interpolate_stack(dates, daily_cube, output_dates, widths, wait_days,
                                        max_backward_gap_days, max_radius_days)
    return stack, transform, crs


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


def delayed_interpolate_series_from_archive(archive_dir, output_dates, widths=DEFAULT_WIDTHS_DAYS,
                                             wait_days=DEFAULT_WAIT_DAYS,
                                             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                                             max_radius_days=DEFAULT_MAX_RADIUS_DAYS):
    """Archive-directory equivalent of delayed_interpolate_series(), for the
    STAC pipeline where VI dates arrive one scene at a time instead of as a
    premade FORCE TSA product."""
    dates, daily_cube, transform, crs = load_vi_cube_from_archive(archive_dir)
    stack = _delayed_interpolate_stack(dates, daily_cube, output_dates, widths, wait_days,
                                        max_backward_gap_days, max_radius_days)
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


def _process_one_tile(tile_path, year, output_root, vi_key_map, widths,
                       output_dates, wait_days, max_backward_gap_days, max_radius_days,
                       vi_pattern_template, skip_existing=True):
    """Run the delayed ensemble interpolation for every VI in one tile.

    Module-level (not nested) so it's picklable for ProcessPoolExecutor.
    skip_existing=True skips a (tile, VI) if its output file already exists,
    with no recompute check against output_dates/widths/etc -- delete the
    file (or pass skip_existing=False) to force a recompute.
    """
    tile_id = os.path.basename(tile_path)
    out_dir = os.path.join(output_root, tile_id)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    skipped = []
    for canonical, file_token in vi_key_map.items():
        tif_path = find_vi_file(tile_path, file_token, year, vi_pattern_template)
        base, ext = os.path.splitext(os.path.basename(tif_path))
        out_path = os.path.join(out_dir, f"{base}_rbf{ext}")

        if skip_existing and os.path.exists(out_path):
            skipped.append(out_path)
            continue

        stack, transform, crs = delayed_interpolate_series(
            tif_path, output_dates, widths=widths,
            wait_days=wait_days, max_backward_gap_days=max_backward_gap_days,
            max_radius_days=max_radius_days,
        )
        write_multiband_tif(out_path, stack, transform, crs, output_dates)
        written.append(out_path)

    return tile_id, written, skipped


def run_year(root_dir, year, output_root, vi_keys=None,
             widths=DEFAULT_WIDTHS_DAYS,
             wait_days=DEFAULT_WAIT_DAYS,
             max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
             max_radius_days=DEFAULT_MAX_RADIUS_DAYS,
             season_start="05-01", season_end="10-31", step_days=5,
             as_of_date=None,
             vi_pattern_template="{year}*_{vi}_TSS.tif",
             max_workers=1,
             skip_existing=True):
    """Run delayed ensemble-kernel interpolation for every tile under root_dir,
    for every requested VI, writing one multiband GeoTIFF per (tile, VI) to
    output_root/<tile_id>/<VI>_delayed_<year>.tif.

    :param vi_keys: list or {canonical_name: file_token} dict, defaults to VI_KEYS.
    :param as_of_date: if given, only process dates with (date + wait_days) <=
        as_of_date -- makes recurring/cron runs only compute newly-eligible dates.
        Defaults to processing the whole season (archival/offline mode).
    :param max_workers: tiles are independent; >1 processes them concurrently.
    :param skip_existing: if True, skip a (tile, VI) whose output file already
        exists on disk (fast path check only, no recompute logic).
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

    tiles = discover_tiles(root_dir)
    print(f"Found {len(tiles)} tile folders under {root_dir}")

    if skip_existing:
        print("skip_existing=True: tiles/VIs with an existing output file will be skipped.")

    total_written = 0
    total_skipped = 0

    if max_workers <= 1:
        for tile_path in tiles:
            tile_id, written, skipped = _process_one_tile(
                tile_path, year, output_root, vi_key_map, widths,
                output_dates, wait_days, max_backward_gap_days, max_radius_days,
                vi_pattern_template, skip_existing,
            )
            for p in written:
                print(f"  {tile_id} -> {p}")
            for p in skipped:
                print(f"  {tile_id} -> skipped (already exists): {p}")
            total_written += len(written)
            total_skipped += len(skipped)
    else:
        print(f"Processing tiles with max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_one_tile, tile_path, year, output_root, vi_key_map,
                    widths, output_dates, wait_days, max_backward_gap_days,
                    max_radius_days, vi_pattern_template, skip_existing,
                ): tile_path
                for tile_path in tiles
            }
            for fut in as_completed(futures):
                tile_path = futures[fut]
                fallback_id = os.path.basename(tile_path)
                try:
                    tile_id, written, skipped = fut.result()
                except Exception as e:
                    print(f"  Tile {fallback_id} failed: {type(e).__name__}: {e}")
                    continue
                for p in written:
                    print(f"  {tile_id} -> {p}")
                for p in skipped:
                    print(f"  {tile_id} -> skipped (already exists): {p}")
                total_written += len(written)
                total_skipped += len(skipped)

    print(f"Done. Wrote {total_written} file(s), skipped {total_skipped} already-existing file(s).")


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


def plot_pixel_comparison(tif_path, year, row=None, col=None,
                           widths=DEFAULT_WIDTHS_DAYS,
                           wait_days=DEFAULT_WAIT_DAYS,
                           max_backward_gap_days=DEFAULT_MAX_BACKWARD_GAP_DAYS,
                           max_radius_days=DEFAULT_MAX_RADIUS_DAYS,
                           season_start="03-01", season_end="10-31", step_days=5,
                           show_components=True, title=None, output_path=None):
    """Diagnostic plot for one pixel: raw observations, each width's
    independent component (dashed), and the final ensemble (bold, what
    run_year() writes to disk). If row/col given, only a small window is
    read from disk; otherwise a full-tile read is needed to find a dense pixel.
    """
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
        for width_days in widths:
            component = np.array([
                delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                         d, (width_days,), wait_days, max_backward_gap_days,
                                         max_radius_days)[local_row, local_col]
                for d in output_dates
            ])
            ax.plot(output_dates, component, linestyle="--", linewidth=1, alpha=0.5, label=f"{width_days}d component")

    ensemble = np.array([
        delayed_smooth_one_date(daily_cube, valid_mask, ordinals, last_valid_cum, next_valid_cum,
                                 d, widths, wait_days, max_backward_gap_days,
                                 max_radius_days)[local_row, local_col]
        for d in output_dates
    ])
    ax.plot(output_dates, ensemble, color="tab:red", linewidth=2.5, marker="o", markersize=4,
            label="Ensemble (production output)")

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
    ROOT_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss_v2"
    OUTPUT_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\interpolated_vi_rbf_10"
    YEAR = 2026

    run_year(ROOT_DIR, YEAR, OUTPUT_ROOT, season_start="03-01", season_end="07-25",
              wait_days=DEFAULT_WAIT_DAYS, max_radius_days=DEFAULT_MAX_RADIUS_DAYS, max_workers=5)

    example_tif = find_vi_file(
        os.path.join(ROOT_DIR, discover_tiles(ROOT_DIR)[24]), "NDVI", YEAR
    )
    plot_pixel_comparison(
        example_tif, YEAR, row=700, col=210,
        widths=(10,),  # only the 10d kernel
        output_path="pixel_comparison_10d_only.png",
    )
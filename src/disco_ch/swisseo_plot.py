"""
swisseo_plot.py

Single-pixel diagnostic plotting for swisseo_stepwise.py -- the
archive-directory counterpart to force_tsi_plot.py. Two ways to get a
pixel's raw series in:

    - plot_pixel(archive_dir, ...): reads from an already-populated
      swisseo per-VI archive directory (archive_root/<VI>/<date>.tif,
      one file per date -- see stac_pull.archive_vi_raster()). Fast --
      no network involved.
    - plot_pixel_from_stac(vi_key, ...): pulls the pixel's series
      DIRECTLY from the STAC API for a given date range -- no local
      archive needed at all. Hits the network once per scene in range
      (via stac_pull.pull_from_stac()), so it's meaningfully slower;
      useful for checking a pixel before anything's been archived for
      it yet, or to sanity-check the archive against a fresh pull.

Both run the exact same despike() + rbf_interpolate() from force_tsi.py
and plot:

    - raw observations (black dots)
    - any points despike() removed as outliers (orange x)
    - each RBF_SIGMA's own component, independently (dashed)
    - the final ensemble -- what run_swisseo_stepwise() would write (bold red)

via a shared _plot_pixel_core() (same pattern as
rbf_interp.py's _plot_pixel_comparison() serving both its
archive-dir and premade-TIF entry points). find_sample_pixel() and
_stepwise_series() are reused directly from force_tsi_plot.py rather
than re-implemented here -- both only operate on an already-loaded
(dates, cube), with no FORCE-specific coupling at all.

Meant for quickly sanity-checking parameter choices (ABOVE_NOISE,
BELOW_NOISE, RBF_SIGMA, RBF_CUTOFF, INT_DAY) against real swisseo data
before committing to a full run_swisseo_stepwise() call.

stepwise=True adds a second curve simulating what
swisseo_stepwise.run_swisseo_stepwise() would have written for this one
pixel -- without running the batch pipeline at all -- so you can "play
around" with wait_days on a single pixel first. See
force_tsi_plot._stepwise_series() for exactly what it simulates and its
one simplifying assumption. stepwise=False (the default) is unchanged,
original behavior for both entry points -- stepwise is strictly
additive, never required.
"""

import warnings

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from src.disco_ch.force_tsi import (
    despike,
    rbf_interpolate,
    output_dates_from_range,
)
from src.disco_ch.force_tsi_plot import find_sample_pixel, _stepwise_series
from src.disco_ch.force_tsi_stepwise import DEFAULT_WAIT_DAYS, recommended_wait_days
from src.disco_ch.swisseo_stepwise import (
    load_vi_cube_from_archive,
    load_vi_pixel_series_from_archive,
)


def _plot_pixel_core(dates, cube, local_row, local_col, row, col,
                      date_range, doy_range, int_day,
                      rbf_sigma, rbf_cutoff, above_noise, below_noise,
                      show_components, title, output_path,
                      stepwise, wait_days, show_full_archive_comparison,
                      ensemble_label_default):
    """Shared plotting core of plot_pixel() / plot_pixel_from_stac(): given
    an already-loaded pixel series, despikes it and draws the raw
    observations -- with any despiked-out points marked distinctly --
    each width's independent component (dashed), and the final ensemble
    (bold). See plot_pixel()'s docstring for what each parameter means;
    identical between both entry points except `ensemble_label_default`,
    which distinguishes the two in the plot legend/title.
    """
    if date_range is None:
        date_range = (dates[0], dates[-1])

    cleaned, removed = despike(dates, cube, above_noise, below_noise)

    raw_series = cube[:, local_row, local_col]
    removed_series = removed[:, local_row, local_col]

    output_dates = output_dates_from_range(date_range, int_day, doy_range)

    fig, ax = plt.subplots(figsize=(11, 5))
    date_arr = np.array(dates)
    kept = ~removed_series
    ax.scatter(date_arr[kept], raw_series[kept], color="black", s=28, zorder=5, label="Raw observations")
    if removed_series.any():
        ax.scatter(date_arr[removed_series], raw_series[removed_series], color="darkorange", marker="x",
                   s=70, linewidths=2, zorder=6, label="Removed as outlier")

    show_full_archive = not stepwise or show_full_archive_comparison
    full_archive_alpha = 1.0 if not stepwise else 0.45

    if show_components and show_full_archive:
        for sigma in rbf_sigma:
            component = rbf_interpolate(dates, cleaned, output_dates, (sigma,), rbf_cutoff)[:, local_row, local_col]
            ax.plot(output_dates, component, linestyle="--", linewidth=1, alpha=0.5 * full_archive_alpha,
                    color="tab:red" if stepwise else None, label=f"{sigma}d component")

    if show_full_archive:
        ensemble = rbf_interpolate(dates, cleaned, output_dates, rbf_sigma, rbf_cutoff)[:, local_row, local_col]
        ensemble_label = "Full-archive ensemble (reference)" if stepwise else ensemble_label_default
        ax.plot(output_dates, ensemble, color="tab:red", linewidth=2.5, alpha=full_archive_alpha, label=ensemble_label)
        valid = ~np.isnan(ensemble)
        ax.scatter(np.array(output_dates)[valid], ensemble[valid], color="tab:red", s=20, zorder=7, alpha=full_archive_alpha)

    if stepwise:
        if wait_days is None:
            wait_days = DEFAULT_WAIT_DAYS
        rec = recommended_wait_days(rbf_sigma, rbf_cutoff)
        print(f"Stepwise: wait_days={wait_days:g} (recommended_wait_days() for this rbf_sigma/rbf_cutoff "
              f"is {rec:.1f} -- below that, stepwise output isn't guaranteed to match an eventual "
              f"full-archive result; see force_tsi_stepwise.py's module docstring).")

        pixel_cube = cube[:, local_row:local_row + 1, local_col:local_col + 1]
        if show_components:
            for sigma in rbf_sigma:
                component_sw = _stepwise_series(dates, pixel_cube, output_dates, wait_days,
                                                 above_noise, below_noise, (sigma,), rbf_cutoff)
                ax.plot(output_dates, component_sw, linestyle=":", linewidth=1, alpha=0.6,
                        color="tab:blue", label=f"{sigma}d component (stepwise)")

        ensemble_stepwise = _stepwise_series(dates, pixel_cube, output_dates, wait_days,
                                              above_noise, below_noise, rbf_sigma, rbf_cutoff)
        ax.plot(output_dates, ensemble_stepwise, color="tab:blue", linewidth=2.5,
                label=f"Stepwise ensemble (wait_days={wait_days:g})")
        valid_sw = ~np.isnan(ensemble_stepwise)
        ax.scatter(np.array(output_dates)[valid_sw], ensemble_stepwise[valid_sw], color="tab:blue", s=20, zorder=7)

    default_title = f"Pixel (row={row}, col={col})"
    if stepwise:
        default_title += f" -- stepwise wait_days={wait_days:g}"
    ax.set_title(title or default_title)
    ax.set_ylabel("VI value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    return fig, (row, col), dates, cube, cleaned, removed


def plot_pixel(archive_dir, row=None, col=None,
                date_range=None, doy_range=(1, 365), int_day=5,
                rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                above_noise=3.0, below_noise=1.0,
                show_components=True, title=None, output_path=None,
                stepwise=False, wait_days=None, show_full_archive_comparison=True):
    """Plot one pixel's raw series, despiking, and RBF ensemble, read from
    a swisseo per-VI archive directory.

    If row/col are given, only a tiny window is read from disk -- this
    is the fast path for testing on a pixel you already know. If left
    None, the whole archive extent is read once just to pick a
    reasonably dense pixel automatically (find_sample_pixel()) -- fine
    for a one-off look, but pass row/col directly for repeated testing.

    date_range defaults to the pixel's own first/last observed date if
    not given, so you don't have to know it up front.

    :param archive_dir: one VI's swisseo archive directory
        (archive_root/<vi_key>, see stac_pull.archive_vi_raster()).
    :param stepwise: if True, ALSO computes and plots a simulated
        swisseo_stepwise.run_swisseo_stepwise() curve for this pixel
        (see force_tsi_plot._stepwise_series()) -- lets you "play around"
        with wait_days on one pixel without running the batch stepwise
        pipeline at all. False (the default) is this function's original
        behavior, completely unchanged.
    :param wait_days: only used when stepwise=True; defaults to
        force_tsi_stepwise.DEFAULT_WAIT_DAYS. Prints
        recommended_wait_days() for this rbf_sigma/rbf_cutoff alongside
        the plot, as a reminder of how far below "guaranteed convergence
        to an eventual full-archive result" this value is (see
        force_tsi_stepwise.py's module docstring).
    :param show_full_archive_comparison: only used when stepwise=True --
        if True (the default), the ordinary full-archive ensemble (and
        components, if show_components) are ALSO drawn, dimmed, for
        direct comparison against the stepwise curve. Set False for a
        stepwise-only view.
    :return: (fig, (row, col), dates, cube, cleaned, removed) -- the
        underlying arrays are returned too, in case you want to inspect
        them further in the notebook.
    """
    if row is not None and col is not None:
        dates, cube, local_row, local_col = load_vi_pixel_series_from_archive(archive_dir, row, col, buffer=0)
    else:
        dates, cube, _, _ = load_vi_cube_from_archive(archive_dir)
        row, col = find_sample_pixel(cube)
        local_row, local_col = row, col

    return _plot_pixel_core(
        dates, cube, local_row, local_col, row, col,
        date_range, doy_range, int_day, rbf_sigma, rbf_cutoff, above_noise, below_noise,
        show_components, title, output_path, stepwise, wait_days, show_full_archive_comparison,
        ensemble_label_default="Ensemble (what run_swisseo_stepwise() writes)",
    )


# ---------------------------------------------------------------------
# Pulling a single pixel directly from STAC -- no local archive needed
# ---------------------------------------------------------------------

def _pixel_bbox_from_template(template_path, row, col, buffer=0):
    """Converts a (row, col) pixel index -- interpreted against
    `template_path`'s own grid, since there's no local archive grid to
    reference when pulling directly from STAC -- into a small
    (minx, miny, maxx, maxy) bbox in the template's own CRS. This is the
    same national/AOI reference grid stac_pull.py aligns every archived
    scene onto (see stac_pull.build_template()), so row/col here mean
    the same thing they would once this pixel is actually archived.

    `buffer` is clamped to at least 1 pixel here: load_and_process_assets()
    calls rioxarray's clip_box() on every band, which refuses (raises
    OneDimensionalRaster) a bbox that's exactly one pixel wide in either
    direction -- buffer=0 would produce exactly that. A buffer of 1
    (3x3 pixels) is the smallest box clip_box() accepts, and doesn't
    change plot_pixel_from_stac()'s result: it always reads the CENTER
    pixel of whatever comes back, same as a true 1-pixel box would have
    given anyway.
    """
    buffer = max(buffer, 1)
    with rasterio.open(template_path) as src:
        transform = src.transform
    col_off, row_off = col - buffer, row - buffer
    col_end, row_end = col + buffer + 1, row + buffer + 1
    minx, maxy = transform * (col_off, row_off)
    maxx, miny = transform * (col_end, row_end)
    return minx, miny, maxx, maxy


def plot_pixel_from_stac(vi_key, row, col, start_date, end_date, template_path, forest_mask_path,
                          buffer=0, stac_loc='https://data.geo.admin.ch/api/stac/v0.9/',
                          date_range=None, doy_range=(1, 365), int_day=5,
                          rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                          above_noise=3.0, below_noise=1.0,
                          show_components=True, title=None, output_path=None,
                          stepwise=False, wait_days=None, show_full_archive_comparison=True,
                          verbose=False):
    """Like plot_pixel(), but pulls the pixel's raw series DIRECTLY from
    the STAC API over [start_date, end_date] -- no local swisseo archive
    needed at all. Reuses stac_pull.py's own scene-loading/QC exactly
    (pull_from_stac() -> load_and_process_assets() -> compute_vis()),
    just clipped to a tiny bbox around one pixel instead of the national
    extent, so the actual network transfer per scene stays small.

    This hits the network once per scene in range, so it's meaningfully
    slower than plot_pixel() against an already-archived directory --
    intended for checking a pixel BEFORE anything's been archived for it
    yet, or to sanity-check the archive against a fresh pull.

    :param vi_key: one of "NDVI", "EVI", "NDMI", "CIRE", "CCI".
    :param start_date, end_date: 'YYYY-MM-DD' bounds for the STAC search
        (see stac_pull.pull_from_stac()) -- distinct from `date_range`,
        the output-date calendar the ensemble is evaluated on (defaults
        to [start_date, end_date] if not given).
    :param template_path, forest_mask_path: same rasters
        stac_pull.update_vi_min_max_interpolated() uses. row/col are
        interpreted against template_path's own grid (see
        _pixel_bbox_from_template()). forest_mask_path=None skips the
        forest-mask constraint entirely (matching force_pull.py's
        forest_mask_path=None option) -- every scene's own cloud/terrain
        masking still applies, just not the forest==1 requirement.
        load_and_process_assets() itself has no None-forest-mask toggle
        (it always does `forest_mask == 1`), so None here is implemented
        by passing it the scalar 1 in place of a real mask DataArray --
        `1 == 1` is always True, so the AND has no effect, without
        needing a fake all-ones raster aligned to every scene's exact
        clipped grid.
    :param buffer: extra pixels around row/col to fetch per side (0 =
        just the one pixel's own bbox). A small buffer (e.g. 1-2) can
        help if row/col lands exactly on a clip_box rounding edge.
    :param verbose: prints one line per scene as it's pulled.
    :return: (fig, (row, col), dates, cube, cleaned, removed) -- same
        shape as plot_pixel()'s return (cube here is a single-pixel
        (n_dates, 1, 1) series, since nothing else was ever fetched).
    """
    import rioxarray as rxr
    from src.disco_ch.stac_pull import (
        pull_from_stac, get_item_datetime, load_and_process_assets, compute_vis,
    )

    bbox = _pixel_bbox_from_template(template_path, row, col, buffer)

    if forest_mask_path is not None:
        forest_mask_da = rxr.open_rasterio(forest_mask_path).rio.clip_box(*bbox)
        fm_y_mid = forest_mask_da.sizes["y"] // 2
        fm_x_mid = forest_mask_da.sizes["x"] // 2
        center_forest_flag = float(forest_mask_da.isel(band=0, y=fm_y_mid, x=fm_x_mid).values)
    else:
        # Scalar pass-through: load_and_process_assets() only ever does
        # `forest_mask == 1`, so a plain 1 disables the constraint
        # (broadcasts against every other condition unchanged) without
        # needing a fake mask aligned to each scene's own clipped grid.
        forest_mask_da = 1
        center_forest_flag = None

    try:
        year = int(start_date.split("-")[0])
        items = sorted(
            pull_from_stac(stac_loc=stac_loc, year=year, start_date=start_date, end_date=end_date),
            key=get_item_datetime,
        )

        n_loaded = 0
        n_valid = 0
        values_by_date = {}
        for i, item in enumerate(items):
            if verbose:
                print(f"  Pulling {i + 1}/{len(items)}: {item.id}")
            bands, valid_mask = load_and_process_assets(item, forest_mask_da, bbox=bbox, verbose=0)
            if bands is None:
                continue
            n_loaded += 1
            vi_da = compute_vis(bands)[vi_key].where(valid_mask == 1, np.nan)
            y_mid, x_mid = vi_da.sizes["y"] // 2, vi_da.sizes["x"] // 2
            value = float(vi_da.isel(band=0, y=y_mid, x=x_mid).values)
            if np.isfinite(value):
                n_valid += 1

            d = get_item_datetime(item).date()
            values_by_date.setdefault(d, []).append(value)
    finally:
        if forest_mask_path is not None:
            forest_mask_da.close()

    if not values_by_date:
        raise ValueError(f"No usable {vi_key} scenes found between {start_date} and {end_date}")

    print(f"{len(items)} scene(s) found, {n_loaded} loaded successfully, "
          f"{n_valid} with a valid (non-masked) value at this pixel.")
    if n_valid == 0:
        if forest_mask_path is not None:
            reason = (f"The forest mask value there is {center_forest_flag!r} -- "
                      f"load_and_process_assets()'s valid_mask requires forest_mask == 1, so this "
                      f"pixel will NEVER produce a value if it isn't forest. Cloud/terrain masking "
                      f"excluding every single scene in range is also possible but far less likely.")
        else:
            reason = ("forest_mask_path=None, so it's not the forest constraint -- every scene's "
                      "own cloud/terrain masking (cloud_mask != 1, terrain_mask <= 63) or red <= 0 "
                      "excluded this pixel on every date instead.")
        print(
            f"  WARNING: every scene was masked out at this exact pixel (row={row}, col={col}). "
            f"{reason} Check whether (row, col) actually falls where you expect before assuming "
            f"this is a bug."
        )

    dates = sorted(values_by_date)
    with warnings.catch_warnings():
        # A date whose only scene(s) were masked out at this pixel is
        # legitimately all-NaN (see the n_valid==0 warning above) --
        # nanmean's "Mean of empty slice" is expected there, not a bug.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        series = np.array([np.nanmean(values_by_date[d]) for d in dates], dtype="float32")
    cube = series.reshape(-1, 1, 1)

    return _plot_pixel_core(
        dates, cube, 0, 0, row, col,
        date_range, doy_range, int_day, rbf_sigma, rbf_cutoff, above_noise, below_noise,
        show_components, title, output_path, stepwise, wait_days, show_full_archive_comparison,
        ensemble_label_default="Ensemble (what run_swisseo_stepwise() writes)",
    )


# Example (from a notebook):
#
#   from src.disco_ch.swisseo_plot import plot_pixel, plot_pixel_from_stac
#
#   fig, (row, col), dates, cube, cleaned, removed = plot_pixel(
#       r"B:\bloomc\DiscoCH_2026_08_03\swisseo\archive\NDVI", row=700, col=210,
#       rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
#       above_noise=3.0, below_noise=1.0, int_day=5,
#   )
#   fig.show()
#
#   # Play with wait_days on this same pixel, pulling its raw series
#   # DIRECTLY from STAC instead of a local archive (slower -- hits the
#   # network once per scene in range -- but works even if this pixel
#   # hasn't been archived yet, or to sanity-check the archive itself
#   # against a fresh pull).
#   fig, *_ = plot_pixel_from_stac(
#       "NDVI", row=700, col=210,
#       start_date="2026-01-01", end_date="2026-10-31",
#       template_path=r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\CH_NoValue_255.tif",
#       forest_mask_path=r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\DRAINS_Forest_Mask.tif",
#       stepwise=True, wait_days=10, verbose=True,
#   )
#   fig.show()

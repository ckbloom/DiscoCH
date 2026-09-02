"""
force_tsi_plot.py

Single-pixel diagnostic plotting for force_tsi.py. Reads just a small
window around one pixel (not the whole tile), runs the exact same
despike() + rbf_interpolate() from force_tsi.py on it, and plots:

    - raw observations (black dots)
    - any points despike() removed as outliers (orange x)
    - each RBF_SIGMA's own component, independently (dashed)
    - the final ensemble -- what run_tsi() would actually write (bold red)

Meant for quickly sanity-checking parameter choices (ABOVE_NOISE,
BELOW_NOISE, RBF_SIGMA, RBF_CUTOFF, INT_DAY) against real data before
committing to a full run_tsi() call.

stepwise=True adds a second curve simulating what force_tsi_stepwise.py
would have written for this one pixel -- without running the batch
pipeline at all -- so you can "play around" with wait_days on a single
pixel first. See _stepwise_series() for exactly what it simulates and
its one simplifying assumption. stepwise=False (the default) is the
plot's original, unchanged full-archive behavior -- stepwise is strictly
additive, never required.

print_effective_bandwidth=True prints, for every output date, the
*realized* smoothing the RBF ensemble actually applied at that point --
see effective_bandwidth_stats() -- rather than the nominal rbf_sigma
values requested. The same rbf_sigma tuple doesn't yield the same
smoothness on sparse vs. dense data (e.g. 2018 vs. 2026): with few real
observations nearby, the "smoother" degenerates toward interpolating a
couple of points; with many, it's a genuine local average. This
diagnostic is how you'd measure that gap directly, e.g. to pick per-year
rbf_sigma values that reproduce a *matched* effective bandwidth instead
of just eyeballing which curve "looks" similarly smooth. Alongside that,
it also prints local_rmse -- how far the fitted curve sits from the real
(despiked) observations feeding it at that point -- as an (in-sample,
optimistic) sense of fit quality at each step.
"""

from datetime import date, timedelta

import numpy as np
import matplotlib.pyplot as plt

from src.disco_ch.force_tsi import (
    load_tss,
    load_tss_pixel,
    despike,
    rbf_interpolate,
    rbf_cutoff_radius,
    output_dates_from_range,
)
from src.disco_ch.force_tsi_stepwise import DEFAULT_WAIT_DAYS, recommended_wait_days


def find_sample_pixel(cube, min_obs=10):
    """Locate a pixel with reasonable observation density, for a quick
    default plot when no row/col is given. Reads the density from an
    already-loaded cube -- callers that don't have one yet should use
    load_tss() once, up front, only for this."""
    obs_count = np.sum(~np.isnan(cube), axis=0)
    candidates = np.argwhere(obs_count >= min_obs)
    if len(candidates) == 0:
        raise ValueError(f"No pixel found with at least {min_obs} valid observations")
    row, col = candidates[len(candidates) // 2]
    return int(row), int(col)


def effective_bandwidth_stats(dates, series, output_dates, rbf_sigma, rbf_cutoff=0.95):
    """
    For one pixel's (already despiked) series, computes -- at every output
    date -- the ACTUAL effective bandwidth/degrees of freedom the RBF
    ensemble achieves, given the real observation density it's fed there.
    This is what makes the same nominal rbf_sigma produce different real
    smoothing on sparse vs. dense data: the requested sigmas describe a
    kernel *shape*, but how much genuine averaging that kernel performs
    depends entirely on how many real observations happen to fall inside
    it at each point in time.

    Reproduces force_tsi.rbf_interpolate()'s own per-observation weighting
    exactly (each sigma's Gaussian, truncated at its own rbf_cutoff_radius,
    summed across sigmas) for one output date at a time, then summarizes
    those weights instead of immediately collapsing them into one averaged
    value:

      - eff_dof: Kish's effective sample size, (sum w)^2 / sum(w^2) --
        "how many independent observations this output value is really
        averaging over", regardless of the nominal window width. A value
        near 1 means the output is essentially just copying one nearby
        observation (no real smoothing achieved); higher values mean
        genuine averaging over several independent points.
      - eff_bandwidth_days: the weighted RMS distance (days) of
        contributing observations from the output date -- the *realized*
        temporal spread of the smoothing actually applied, as opposed to
        the nominal sigma(s) requested.
      - local_rmse: the weighted RMS difference (in VI units) between
        this output date's fitted (ensemble) value and the real nearby
        observations that fed it -- i.e. how far off the RBF curve is
        from the actual data at this point. NOTE: this is an IN-SAMPLE
        residual (computed from the same weighted window the fit itself
        used), not a held-out/cross-validated error, so it's optimistic
        -- especially where eff_dof is low, since a fit hugging just one
        or two nearby points will trivially look like a near-perfect
        match to them. Still directly comparable across pixels/years
        since it always uses the exact same weighting the ensemble
        applies.

    :param series: this pixel's cleaned 1D series (n_dates,), e.g.
        cleaned[:, row, col] from despike() -- also what "actual data" in
        local_rmse is measured against (so points despike() already
        removed as outliers don't inflate it).
    :return: list of dicts, one per output date, each
        {"date", "n_obs", "eff_dof", "eff_bandwidth_days", "rbf_value",
        "local_rmse"} -- all but "date"/"n_obs" are NaN when no real
        observation falls within any sigma's cutoff radius
        (rbf_interpolate would also output NaN there).
    """
    ordinals = np.array([d.toordinal() for d in dates], dtype="float64")
    valid = ~np.isnan(series)

    sigmas = np.asarray(rbf_sigma, dtype="float64")
    radii = np.array([rbf_cutoff_radius(s, rbf_cutoff) for s in sigmas])
    max_radius = radii.max()

    nan_entry = lambda od, n_obs: {
        "date": od, "n_obs": n_obs, "eff_dof": np.nan,
        "eff_bandwidth_days": np.nan, "rbf_value": np.nan, "local_rmse": np.nan,
    }

    stats = []
    for od in output_dates:
        target = od.toordinal()
        idx = np.where((np.abs(ordinals - target) <= max_radius) & valid)[0]
        if idx.size == 0:
            stats.append(nan_entry(od, 0))
            continue

        deltas = ordinals[idx] - target
        weight = np.exp(-0.5 * (deltas[None, :] / sigmas[:, None]) ** 2)
        within = np.abs(deltas)[None, :] <= radii[:, None]
        weight = np.where(within, weight, 0.0).sum(axis=0)  # combined ensemble weight per obs

        wsum = weight.sum()
        if wsum <= 0:
            stats.append(nan_entry(od, int(idx.size)))
            continue

        values = series[idx].astype("float64")
        fitted = np.sum(weight * values) / wsum
        eff_dof = wsum ** 2 / np.sum(weight ** 2)
        eff_bandwidth = np.sqrt(np.sum(weight * deltas ** 2) / wsum)
        local_rmse = np.sqrt(np.sum(weight * (values - fitted) ** 2) / wsum)
        stats.append({
            "date": od, "n_obs": int(idx.size),
            "eff_dof": float(eff_dof), "eff_bandwidth_days": float(eff_bandwidth),
            "rbf_value": float(fitted), "local_rmse": float(local_rmse),
        })

    return stats


def _print_effective_bandwidth(dates, cleaned_series, output_dates, rbf_sigma, rbf_cutoff):
    stats = effective_bandwidth_stats(dates, cleaned_series, output_dates, rbf_sigma, rbf_cutoff)
    print(f"Effective bandwidth/dof/local RMSE per output date (rbf_sigma={rbf_sigma}, rbf_cutoff={rbf_cutoff}):")
    for s in stats:
        if np.isnan(s["eff_dof"]):
            print(f"  {s['date']}: n_obs={s['n_obs']:3d}  (no data within cutoff radius)")
        else:
            print(f"  {s['date']}: n_obs={s['n_obs']:3d}  eff_dof={s['eff_dof']:5.2f}  "
                  f"eff_bandwidth={s['eff_bandwidth_days']:6.1f}d  local_rmse={s['local_rmse']:7.4f}")

    valid_stats = [s for s in stats if not np.isnan(s["eff_dof"])]
    if valid_stats:
        dofs = np.array([s["eff_dof"] for s in valid_stats])
        bws = np.array([s["eff_bandwidth_days"] for s in valid_stats])
        rmses = np.array([s["local_rmse"] for s in valid_stats])
        overall_rmse = np.sqrt(np.mean(rmses ** 2))
        print(f"Summary across {len(valid_stats)}/{len(stats)} output dates with data: "
              f"eff_dof median={np.median(dofs):.2f} (range {dofs.min():.2f}-{dofs.max():.2f}), "
              f"eff_bandwidth median={np.median(bws):.1f}d (range {bws.min():.1f}-{bws.max():.1f}d), "
              f"local_rmse median={np.median(rmses):.4f} (range {rmses.min():.4f}-{rmses.max():.4f}), "
              f"overall_rmse={overall_rmse:.4f}")
    else:
        print("Summary: no output date had any observation within the cutoff radius.")


def _stepwise_series(dates, pixel_cube, output_dates, wait_days,
                      above_noise, below_noise, rbf_sigma, rbf_cutoff):
    """Simulates what force_tsi_stepwise.py would have written for THIS
    pixel, one output date at a time: for output date `od`, only
    acquisition dates up to (od + wait_days) are used -- for BOTH
    despike() and rbf_interpolate() -- matching a date being "locked in"
    the moment it first becomes eligible (see
    force_tsi_stepwise.py's module docstring on eligibility and its
    wait_days-vs-cutoff-radius caveat). An `od` with no acquisition that
    far forward yet is left NaN, same as stepwise simply not having
    written it yet.

    Despiking is redone on each as-of subset rather than sliced out of
    one whole-archive despike, since force_tsi_stepwise.py always
    despikes whatever's currently on disk (itself a growing subset over
    time) -- a point's outlier status can genuinely change once more
    neighbors arrive later.

    Simplifying assumption: this assumes the stepwise pipeline runs
    immediately/continuously, so a date is locked in using data through
    EXACTLY (od + wait_days) -- the earliest-possible-write scenario. A
    real cron-style schedule that only runs every few days could see a
    little more data by its next actual run than this idealizes, but
    only ever as much as the SAME wait_days-vs-cutoff-radius caveat
    already covers (see recommended_wait_days()).

    :param pixel_cube: this pixel's raw series already sliced to
        (n_dates, 1, 1) -- despike()/rbf_interpolate() both expect a
        (dates, h, w) cube.
    """
    last_raw_date = max(dates)
    values = np.full(len(output_dates), np.nan, dtype="float32")
    for oi, od in enumerate(output_dates):
        cutoff = od + timedelta(days=wait_days)
        if cutoff > last_raw_date:
            continue  # not eligible yet -- stepwise wouldn't have written this
        keep_idx = [i for i, d in enumerate(dates) if d <= cutoff]
        dates_asof = [dates[i] for i in keep_idx]
        cube_asof = pixel_cube[keep_idx]
        cleaned_asof, _removed = despike(dates_asof, cube_asof, above_noise, below_noise)
        value = rbf_interpolate(dates_asof, cleaned_asof, [od], rbf_sigma, rbf_cutoff)
        values[oi] = value[0, 0, 0]
    return values


def plot_pixel(tss_path, row=None, col=None,
                date_range=None, doy_range=(1, 365), int_day=5,
                rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
                above_noise=3.0, below_noise=1.0,
                show_components=True, title=None, output_path=None,
                stepwise=False, wait_days=None, show_full_archive_comparison=True,
                print_effective_bandwidth=False):
    """Plot one pixel's raw series, despiking, and RBF ensemble.

    If row/col are given, only a tiny window is read from disk -- this
    is the fast path for testing on a pixel you already know. If left
    None, the whole tile is read once just to pick a reasonably dense
    pixel automatically (find_sample_pixel()) -- fine for a one-off look,
    but pass row/col directly for repeated testing.

    date_range defaults to the pixel's own first/last observed date if
    not given, so you don't have to know it up front.

    :param stepwise: if True, ALSO computes and plots a simulated
        force_tsi_stepwise.py curve for this pixel (see
        _stepwise_series()) -- lets you "play around" with wait_days on
        one pixel without running the batch stepwise pipeline at all.
        False (the default) is this function's original behavior,
        completely unchanged.
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
    :param print_effective_bandwidth: if True, prints (to stdout) the
        ACTUAL effective bandwidth/degrees of freedom the RBF ensemble
        achieves at every output date for this pixel, given the real
        observation density it's fed -- see effective_bandwidth_stats().
        Use this to compare, e.g., a sparse 2018 pixel against a dense
        2026 pixel: the same rbf_sigma doesn't produce the same realized
        smoothing on both, and this is how you'd measure that gap (and
        pick per-year sigmas that match a target effective bandwidth)
        instead of just eyeballing which curve looks smoother. Each line
        also prints local_rmse, the (in-sample) RMS gap between the
        fitted curve and the real nearby observations at that point.
    :return: (fig, (row, col), dates, cube, cleaned, removed) -- the
        underlying arrays are returned too, in case you want to inspect
        them further in the notebook.
    """
    if row is not None and col is not None:
        dates, cube, local_row, local_col = load_tss_pixel(tss_path, row, col, buffer=0)
    else:
        dates, cube, _, _ = load_tss(tss_path)
        row, col = find_sample_pixel(cube)
        local_row, local_col = row, col

    if date_range is None:
        date_range = (dates[0], dates[-1])

    cleaned, removed = despike(dates, cube, above_noise, below_noise)

    raw_series = cube[:, local_row, local_col]
    cleaned_series = cleaned[:, local_row, local_col]
    removed_series = removed[:, local_row, local_col]

    output_dates = output_dates_from_range(date_range, int_day, doy_range)

    if print_effective_bandwidth:
        _print_effective_bandwidth(dates, cleaned_series, output_dates, rbf_sigma, rbf_cutoff)

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
        ensemble_label = "Full-archive ensemble (reference)" if stepwise else "Ensemble (what run_tsi() writes)"
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


# Example (from a notebook):
#
#   from force_tsi_plot import plot_pixel
#
#   fig, (row, col), dates, cube, cleaned, removed = plot_pixel(
#       "NDVI_TSS.tif", row=700, col=210,
#       rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95,
#       above_noise=3.0, below_noise=1.0, int_day=5,
#   )
#   fig.show()
"""
merge_multiyear_tss.py

One-time preprocessing pass that concatenates each tile's per-year FORCE
TSA VI TSS tifs into a single, continuous, date-sorted multiband TSS tif
per (tile, VI) -- across as many years as you point it at.

WHY: force_tsi_batch.py / force_pull.py currently assume exactly one TSS
file per (tile, VI, year). But force_tsi.py's RBF interpolation
(rbf_sigma=(10, 20, 30, 50), rbf_cutoff=0.95) pulls real observations
from up to ~98 days on EITHER side of every output date (the widest
sigma's cutoff radius) -- a single calendar-year file structurally can't
supply that near a year boundary (e.g. a March output date can reach
back into the prior November). Concatenating years up front, once, lets
the existing pipeline see that context with ZERO changes to
force_tsi_batch.py/force_pull.py: each merged file is named the same way
FORCE's own TSA output already is (<start_date>-<end_date>_<vi>_TSS.tif),
so it drops straight into vi_pattern_template="{year}*_{vi}_TSS.tif" --
just point run_tsi_tiles()/run_tsa_workflow_tiled() at DST_ROOT below and
pass `year=` as the EARLIEST year you merged (see the printed hint at the
end of a run).

LAYOUT: each source year is a separate root directory (YEAR_ROOTS below),
each holding the standard X####_Y#### tile-folder layout
(force_tsi_batch.discover_tile_dirs()'s own assumption). Per-year VI
file-naming tokens can differ (e.g. 2018's "NDV" vs 2026's "NDVI" -- see
force_pull.VI_KEYS / force_tsi_batch.run_tsi_tiles()'s own vi_keys
examples) -- set that per year in VI_FILE_TOKENS; a year not listed there
is assumed to already use canonical VI names.

SAFETY CHECKS: every source year for a given (tile, VI) must share the
exact same grid (CRS, transform, width, height) and the same
dtype/nodata -- checked up front, and that (tile, VI) job fails loudly
(not silently) if they don't line up, since stacking misaligned rasters
would corrupt the merged file with no obvious symptom downstream. Every
band's description must also parse as an 8-digit YYYYMMDD date (matching
force_tsi.DATE_RE) -- same reason: force_tsi.load_tss() identifies every
band's date purely from this string, so a bad/missing description would
otherwise fail silently much later, deep inside despike()/rbf_interpolate().

MEMORY: bands are read and written ONE AT A TIME (no whole-tile cube ever
held in memory), so peak memory per job is roughly one band's worth of
pixel data regardless of how many years/bands are being merged.

Usage:
    Set YEAR_ROOTS / VI_FILE_TOKENS / DST_ROOT / WORKERS below, then:
        python scripts/merge_multiyear_tss.py

Re-run is safe: a destination file that already has the expected merged
band count is left untouched, so an interrupted run can just be
restarted.
"""

import os
import re
import glob
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import rasterio

# ---------------------------------------------------------------------
# Config -- edit these before running
# ---------------------------------------------------------------------

# One root directory per year, each holding the standard X####_Y####
# tile-folder layout. Add as many years as you have.
YEAR_ROOTS = {
    # 2018: r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level_3_2018_tiled",
    # 2024: r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss",
    2025: r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_testing",
    2026: r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_testing",
}

# Per-year {canonical_vi: file_token} -- only needed where a year's
# filenames use a different abbreviation than the canonical name. A year
# not listed here is assumed to use canonical names directly
# (file_token == canonical name).
VI_FILE_TOKENS = {
    2018: {"NDVI": "NDV", "EVI": "EVI", "NDMI": "NDM", "CIRE": "CRE", "CCI": "CCI"},
    2024: {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"},
    2025: {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"},
    2026: {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"},
}

VI_KEYS = ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]

# Matches force_tsi_batch.DEFAULT_VI_PATTERN_TEMPLATE -- how source files
# are located within each year's tile folders.
VI_PATTERN_TEMPLATE = "{year}*_{vi}_TSS.tif"

DST_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss_multiyear_testing"
WORKERS = 24

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TILE_REGEX = re.compile(r"^X\d{4}_Y\d{4}$")
DATE_RE = re.compile(r"(\d{8})")  # matches force_tsi.DATE_RE


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

def discover_tiles(root_dir):
    return sorted(
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name)) and TILE_REGEX.match(name)
    )


def _vi_token(year, vi):
    return VI_FILE_TOKENS.get(year, {}).get(vi, vi)


def _find_source(tile_dir, vi, year):
    pattern = VI_PATTERN_TEMPLATE.format(year=year, vi=_vi_token(year, vi))
    matches = sorted(glob.glob(os.path.join(tile_dir, pattern)))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple files matching {pattern!r} in {tile_dir}: {matches}")
    return matches[0]


def build_jobs():
    """One job per (tile, VI) that has a source file in at least one
    year -- (tile_id, vi, [(year, path), ...] sorted by year)."""
    years = sorted(YEAR_ROOTS)
    all_tiles = sorted(set().union(*(
        discover_tiles(YEAR_ROOTS[y]) for y in years if os.path.isdir(YEAR_ROOTS[y])
    )))
    log.info("Found %d distinct tile(s) across %d year root(s)", len(all_tiles), len(years))

    jobs = []
    for tile_id in all_tiles:
        for vi in VI_KEYS:
            sources = []
            for year in years:
                tile_dir = os.path.join(YEAR_ROOTS[year], tile_id)
                if not os.path.isdir(tile_dir):
                    continue
                path = _find_source(tile_dir, vi, year)
                if path is not None:
                    sources.append((year, path))
            if sources:
                jobs.append((tile_id, vi, sources))
            else:
                log.warning("%s/%s: no source file found in any year -- skipped", tile_id, vi)
    return jobs


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------

def _band_date(desc, path, band_idx):
    m = DATE_RE.match(desc) if desc else None
    if not m:
        raise ValueError(
            f"{path} band {band_idx}: description {desc!r} doesn't start with an "
            f"8-digit YYYYMMDD date -- force_tsi.py can't parse this band's date"
        )
    return datetime.strptime(m.group(1), "%Y%m%d").date()


def dst_is_valid(dst_path, expected_count):
    """Cheap check so a partially-completed run can be resumed safely."""
    if not os.path.exists(dst_path):
        return False
    try:
        with rasterio.open(dst_path) as ds:
            return ds.count == expected_count
    except rasterio.errors.RasterioIOError:
        return False


def merge_one(tile_id, vi, sources):
    """
    sources: [(year, path), ...] for this (tile, vi), at least one entry.

    Reads every band's description (header-only, cheap) from every
    source first to validate consistency and compute the merged date
    range -- the merged filename is built from that range, matching
    FORCE's own <start_date>-<end_date>_..._<vi>_TSS.tif convention (see
    module docstring), so this has to happen before deciding whether to
    skip an already-done job.

    :return: (dst_path, status) -- status is "written" or
        "skipped (already done)".
    :raises ValueError: on a grid/dtype/nodata mismatch across years, or
        a band description that doesn't parse as YYYYMMDD.
    """
    entries_meta = []  # (date, path, 1-based band_idx) -- no pixel data yet
    ref_profile = None

    for year, path in sources:
        with rasterio.open(path) as src:
            profile = (src.crs, src.transform, src.width, src.height, src.dtypes[0], src.nodata)
            if ref_profile is None:
                ref_profile = profile
            elif profile != ref_profile:
                raise ValueError(
                    f"{tile_id}/{vi}: grid/dtype/nodata mismatch -- "
                    f"{sources[0][1]} has {ref_profile}, {path} has {profile}"
                )
            for i, desc in enumerate(src.descriptions):
                date = _band_date(desc, path, i + 1)
                entries_meta.append((date, path, i + 1))

    entries_meta.sort(key=lambda e: e[0])
    start_date, end_date = entries_meta[0][0], entries_meta[-1][0]

    dst_dir = os.path.join(DST_ROOT, tile_id)
    dst_path = os.path.join(dst_dir, f"{start_date:%Y%m%d}-{end_date:%Y%m%d}_{vi}_TSS.tif")

    if dst_is_valid(dst_path, len(entries_meta)):
        return dst_path, "skipped (already done)"

    crs, transform, width, height, dtype, nodata = ref_profile
    os.makedirs(dst_dir, exist_ok=True)
    tmp_path = dst_path + ".tmp"

    open_srcs = {}
    try:
        with rasterio.open(
            tmp_path, "w", driver="GTiff", height=height, width=width,
            count=len(entries_meta), dtype=dtype, crs=crs, transform=transform,
            nodata=nodata, compress="ZSTD", predictor=2, tiled=True,
            blockxsize=256, blockysize=256, bigtiff="IF_SAFER",
        ) as dst:
            for out_idx, (date, path, band_idx) in enumerate(entries_meta, start=1):
                if path not in open_srcs:
                    open_srcs[path] = rasterio.open(path)
                src = open_srcs[path]
                dst.write(src.read(band_idx), out_idx)
                dst.set_band_description(out_idx, src.descriptions[band_idx - 1])
    finally:
        for s in open_srcs.values():
            s.close()

    os.replace(tmp_path, dst_path)
    return dst_path, "written"


def _merge_worker(job):
    tile_id, vi, sources = job
    try:
        dst_path, status = merge_one(tile_id, vi, sources)
        return tile_id, vi, status, dst_path, None
    except Exception as exc:  # surfaced in the parent via the future result
        return tile_id, vi, "failed", None, str(exc)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def main(dst_root=DST_ROOT, workers=WORKERS):
    jobs = build_jobs()
    if not jobs:
        raise SystemExit("No (tile, VI) jobs found -- check YEAR_ROOTS/VI_KEYS/VI_FILE_TOKENS")
    log.info("Queued %d (tile, VI) merge job(s) across %d year(s) (%d workers)",
              len(jobs), len(YEAR_ROOTS), workers)

    written = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_merge_worker, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            tile_id, vi, status, dst_path, error = future.result()
            if status == "written":
                written += 1
            elif status.startswith("skipped"):
                skipped += 1
            else:
                failed += 1
                log.error("FAILED: %s/%s -- %s", tile_id, vi, error)
            if i % 10 == 0 or i == len(jobs):
                log.info("Progress: %d/%d (written=%d, skipped=%d, failed=%d)",
                          i, len(jobs), written, skipped, failed)

    log.info("Done. written=%d skipped=%d failed=%d", written, skipped, failed)
    log.info(
        "To use the merged archive: point run_tsi_tiles()/run_tsa_workflow_tiled() "
        "at root_dir=%r with year=%d (the EARLIEST year merged) -- "
        "vi_pattern_template can stay the default \"{year}*_{vi}_TSS.tif\", "
        "and vi_keys can be a plain canonical list (e.g. %r) since merged "
        "filenames always use canonical VI names regardless of each "
        "source year's own naming.",
        dst_root, min(YEAR_ROOTS), VI_KEYS,
    )
    if failed:
        raise SystemExit(f"{failed} job(s) failed -- see log above")


if __name__ == "__main__":
    main()

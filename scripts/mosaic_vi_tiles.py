import os
import shutil

from src.disco_ch.force_tsi_batch import discover_tile_dirs, find_tile_vi_file
from src.disco_ch.normalize_vi_batch import (
    extract_vi_tile_subsets,
    extract_vi_tile_subsets_from_dated_files,
    find_tile_vi_dated_files,
    mosaic_vi_tiles_by_date,
    DEFAULT_NODATA_INT16,
)

"""
Welcome to the multi-temporal VI mosaic tool.

Mosaics a FORCE-tiled archive (one X####_Y#### folder per tile) into
AOI-wide rasters. Works against ANY of:
  - a multiband, per-band-dated VI archive (e.g.
    force_tsi_batch.run_tsi_tiles()'s TSI output, or
    normalize_vi_batch.normalize_vi_tiles()'s normalized output) --
    ONE OUTPUT FILE PER CALENDAR TIMESTEP, matching each band's own ISO
    date description;
  - a single-band file with no date information at all (e.g. a min/max
    raster -- stac_pull.save_minmax_rasters() writes one min tif and one
    max tif per VI, no band description) -- ONE OUTPUT FILE for that
    lone band, no per-date splitting/filtering involved; or
  - MANY per-date, single-band files per tile (e.g. force_pull.py's
    update_vi_min_max_tsa(export_normalized_vi_dir=...) output --
    <tile_id>/<vi>/<vi>_<date>_NORM.tif, date in the FILENAME, not a
    band description) -- set DATED_FILES_MODE = True for this one; it
    needs a different discovery/extraction path since
    find_tile_vi_file() (used by the other two cases) requires exactly
    one match per tile, not one per date. See
    normalize_vi_batch.find_tile_vi_dated_files() /
    extract_vi_tile_subsets_from_dated_files().
Which of the first two applies is detected automatically per file (see
normalize_vi_batch._resolve_band_labels()) -- just point VI_PATTERN_TEMPLATE
at whichever archive you want (e.g. "{vi}_min_{year}.tif" for min
rasters), and leave START_DATE/END_DATE as None if it's the undated case
(they're simply ignored, with a notice, if there's nothing to date-filter).
The third case (DATED_FILES_MODE) is a separate, explicit switch since it
isn't detectable from one file alone.

TWO PHASES, both parallel, each parallelized across the thing that's
actually independent at that phase:

  1. EXTRACT (extract_vi_tile_subsets(), parallel ACROSS TILES): pulls
     just the [START_DATE, END_DATE] band subset out of every tile, ONE
     combined read per tile (not one read per date), into small, local,
     tiled + band-interleaved files under SCRATCH_DIR. This is the phase
     that actually dominates wall time when tile_paths live on slow or
     networked storage -- EXTRACT_MAX_WORKERS lets N tiles' reads happen
     concurrently instead of queued one after another.

  2. MOSAIC (mosaic_vi_tiles_by_date(), parallel ACROSS DATES): merges
     the small extracted files per date into an AOI-wide raster, one
     band (one date) at a time -- memory stays bounded to a single
     timestep's worth of data, not the whole date range at once. Because
     phase 1 already made every tile small/local/tiled, this phase is
     normally fast even sequentially (MOSAIC_MAX_WORKERS=1); raise it
     only if profiling shows it still helps.

WHY NOT JUST PARALLELIZE THE MOSAIC ITSELF ACROSS DATES (as a first cut
of this script did)? Because mosaic_vi_tiles_by_date()'s own parallelism
only helps once MULTIPLE dates are in flight -- a single slow date still
reads every tile SERIALLY within whichever one process is handling it.
If tile_paths sit on a slow share, that first date "churns forever" no
matter how many workers are configured, since none of them are helping
with THAT read. Extracting per TILE in parallel (phase 1) is what
actually overlaps the slow part.

CLEANUP_SCRATCH (default True) deletes each VI's extracted subfolder
after that VI's mosaics are written -- the extracted files are a
disposable intermediate, not a second copy of the archive you want to
keep around. Set False to leave them (e.g. to remosaic a different date
window without re-paying the extract step, as long as it covers a
subset of what's already extracted).

No command-line arguments -- just edit the config below and run:
    python mosaic_vi_tiles.py
"""

# Root directory containing the X####_Y#### tile folders, each holding
# one multiband tif per VI.
ROOT_DIR = r"F:\cb_overflow\2026_09_02_FORCE\FORCE\DiscoCH_rbf_10_20_30\output\level5norm\2026"
# r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\minmax\2026" # "B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\level5_norm\2026"

# VIs to mosaic (looped one at a time -- band count/order only needs to
# match within one VI's own tile files, not across VIs), and the
# file-naming token each VI actually uses in this archive's filenames
# (only differs from the canonical name for some years -- see
# force_pull.VI_KEYS).
VI_LIST = ["EVI", "NDMI", "CIRE", "NDVI"]  # CCI
FILE_TOKENS = {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}
YEAR = 2026

# Set True for the per-date, single-band file layout (e.g.
# force_pull.py's export_normalized_vi_dir output --
# <tile_id>/<vi>/<vi>_<date>_NORM.tif, one file per date, per VI, per
# tile) instead of ONE multiband file per tile. See module docstring.
DATED_FILES_MODE = True

# Glob pattern matching each tile's file(s) within its tile folder.
#
# DATED_FILES_MODE = False (one multiband file per tile): pattern takes
# {year}/{vi} placeholders -- match whichever suffix your archive
# actually uses (normalize_vi_batch's default output suffix is "_NORM";
# plain "_TSI" for un-normalized RBF-interpolated output).
#   VI_PATTERN_TEMPLATE = "{vi}_{year}*_NORM.tif"
#   VI_PATTERN_TEMPLATE = "{vi}_min_{year}.tif"
#
# DATED_FILES_MODE = True (many per-date files per tile): pattern takes
# only {vi} (filled in twice -- once for the <vi> subfolder, once for
# the filename prefix), no {year} -- date filtering happens via
# START_DATE/END_DATE instead, parsed straight out of each matched
# filename. Matches force_pull.py's own <vi>/<vi>_<date>_NORM.tif
# convention; adjust if your archive's suffix/subfolder differs.
VI_PATTERN_TEMPLATE = "{vi}/{vi}_*_NORM.tif" if DATED_FILES_MODE else "{vi}_{year}*_NORM.tif"

# Local, fast-disk scratch directory for phase 1's extracted per-tile
# subsets -- put this on LOCAL disk even if ROOT_DIR is a network share.
SCRATCH_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\level5_norm\2026\temp"
CLEANUP_SCRATCH = True  # delete each VI's extracted subfolder once its mosaics are written

OUT_DIR = r"F:\cb_overflow\2026_09_02_FORCE\FORCE\DiscoCH_rbf_10_20_30\output\level5norm\2026\mosaic" # r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\mask\min\2026\mosaic"

# Optional 'YYYY-MM-DD' bounds restricting which timesteps get
# extracted/mosaicked -- None (default) processes every date found in
# the archive.
START_DATE = "2026-06-01"
END_DATE = "2026-09-02"

# nodata value written into gaps in the mosaic -- MUST match the source
# tiles' own dtype/convention: DEFAULT_NODATA_INT16 (-9999) for int16
# sources (e.g. normalize_vi_tiles()'s default output), or float('nan')
# for float32 sources -- min/max rasters (stac_pull.save_minmax_rasters())
# are float32, so use float('nan') when VI_PATTERN_TEMPLATE points at those.
NODATA_VALUE = DEFAULT_NODATA_INT16  # e.g. float('nan') for the min/max case

# CSV (Tile_ID column) restricting which tiles get processed. None
# (default) processes every tile folder found under ROOT_DIR, unfiltered.
TILES_CSV = None
TILES_COLUMN = "Tile_ID"
TILES_CSV_MODE = "include"

# Phase 1: tiles are fully independent, so this parallelizes cleanly --
# pick this based on how many concurrent reads your storage can actually
# sustain (a network share may saturate well before your CPU core count
# does; try a handful, e.g. 4-8, before assuming higher helps).
EXTRACT_MAX_WORKERS = 20

# Phase 2: > 1 mosaics several dates concurrently (separate processes).
# Usually fine to leave at 1 once phase 1 has made the inputs small and
# local -- raise it only if profiling shows it still helps.
MOSAIC_MAX_WORKERS = 1


def find_vi_tile_paths(root_dir, vi, file_token, year, vi_pattern_template,
                        tiles_csv, tiles_column, tiles_csv_mode):
    """Every tile's own file for one VI, across a FORCE-tiled archive --
    reuses discover_tile_dirs()/find_tile_vi_file() as-is, the same tile
    discovery force_tsi_batch.run_tsi_tiles()/normalize_vi_batch.
    normalize_vi_tiles() use. A tile missing this VI's file is skipped
    (printed), not fatal -- every other tile is still included."""
    tile_dirs = discover_tile_dirs(root_dir, tiles_csv, tiles_column, tiles_csv_mode)
    print(f"Found {len(tile_dirs)} tile(s) under {root_dir}")

    tile_paths = []
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        try:
            tile_paths.append(find_tile_vi_file(tile_dir, file_token, vi_pattern_template, year=year))
        except FileNotFoundError as e:
            print(f"  {tile_id}/{vi}: not found -- {e}")
    return tile_paths


def find_vi_tile_dated_files(root_dir, vi, file_token, vi_pattern_template, start_date, end_date,
                              tiles_csv, tiles_column, tiles_csv_mode):
    """DATED_FILES_MODE counterpart to find_vi_tile_paths(): every tile's
    per-date files for one VI (see normalize_vi_batch.find_tile_vi_dated_files()),
    already filtered to [start_date, end_date]. A tile with nothing found
    is included as an empty list (printed), not fatal -- every other tile
    is still processed; extract_vi_tile_subsets_from_dated_files() skips
    empty entries on its own."""
    tile_dirs = discover_tile_dirs(root_dir, tiles_csv, tiles_column, tiles_csv_mode)
    print(f"Found {len(tile_dirs)} tile(s) under {root_dir}")

    tile_dated_files = {}
    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        dated_files = find_tile_vi_dated_files(
            tile_dir, file_token, vi_pattern_template, start_date=start_date, end_date=end_date,
        )
        if not dated_files:
            print(f"  {tile_id}/{vi}: no files found in [{start_date}, {end_date}]")
        tile_dated_files[tile_id] = dated_files
    return tile_dated_files


if __name__ == "__main__":
    for vi in VI_LIST:
        print(f"\n=== {vi} ===")
        vi_scratch_dir = os.path.join(SCRATCH_DIR, vi)

        if DATED_FILES_MODE:
            tile_dated_files = find_vi_tile_dated_files(
                ROOT_DIR, vi, FILE_TOKENS[vi], VI_PATTERN_TEMPLATE, START_DATE, END_DATE,
                TILES_CSV, TILES_COLUMN, TILES_CSV_MODE,
            )
            n_files = sum(len(files) for files in tile_dated_files.values())
            if n_files == 0:
                print(f"  No files found for {vi} -- skipping.")
                continue

            print(f"  Extracting {n_files} file(s) across {len(tile_dated_files)} tile(s) into "
                  f"{vi_scratch_dir} (max_workers={EXTRACT_MAX_WORKERS})")
            extracted_paths, labels = extract_vi_tile_subsets_from_dated_files(
                tile_dated_files, vi_scratch_dir, max_workers=EXTRACT_MAX_WORKERS,
            )
        else:
            tile_paths = find_vi_tile_paths(
                ROOT_DIR, vi, FILE_TOKENS[vi], YEAR, VI_PATTERN_TEMPLATE,
                TILES_CSV, TILES_COLUMN, TILES_CSV_MODE,
            )
            if not tile_paths:
                print(f"  No tile files found for {vi} -- skipping.")
                continue

            print(f"  Extracting {len(tile_paths)} tile(s) into {vi_scratch_dir} "
                  f"(max_workers={EXTRACT_MAX_WORKERS})")
            extracted_paths, labels = extract_vi_tile_subsets(
                tile_paths, vi_scratch_dir,
                start_date=START_DATE, end_date=END_DATE,
                max_workers=EXTRACT_MAX_WORKERS,
            )

        # labels is [''] for a source with no per-band dates at all (e.g. a
        # min/max raster) -- just one lone band, nothing to range-print.
        if labels == [""]:
            print(f"  Extracted 1 band (no per-band dates -- e.g. a min/max raster)")
        else:
            print(f"  Extracted {len(labels)} date(s): {labels[0]}..{labels[-1]}")

        vi_out_dir = os.path.join(OUT_DIR, vi)
        written = mosaic_vi_tiles_by_date(
            extracted_paths, vi_out_dir, vi,
            nodata_value=NODATA_VALUE, max_workers=MOSAIC_MAX_WORKERS,
        )
        print(f"  Wrote {len(written)} mosaic file(s) to {vi_out_dir}")

        if CLEANUP_SCRATCH:
            shutil.rmtree(vi_scratch_dir, ignore_errors=True)
            print(f"  Cleaned up {vi_scratch_dir}")

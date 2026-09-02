"""
retile_2018_archive.py

One-time preprocessing pass over the FORCE level_3_2018 archive.

The 2018 FORCE TSS TIFs were written as untiled, row-strip GeoTIFFs
(Block=3000x300, i.e. one strip per 1/10 of the image height, LZW
compressed). The 2026 archive (level3_tss_v2) is properly tiled
(Block=256x256, ZSTD compressed). Because both archives use
INTERLEAVE=BAND, reading the 2018 files forces GDAL to decompress huge
row-strips per band into its block cache, which is what blows up memory
during RBF interpolation even for a single worker.

This script rewrites every VI TSS TIF under a source tile tree into a new
tile tree with the same directory layout, band count, dtype, nodata, and
band descriptions -- just re-blocked as 256x256 tiles and recompressed
with ZSTD (predictor=2, matching the 2026 archive). Originals are left
untouched; output goes to a separate --dst-root.

Usage:
    Set SRC_ROOT / DST_ROOT / WORKERS below, then just run:
        python scripts/retile_2018_archive.py

Re-run is safe: any destination file that already exists and opens
cleanly is skipped, so an interrupted run can just be restarted.
"""

import os
import re
import glob
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio

# ---------------------------------------------------------------------
# Config -- edit these before running
# ---------------------------------------------------------------------
SRC_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level_3_2018"
DST_ROOT = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level_3_2018_tiled"
WORKERS = 1
BLOCKSIZE = 256
COMPRESS = "ZSTD"
PREDICTOR = 2

# Recycle each worker process after this many files. Works around GDAL/libtiff
# internal state (not freed by our explicit closes) that degrades after a
# long-running process has opened many files in sequence, eventually causing
# spurious "Cannot open TIFF image" failures on perfectly healthy files.
MAX_TASKS_PER_CHILD = 20

# Backstop for whatever transient open failures slip through before a worker
# gets recycled.
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TILE_REGEX = re.compile(r"^X\d{4}_Y\d{4}$")
VI_TIF_GLOB = "*_TSS.tif"


def discover_tiles(root_dir):
    return [
        os.path.join(root_dir, name)
        for name in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, name)) and TILE_REGEX.match(name)
    ]


def _open_with_retry(path, mode="r", retries=RETRY_ATTEMPTS, backoff=RETRY_BACKOFF_SECONDS, **kwargs):
    """rasterio.open() with retries for transient GDAL open failures."""
    for attempt in range(1, retries + 1):
        try:
            return rasterio.open(path, mode, **kwargs)
        except rasterio.errors.RasterioIOError as exc:
            if attempt == retries:
                raise
            wait = backoff * attempt
            log.warning("Open failed (attempt %d/%d) for %s: %s -- retrying in %.1fs",
                        attempt, retries, path, exc, wait)
            time.sleep(wait)


def dst_is_valid(dst_path, expected_count):
    """Cheap check so a partially-completed run can be resumed safely."""
    if not os.path.exists(dst_path):
        return False
    try:
        with rasterio.open(dst_path) as ds:
            return ds.count == expected_count
    except rasterio.errors.RasterioIOError:
        return False


def retile_one(src_path, dst_path, blocksize=256, compress="ZSTD", predictor=2, overwrite=False):
    """
    Rewrites a single multiband TIF with 256x256 tiling + the given
    compression, preserving dtype/nodata/band descriptions/CRS/transform.
    Bands are read/written one at a time to keep peak memory to a single
    band's worth of data regardless of how many bands the file has.
    Writes to a .tmp path and atomically replaces the destination so a
    crash mid-file never leaves a corrupt "finished" output behind.
    """
    with _open_with_retry(src_path) as src:
        if dst_is_valid(dst_path, src.count):
            return dst_path, "skipped (already done)", []

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            compress=compress,
            predictor=predictor,
            interleave="BAND",
            bigtiff="IF_SAFER",
        )

        tmp_path = dst_path + ".tmp"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        patched_bands = []
        with _open_with_retry(tmp_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                try:
                    arr = src.read(band_idx)
                except rasterio.errors.RasterioIOError as exc:
                    patched_bands.append(band_idx)
                    log.warning(
                        "%s band %d unreadable (corrupt strip/tile) -- writing "
                        "nodata-filled band instead: %s", src_path, band_idx, exc
                    )
                    arr = np.full((src.height, src.width), src.nodata, dtype=src.dtypes[band_idx - 1])
                dst.write(arr, band_idx)
                desc = src.descriptions[band_idx - 1]
                if desc:
                    dst.set_band_description(band_idx, desc)

    os.replace(tmp_path, dst_path)
    if patched_bands:
        return dst_path, "converted", patched_bands
    return dst_path, "converted", []


def _retile_worker(args):
    src_path, dst_path, blocksize, compress, predictor = args
    try:
        _, status, patched_bands = retile_one(src_path, dst_path, blocksize, compress, predictor)
        return src_path, status, None, patched_bands
    except Exception as exc:  # surfaced in the parent via the future result
        return src_path, "failed", str(exc), []


def main(src_root=SRC_ROOT, dst_root=DST_ROOT, workers=WORKERS,
         blocksize=BLOCKSIZE, compress=COMPRESS, predictor=PREDICTOR,
         max_tasks_per_child=MAX_TASKS_PER_CHILD):
    tiles = discover_tiles(src_root)
    if not tiles:
        raise SystemExit(f"No X####_Y#### tile directories found under {src_root}")
    log.info("Found %d tile directories under %s", len(tiles), src_root)

    jobs = []
    for tile_dir in tiles:
        tile_name = os.path.basename(tile_dir)
        for src_path in sorted(glob.glob(os.path.join(tile_dir, VI_TIF_GLOB))):
            dst_path = os.path.join(dst_root, tile_name, os.path.basename(src_path))
            jobs.append((src_path, dst_path, blocksize, compress, predictor))

    log.info("Queued %d files for re-tiling (%d workers)", len(jobs), workers)

    converted = skipped = failed = 0
    patched_files = []
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=max_tasks_per_child) as executor:
        futures = {executor.submit(_retile_worker, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            src_path, status, error, patched_bands = future.result()
            if status == "converted":
                converted += 1
                if patched_bands:
                    patched_files.append((src_path, patched_bands))
            elif status.startswith("skipped"):
                skipped += 1
            else:
                failed += 1
                log.error("FAILED: %s -- %s", src_path, error)
            if i % 25 == 0 or i == len(jobs):
                log.info("Progress: %d/%d (converted=%d, skipped=%d, failed=%d)",
                          i, len(jobs), converted, skipped, failed)

    log.info("Done. converted=%d skipped=%d failed=%d", converted, skipped, failed)
    if patched_files:
        log.warning("%d file(s) had unreadable bands patched with nodata:", len(patched_files))
        for src_path, patched_bands in patched_files:
            log.warning("  %s -- bands %s", src_path, patched_bands)
    if failed:
        raise SystemExit(f"{failed} file(s) failed -- see log above")


if __name__ == "__main__":
    main()

"""
update_FORCE_tss.py

Copies only files whose FILENAME contains moniker from a source tree
(e.g. a network share) to a destination tree, preserving the relative
folder structure. Skips files that already exist at the destination
with the same size and modified time, so you can re-run this safely
if a large network copy gets interrupted.

No command-line args -- just edit the CONFIG section below and run:

    python sync_2026_files.py
"""

import os
import shutil
import time

# ─── CONFIG ──────────────────────────────────────────────────────────
SOURCE_DIR = r"\\speedy16-36\Data_23\FORCE\FORCE_Kingslide\level3_tss"
DEST_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss"
NAME_MATCH = "2026"      # substring to look for in filenames
DRY_RUN = False           # set to False once you've reviewed the preview
# ─────────────────────────────────────────────────────────────────────


def find_matching_files(source_dir, name_match):
    """Yield full paths of every file under source_dir whose name contains name_match."""
    for root, _dirs, files in os.walk(source_dir):
        for filename in files:
            if name_match in filename:
                yield os.path.join(root, filename)


def needs_copy(src_path, dst_path):
    """True if dst doesn't exist, or differs from src in size/mtime."""
    if not os.path.exists(dst_path):
        return True
    src_stat = os.stat(src_path)
    dst_stat = os.stat(dst_path)
    if src_stat.st_size != dst_stat.st_size:
        return True
    # allow a 2-second tolerance for filesystem mtime rounding differences
    if abs(src_stat.st_mtime - dst_stat.st_mtime) > 2:
        return True
    return False


def copy_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)  # copy2 preserves timestamps


def relative_dest_path(src_path, source_dir, dest_dir):
    rel = os.path.relpath(src_path, source_dir)
    return os.path.join(dest_dir, rel)


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    print(f"Scanning source: {SOURCE_DIR}")
    print(f"Matching filenames containing: '{NAME_MATCH}'")
    print(f"Destination: {DEST_DIR}")
    print(f"Dry run: {DRY_RUN}")
    print("-" * 60)

    to_copy = []
    already_synced = 0
    total_matched = 0

    for src_path in find_matching_files(SOURCE_DIR, NAME_MATCH):
        total_matched += 1
        dst_path = relative_dest_path(src_path, SOURCE_DIR, DEST_DIR)
        if needs_copy(src_path, dst_path):
            to_copy.append((src_path, dst_path))
        else:
            already_synced += 1

    total_bytes = sum(os.path.getsize(s) for s, _ in to_copy)

    print(f"Matched files:        {total_matched}")
    print(f"Already up to date:   {already_synced}")
    print(f"Need copying:         {len(to_copy)}  ({format_bytes(total_bytes)})")
    print("-" * 60)

    if not to_copy:
        print("Nothing to copy. Done.")
        return

    if DRY_RUN:
        print("DRY RUN -- files that would be copied:")
        for src, dst in to_copy:
            print(f"  {src}  ->  {dst}")
        print("\nSet DRY_RUN = False in the CONFIG section to actually copy.")
        return

    copied = 0
    copied_bytes = 0
    start = time.time()

    for i, (src, dst) in enumerate(to_copy, start=1):
        size = os.path.getsize(src)
        print(f"[{i}/{len(to_copy)}] Copying {src} -> {dst} ({format_bytes(size)})")
        try:
            copy_file(src, dst)
            copied += 1
            copied_bytes += size
        except Exception as e:
            print(f"  ERROR copying {src}: {e}")

    elapsed = time.time() - start
    print("-" * 60)
    print(f"Copied {copied}/{len(to_copy)} files ({format_bytes(copied_bytes)}) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
data_transfer.py

Copies .tif/.tiff files whose filename contains a given year from a source
directory (e.g. a network share) to a destination folder on disk, preserving
the subfolder structure.

"""

import os
import shutil
import concurrent.futures
import time

# ======================= CONFIG - edit these =======================

SOURCE_DIR = r"\\speedy16-36\Data_23\FORCE\FORCE_Kingslide\level2\tsa\real_values_flagged"   # network/source directory to scan
DEST_DIR = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level_3_2018"               # where to copy matching files to

YEARS = ["2018"]                          # years (or any substrings) to match in filename
EXTENSIONS = [".tif"]            # file extensions to include
REQUIRED_STRINGS = ["NDV", "CCI", "EVI", "CRE", "NDM"]  # filename must contain at least one of these

WORKERS = 20         # number of concurrent copy threads (network I/O benefits from >1)
DRY_RUN = False      # set True to just preview matches without copying anything
OVERWRITE = False    # set True to re-copy even if a matching file already exists at dest

# =====================================================================


def matches(filename, years, exts, required_strings):
    name_lower = filename.lower()
    if not any(name_lower.endswith(e.lower()) for e in exts):
        return False
    if not any(str(y) in filename for y in years):
        return False
    if required_strings and not any(s in filename for s in required_strings):
        return False
    return True


def iter_matching_files(source_root, years, exts, required_strings):
    """Walk source_root with os.scandir and yield (full_path, relative_path) for matches."""
    stack = [source_root]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            if matches(entry.name, years, exts, required_strings):
                                rel_path = os.path.relpath(entry.path, source_root)
                                yield entry.path, rel_path
                    except OSError as e:
                        print(f"  [warn] could not stat {entry.path}: {e}")
        except OSError as e:
            print(f"  [warn] could not list {current_dir}: {e}")


def needs_copy(src_path, dst_path):
    """Skip copy if dest already exists with same size and mtime (cheap resumability)."""
    if not os.path.exists(dst_path):
        return True
    try:
        s_stat = os.stat(src_path)
        d_stat = os.stat(dst_path)
        same_size = s_stat.st_size == d_stat.st_size
        same_mtime = int(s_stat.st_mtime) == int(d_stat.st_mtime)
        return not (same_size and same_mtime)
    except OSError:
        return True


def copy_one(src_path, dst_path, dry_run, overwrite):
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if not overwrite and not needs_copy(src_path, dst_path):
            return ("skip", src_path, None)
        if dry_run:
            return ("would_copy", src_path, None)
        shutil.copy2(src_path, dst_path)
        return ("copy", src_path, None)
    except Exception as e:
        return ("error", src_path, str(e))


def main():
    source_root = SOURCE_DIR
    dest_root = os.path.abspath(DEST_DIR)

    if not os.path.exists(source_root):
        print(f"Source directory not found: {source_root}")
        return

    if not DRY_RUN:
        os.makedirs(dest_root, exist_ok=True)

    print(f"Scanning: {source_root}")
    print(f"Matching years: {YEARS}  extensions: {EXTENSIONS}  required strings: {REQUIRED_STRINGS}")
    print(f"Destination: {dest_root}{' (dry run, no files will be copied)' if DRY_RUN else ''}\n")

    start = time.time()
    counts = {"copy": 0, "would_copy": 0, "skip": 0, "error": 0}
    total_found = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = []
        for src_path, rel_path in iter_matching_files(source_root, YEARS, EXTENSIONS, REQUIRED_STRINGS):
            total_found += 1
            dst_path = os.path.join(dest_root, rel_path)
            futures.append(pool.submit(copy_one, src_path, dst_path, DRY_RUN, OVERWRITE))

        for fut in concurrent.futures.as_completed(futures):
            status, path, err = fut.result()
            counts[status] += 1
            if status == "error":
                print(f"  [error] {path}: {err}")
            elif status in ("copy", "would_copy"):
                label = "[dry-run] would copy" if status == "would_copy" else "[copied]"
                print(f"  {label} {path}")

    elapsed = time.time() - start
    print("\nDone.")
    print(f"  Matched files found : {total_found}")
    print(f"  Copied              : {counts['copy']}")
    print(f"  Would copy (dry run): {counts['would_copy']}")
    print(f"  Skipped (up to date): {counts['skip']}")
    print(f"  Errors              : {counts['error']}")
    print(f"  Elapsed             : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
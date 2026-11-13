"""Summarize audio file lengths in a folder (reads headers only).

Prints count, median, 95th percentile, max, and how many files are over a
limit, optionally grouped by a regex on the path (e.g. the task name).

Usage:
  python scripts/clip_lengths.py FOLDER
  python scripts/clip_lengths.py FOLDER --group 'task-([A-Za-z]+)'   # e.g. SPROUT tasks
  python scripts/clip_lengths.py FOLDER --exclude derivatives --exclude pipeline_processing
"""
import argparse
import collections
import glob
import os
import re

import soundfile as sf

EXTS = (".wav", ".flac", ".mp3", ".m4a")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--group", help="regex with one group; files are grouped by its match")
    ap.add_argument("--exclude", action="append", default=[], help="skip paths containing this text (repeatable)")
    ap.add_argument("--over", type=float, default=30.0, help="count files longer than this many seconds")
    args = ap.parse_args()

    groups = collections.defaultdict(list)
    unreadable = 0
    for path in glob.iglob(os.path.join(args.folder, "**", "*"), recursive=True):
        if not path.lower().endswith(EXTS) or any(x in path for x in args.exclude):
            continue
        key = "all"
        if args.group:
            m = re.search(args.group, path)
            key = m.group(1) if m else "?"
        try:
            groups[key].append(sf.info(path).duration)
        except Exception:
            unreadable += 1

    for key in sorted(groups):
        d = sorted(groups[key])
        n = len(d)
        print(f"{key:20} n={n:<7} median={d[n // 2]:.1f}s  p95={d[int(n * 0.95)]:.1f}s  "
              f"max={d[-1]:.1f}s  >{args.over:g}s={sum(x > args.over for x in d)}")
    if unreadable:
        print(f"unreadable files: {unreadable}")


if __name__ == "__main__":
    main()

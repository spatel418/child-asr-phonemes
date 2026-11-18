"""Split SEED speakers in two halves by speaker: one half may be used (audio only, no labels) in mid-training,
the other half stays a clean test set. Decided 2026-11-16 (MIDTRAINING.md).

    $PY -I scripts/seed_split.py

- Sources: raw/SEED/{auburn,cincinnati}/{children,adults}. pipeline_processing/ repeats files and is skipped.
- Speaker = study code + speaker number from the start of each file name (some folders are named after
  subtests). One speaker can appear under more than one full code (e.g. D vs N), so this unit keeps all of
  them together.
- Halves are drawn within site x {child SSD, child typical, adult} (seed 0), alternating after a shuffle.
- SEED sites (Auburn, Cincinnati) are not SPROUT sites, so no SEED child is in SPROUT.
Writes <data>/splits/seed/{midtrain,test}.txt (one speaker unit per line) + summary.json (counts only), read-only.
"""
import collections
import json
import os
import random
import re
import stat
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import OUT  # noqa: E402

DATA = os.path.dirname(OUT.rstrip("/"))
SEED = os.path.join(DATA, "raw", "SEED")
GROUPS = {("children", "child_disordered_speech"): "child_SSD", ("children", "child_nondisordered_speech"): "child_typical",
          ("adults", "adult_nondisordered_speech"): "adult"}


FIX = {"U101": "UC101"}  # one Cincinnati folder spells the study code without the C


def unit_of(name, dirpath):
    """Study code + speaker number, e.g. 2XY100-07_rest.wav -> XY100-07 (made-up name). The leading digit is dropped
    (some files omit it). A file without a code (one Cincinnati vowel file) takes its speaker folder's code."""
    for cand in [name] + dirpath.split(os.sep)[::-1]:
        m = re.match(r"^\d?([A-Z]+\d+)-0*(\d+)", cand)
        if m:
            return f"{FIX.get(m.group(1), m.group(1))}-{m.group(2)}"
    raise SystemExit(f"unparsed SEED speaker code: {name}")


def main():
    hours = collections.Counter()
    units = collections.defaultdict(set)   # (site, group) -> units
    group_of = {}
    for site in ("auburn", "cincinnati"):
        for (who, sub), g in GROUPS.items():
            root = os.path.join(SEED, site, who, sub)
            if not os.path.isdir(root):
                continue
            for dp, _, fs in os.walk(root):
                for f in fs:
                    if not f.endswith(".wav"):
                        continue
                    u = unit_of(f, dp)
                    if group_of.setdefault(u, g) != g:
                        raise SystemExit(f"speaker unit in two groups: {u}")
                    units[(site, g)].add(u)
                    hours[(site, g, u)] += sf.info(os.path.join(dp, f)).duration / 3600
    rng = random.Random(0)
    half = {}
    for key in sorted(units):
        us = sorted(units[key])
        rng.shuffle(us)
        for j, u in enumerate(us):
            half[u] = "midtrain" if j % 2 == 0 else "test"
    out = os.path.join(DATA, "splits", "seed")
    os.makedirs(out, exist_ok=True)
    summ = {"seed": 0, "unit": "site + speaker number", "cells": {}}
    for key in sorted(units):
        c = summ["cells"].setdefault(f"{key[0]} | {key[1]}", {})
        for h in ("midtrain", "test"):
            us = [u for u in units[key] if half[u] == h]
            c[h] = {"speakers": len(us), "hours": round(sum(hours[(key[0], key[1], u)] for u in us), 2)}
    for h in ("midtrain", "test"):
        p = os.path.join(out, f"{h}.txt")
        if os.path.exists(p):
            os.chmod(p, 0o640)
        open(p, "w").write("".join(f"{u}\n" for u in sorted(u for u in half if half[u] == h)))
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP)
    p = os.path.join(out, "summary.json")
    if os.path.exists(p):
        os.chmod(p, 0o640)
    json.dump(summ, open(p, "w"), indent=1)
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

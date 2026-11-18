"""Mid-training audio from CHAT sessions of TRAIN children only (MIDTRAINING.md). No labels are used.

    $PY -I scripts/midtrain_turns.py            # writes the manifest and prints hours

Two kinds of timed turns, from every .cha with audio whose child is in splits/phonetic/v2/train.txt:
  adult:          any non-CHI speaker turn (mother, examiner, ...), 0.5-20 s, not overlapping a CHI turn
  child_nolabel:  CHI turns that have no %pho line (so not in the phonetic manifest), 0.5-20 s,
                  child age >= 24 months (Providence under 2 is skipped, decided 2026-11-16)
Sessions of val / test children are never read past the child-ID check. The sealed list is checked too.
Writes <data>/manifests/midtrain/turns.jsonl (mode 640). Rows carry the session child's ID for the leak audit.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import (BULLET, CHAT_ROOT, CHI_ID, MEDIA, MEDIA_EXT, OUT, RAW, SEALED,  # noqa: E402
                            _child_from_path, age_months, clip_id)

CORPORA = ["ACAD", "Cummings", "PERCEPT-R", "PERCEPT-GFTA", "Penney", "Providence"]
MIN_S, MAX_S = 0.5, 20.0
SPLITS = os.path.join(os.path.dirname(OUT.rstrip("/")), "splits", "phonetic", "v2")


def train_children():
    kids = {l.split()[0] for l in open(os.path.join(SPLITS, "train.txt")) if l.strip()}
    sealed = set(open(SEALED).read().split())
    bad = {k for k in kids if k.split("_", 1)[-1] in sealed}
    if bad:
        raise SystemExit(f"{len(bad)} sealed children in train.txt")
    return kids


def turns(corpus, kids):
    root = os.path.join(CHAT_ROOT, corpus)
    for cha in sorted(glob.glob(os.path.join(root, "**", "*.cha"), recursive=True)):
        rel = os.path.relpath(cha, root)
        child = _child_from_path(corpus, rel)
        child_id = f"PERCEPT_{child}" if corpus.startswith("PERCEPT") else f"{corpus}_{child}"
        if child_id not in kids:
            continue
        text = open(cha, encoding="utf-8", errors="replace").read()
        m = MEDIA.search(text)
        audio = None
        if m:
            for ext in MEDIA_EXT:
                cand = os.path.join(RAW, corpus, "media", os.path.dirname(rel), m.group(1) + ext)
                if os.path.exists(cand):
                    audio = cand
                    break
        if audio is None:
            continue
        idm = CHI_ID.search(text)
        age = age_months(idm.group(3)) if idm else None
        blocks = []
        for block in re.split(r"\n(?=\*)", text):
            sm = re.match(r"^\*([A-Z0-9]+):", block)
            t = BULLET.search(block.split("\n%")[0]) if sm else None
            if t:
                blocks.append((sm.group(1), int(t.group(1)) / 1000, int(t.group(2)) / 1000, "%pho:" in block))
        chi = sorted((s, e) for sp, s, e, _ in blocks if sp == "CHI")
        for sp, s, e, has_pho in blocks:
            d = e - s
            if not MIN_S <= d <= MAX_S:
                continue
            if sp == "CHI":
                if has_pho or age is None or age < 24:
                    continue
                kind = "child_nolabel"
            else:
                if any(cs < e and ce > s for cs, ce in chi):
                    continue
                kind = "adult"
            yield {"clip_id": clip_id(corpus, rel, s, e, "midtrain"), "corpus": corpus, "kind": kind, "speaker": sp,
                   "session_child_id": child_id, "age_months": age, "source_path": audio,
                   "start": s, "end": e, "duration": round(d, 3)}


def main():
    kids = train_children()
    out_dir = os.path.join(OUT, "midtrain")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "turns.jsonl")
    hours = collections.Counter()
    seen = set()
    with open(path, "w") as f:
        for c in CORPORA:
            for r in turns(c, kids):
                if r["clip_id"] in seen:
                    continue
                seen.add(r["clip_id"])
                f.write(json.dumps(r) + "\n")
                hours[(c, r["kind"])] += r["duration"] / 3600
    os.chmod(path, 0o640)
    for k in sorted(hours):
        print(f"{k[0]:14s} {k[1]:14s} {hours[k]:7.1f} h")
    for kind in ("adult", "child_nolabel"):
        print(f"total {kind}: {sum(v for k, v in hours.items() if k[1] == kind):.1f} h")


if __name__ == "__main__":
    main()

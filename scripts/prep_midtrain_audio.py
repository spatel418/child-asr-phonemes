"""Cut mid-training audio to 16 kHz mono FLAC (MIDTRAINING.md). No labels. Plain only (no denoising, like the
plain SFT input). Saved at the source level; training rescales each clip to unit variance anyway.

    $PY -I scripts/prep_midtrain_audio.py --shard 0 --nshards 32     # one Slurm array task

Inputs:
  manifests/midtrain/turns.jsonl  (adult + unlabeled child turns from train children's sessions)
  SEED speakers in splits/seed/midtrain.txt (whole files; files over 20 s are cut into 15 s windows)
Writes processed/midtrain/<source>/<clip_id>.flac and processed/midtrain_index/shard-XXXX.jsonl.
Re-running skips clips already written.
"""
import argparse
import collections
import json
import os
import sys
import zlib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import OUT, clip_id  # noqa: E402
from prep_audio import PROCESSED, load_mono, to16k  # noqa: E402
from seed_split import SEED, GROUPS, unit_of  # noqa: E402

DATA = os.path.dirname(OUT.rstrip("/"))
WIN, MAX_S, MIN_S = 15.0, 20.0, 0.5


def seed_rows():
    keep = set(open(os.path.join(DATA, "splits", "seed", "midtrain.txt")).read().split())
    for site in ("auburn", "cincinnati"):
        for (who, sub), g in GROUPS.items():
            for dp, _, fs in os.walk(os.path.join(SEED, site, who, sub)):
                for f in sorted(fs):
                    if not f.endswith(".wav"):
                        continue
                    u = unit_of(f, dp)
                    if u not in keep:
                        continue
                    p = os.path.join(dp, f)
                    d = sf.info(p).duration
                    spans = [(None, None)] if d <= MAX_S else [(s, min(s + WIN, d)) for s in np.arange(0, d - MIN_S, WIN)]
                    for s, e in spans:
                        yield {"clip_id": clip_id("SEED", p, s, e), "corpus": "SEED",
                               "kind": "adult" if g == "adult" else "child_nolabel", "speaker_unit": u,
                               "group": g, "source_path": p, "start": s, "end": e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only this many source files (testing)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "midtrain", "turns.jsonl"))] + list(seed_rows())
    by_src = collections.defaultdict(list)
    for r in rows:
        if zlib.crc32(r["source_path"].encode()) % a.nshards == a.shard:
            by_src[r["source_path"]].append(r)
    srcs = sorted(by_src)[: a.limit or None]
    idx_dir = os.path.join(PROCESSED, "midtrain_index")
    os.makedirs(idx_dir, exist_ok=True)
    idx_path = os.path.join(idx_dir, f"shard-{a.shard:04d}.jsonl")
    done = {json.loads(l)["clip_id"] for l in open(idx_path)} if os.path.exists(idx_path) else set()
    n_new = n_fail = 0
    with open(idx_path, "a") as idx:
        for i, src in enumerate(srcs):
            todo = [r for r in by_src[src] if r["clip_id"] not in done]
            if not todo:
                continue
            try:
                y, sr = load_mono(src)
            except Exception as e:
                print(f"FAIL {src}: {type(e).__name__}: {e}", flush=True)
                n_fail += len(todo)
                continue
            for r in todo:
                s = 0 if r["start"] is None else int(round(r["start"] * sr))
                e = len(y) if r["end"] is None else min(len(y), int(round(r["end"] * sr)))
                if e - s < int(MIN_S * sr):
                    n_fail += 1
                    continue
                clip = np.clip(to16k(y[s:e], sr), -1.0, 1.0)
                if not np.any(clip):
                    n_fail += 1
                    continue
                p = os.path.join(PROCESSED, "midtrain", r["corpus"], f"{r['clip_id']}.flac")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, clip, 16000, subtype="PCM_16")
                out = {k: r[k] for k in ("clip_id", "corpus", "kind") if k in r}
                out.update({"who": r.get("session_child_id") or r.get("speaker_unit"), "path": p,
                            "duration": round(len(clip) / 16000, 3)})
                idx.write(json.dumps(out) + "\n")
                n_new += 1
            idx.flush()
            if i % 50 == 0:
                print(f"[{i + 1}/{len(srcs)}] files, {n_new} clips", flush=True)
    print(f"shard {a.shard}: {len(srcs)} files, {n_new} new clips, {n_fail} skipped")


if __name__ == "__main__":
    main()

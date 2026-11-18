"""Screen child clips for adult voices by pitch (librosa pYIN, 70-600 Hz).
Per clip: median F0 over voiced frames, and the share of voiced frames under 200 Hz (adult range).
A clip is flagged when >= 25% of its voiced frames are under 200 Hz and it has >= 0.3 s of voicing.
Train + val only (test splits are never read). Output stays on Quest:
  <out>/pitch.jsonl  (audio path, corpus, task, split, group, numbers)  mode 600
  <out>/summary.json (counts only)
    $PY -I scripts/adult_pitch_scan.py --corpus SPROUT --corpus ACAD --out RUNS/adult-scan
"""
import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402

LOW_HZ, FLAG_SHARE, MIN_VOICED_S = 200.0, 0.25, 0.3
HOP = 160  # 10 ms at 16 kHz


def one(path):
    import librosa
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if len(y) < sr * 0.15:
        return None
    f0, vflag, _ = librosa.pyin(y, fmin=70, fmax=600, sr=sr, frame_length=1024, hop_length=HOP)
    v = f0[vflag & np.isfinite(f0)]
    if len(v) == 0:
        return {"voiced_s": 0.0, "f0_median": None, "low_share": None}
    return {"voiced_s": round(len(v) * HOP / sr, 3), "f0_median": round(float(np.median(v)), 1),
            "low_share": round(float((v < LOW_HZ).mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open("RUNS/sft-r16-plain-word-s0/config.json"))
    items = T.load_items(argparse.Namespace(**cfg))[0]
    task = {}
    path_of = {}
    for p in sorted(os.listdir(os.path.join(T.DATA, "processed", "index"))):
        for line in open(os.path.join(T.DATA, "processed", "index", p)):
            r = json.loads(line)
            path_of[r["clip_id"]] = r["plain_path"]
    for c in a.corpus:
        for line in open(os.path.join(T.MANIFESTS, "phonetic", f"{c}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["clip_id"] in path_of:
                task[path_of[r["clip_id"]]] = r.get("task")
    todo = [(s, it) for s in ("train", "val") for it in items[s] if it["corpus"] in a.corpus]
    with Pool(a.procs) as pool:
        res = pool.map(one, [it["path"] for _, it in todo], chunksize=16)
    out = os.path.join(a.out, "pitch.jsonl")
    summ = collections.defaultdict(lambda: collections.Counter())
    with open(out, "w") as f:
        for (s, it), r in zip(todo, res):
            if r is None:
                continue
            flag = r["low_share"] is not None and r["voiced_s"] >= MIN_VOICED_S and r["low_share"] >= FLAG_SHARE
            row = {"path": it["path"], "corpus": it["corpus"], "task": task.get(it["path"]), "split": s,
                   "group": it["group"], "dur": it["dur"], **r, "flag_adult": flag}
            f.write(json.dumps(row) + "\n")
            k = f"{it['corpus']}|{task.get(it['path'])}|{s}"
            summ[k]["clips"] += 1
            summ[k]["flagged"] += int(flag)
            summ[k]["hours"] += it["dur"] / 3600
            summ[k]["flagged_hours"] += it["dur"] / 3600 * flag
    os.chmod(out, 0o600)
    summary = {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sorted(summ.items())}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"{'corpus|task|split':34s}{'clips':>7s}{'flagged':>8s}{'share':>7s}{'flag_h':>7s}")
    for k, v in summary.items():
        print(f"{k:34s}{int(v['clips']):7d}{int(v['flagged']):8d}{v['flagged']/max(v['clips'],1):7.1%}{v['flagged_hours']:7.2f}")


if __name__ == "__main__":
    main()

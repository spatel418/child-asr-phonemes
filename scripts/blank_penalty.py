"""Decode-time blank penalty: subtract b from the blank logit before greedy decoding. No retraining.
Our model deletes far more than it inserts (val del 0.12 vs ins 0.03; SPROUT output 12% too short).
To avoid tuning on the clips we score: val children are split in two halves (fixed seed, within each
corpus); b is picked on half A (select score) and reported on half B. Val split only.

    $PY -I scripts/blank_penalty.py RUNS/sft-r16-plain-word-s0
Writes <run>/blank_penalty.json.
"""
import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402

PENALTIES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def ops(r, y):
    s = d = i = 0
    for o in Levenshtein.editops(r, y):
        s += o.tag == "replace"
        d += o.tag == "delete"
        i += o.tag == "insert"
    return s, d, i


def summarize(acc):
    out = {}
    for key, (r, s, d, i) in acc.items():
        out[key] = {"per": (s + d + i) / max(r, 1), "sub": s / max(r, 1), "del": d / max(r, 1), "ins": i / max(r, 1)}
    tags = [t for t in T.SCORE_TAGS if t in out]
    out["select_score"] = float(np.mean([out[t]["per"] for t in tags]))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _, cfg, _ = G.load_sft(a.run_dir, device)
    model.eval()
    val = T.load_items(argparse.Namespace(**cfg))[0]["val"]
    by_corpus = collections.defaultdict(set)
    for it in val:
        by_corpus[it["corpus"]].add(it["child"])
    rng = random.Random(0)
    half = {}
    for c in sorted(by_corpus):
        kids = sorted(by_corpus[c])
        rng.shuffle(kids)
        for j, k in enumerate(kids):
            half[k] = "A" if j % 2 == 0 else "B"
    acc = {h: {b: collections.defaultdict(lambda: np.zeros(4)) for b in PENALTIES} for h in ("A", "B", "all")}
    ds = T.Clips(val)
    for bt in G.batches_by_seconds(range(len(val)), val, 200.0):
        x, mask = G.load_batch(ds, bt, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, n = G.frame_logits(model, x, mask)
        logits = logits.float()
        for b in PENALTIES:
            lg = logits.clone()
            lg[..., 0] -= b
            hyps, _ = T.greedy(lg, n.tolist())
            for k, i in enumerate(bt):
                it = val[i]
                o = ops(it["labels"], hyps[k])
                for h in (half[it["child"]], "all"):
                    for key in ("all", it["tag"], f"corpus_{it['corpus']}"):
                        acc[h][b][key] += (len(it["labels"]), *o)
    res = {h: {str(b): summarize(acc[h][b]) for b in PENALTIES} for h in acc}
    best = min(PENALTIES, key=lambda b: res["A"][str(b)]["select_score"])
    res["picked_on_A"] = best
    res["children"] = {h: sum(v == h for v in half.values()) for h in ("A", "B")}
    json.dump(res, open(os.path.join(a.run_dir, "blank_penalty.json"), "w"), indent=1)
    print(f"picked b={best} on half A ({res['children']['A']} children); report on half B ({res['children']['B']} children)")
    print(f"{'b':>5s} {'A sel':>7s} {'B sel':>7s} {'B all':>7s} {'B del':>6s} {'B ins':>6s} {'all':>6s} "
          f"{'SPROUT':>7s} {'Preston':>8s} {'Provid':>7s}")
    for b in PENALTIES:
        A, B, L = res["A"][str(b)], res["B"][str(b)], res["all"][str(b)]
        print(f"{b:5.2f} {A['select_score']:7.4f} {B['select_score']:7.4f} {B['all']['per']:7.4f} {B['all']['del']:6.3f} "
              f"{B['all']['ins']:6.3f} {L['all']['per']:6.3f} {L['corpus_SPROUT']['per']:7.3f} "
              f"{L['corpus_Preston']['per']:8.3f} {L['corpus_Providence']['per']:7.3f}")


if __name__ == "__main__":
    main()

"""Consensus (MBR) decoding: no training. Val split only.

For each clip: greedy + k samples at temperature T. Candidates = greedy + the distinct samples. Pick the candidate
with the smallest total edit distance to all k samples (the model's own expected PER estimate); ties go to greedy.
Settings: T in {0.7, 1.0}, k in {16, 32}. To avoid tuning on the scored clips, val children are split in two
halves within each corpus (seed 0, same split as blank_penalty.py): the setting is picked on half A (select
score) and reported on half B.

    $PY -I scripts/mbr_decode.py --sft RUNS/sft-r16-plain-word-s0-prov2 [--rl RUNS/grpo-m1-prov2] --out RUNS/mbr-prov2
Writes <out>/mbr.json (numbers only).
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
import blank_penalty as B  # noqa: E402
import rl_diagnose as D  # noqa: E402
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402

TEMPS, KS = [0.7, 1.0], [16, 32]


def mbr(cands, samples):
    best, best_cost = 0, None
    for j, c in enumerate(cands):
        cost = sum(Levenshtein.distance(c, s) for s in samples)
        if best_cost is None or cost < best_cost:
            best, best_cost = j, cost
    return cands[best]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--rl", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    model, cfg = D.load(a.sft, a.rl, device)
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
    settings = ["greedy"] + [f"T{t}_k{k}" for t in TEMPS for k in KS]
    acc = {h: {s: collections.defaultdict(lambda: np.zeros(4)) for s in settings} for h in ("A", "B", "all")}
    gen = torch.Generator(device=device).manual_seed(a.seed)
    ds = T.Clips(val)
    for bt in G.batches_by_seconds(range(len(val)), val, 200.0):
        x, mask = G.load_batch(ds, bt, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, n = G.frame_logits(model, x, mask)
        logits = logits.float()
        g, _ = T.greedy(logits, n.tolist())
        picks = {"greedy": g}
        for t in TEMPS:
            samp = G.collapse(G.sample_paths(logits, n, max(KS), t, generator=gen))
            for k in KS:
                out = []
                for c in range(len(bt)):
                    ss = samp[c][:k]
                    cands = [list(g[c])] + [list(h) for h in dict.fromkeys(tuple(h) for h in ss) if list(h) != list(g[c])]
                    out.append(mbr(cands, ss))
                picks[f"T{t}_k{k}"] = out
        for s, hyps in picks.items():
            for c, i in enumerate(bt):
                it = val[i]
                o = B.ops(it["labels"], hyps[c])
                for h in (half[it["child"]], "all"):
                    for key in ("all", it["tag"], f"corpus_{it['corpus']}"):
                        acc[h][s][key] += (len(it["labels"]), *o)
    res = {h: {s: B.summarize(acc[h][s]) for s in settings} for h in acc}
    best = min(settings[1:], key=lambda s: res["A"][s]["select_score"])
    res["picked_on_A"] = best
    res["children"] = {h: sum(v == h for v in half.values()) for h in ("A", "B")}
    res["model"] = {"sft": a.sft, "rl": a.rl}
    json.dump(res, open(os.path.join(a.out, "mbr.json"), "w"), indent=1)
    print(f"picked {best} on half A; half B: greedy {res['B']['greedy']['select_score']:.4f} -> "
          f"{res['B'][best]['select_score']:.4f}")
    print(f"{'setting':>10s} {'A sel':>7s} {'B sel':>7s} {'all':>6s} {'del':>6s} {'ins':>6s} {'SPROUT':>7s} {'Preston':>8s} {'Provid':>7s}")
    for s in settings:
        A_, B_, L = res["A"][s], res["B"][s], res["all"][s]
        print(f"{s:>10s} {A_['select_score']:7.4f} {B_['select_score']:7.4f} {L['all']['per']:6.3f} {L['all']['del']:6.3f} "
              f"{L['all']['ins']:6.3f} {L['corpus_SPROUT']['per']:7.3f} {L['corpus_Preston']['per']:8.3f} "
              f"{L['corpus_Providence']['per']:7.3f}")


if __name__ == "__main__":
    main()

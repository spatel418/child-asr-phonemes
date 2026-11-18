"""Compare systems on val greedy outputs: pooled PER, feature-weighted PER (fPER), and deletion / insertion /
substitution rates, with a per-child bootstrap of the difference against a baseline. Numbers only to stdout.

    $PY -I scripts/compare_val.py --base plain=RUNS/grpo-m1-prov2 --sys fper=RUNS/grpo-m1-prov2-fper \
        --sys sft=RUNS/sft-r16-plain-word-s0-prov2 --sys pasketti=pasketti
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phone_distance as PD  # noqa: E402
import train_sft as T  # noqa: E402

PASKETTI = "RUNS/pasketti-phonetic/hyps_val_plain.jsonl"


def cid(p):
    return os.path.splitext(os.path.basename(p))[0]


def load_hyps(src):
    if src == "pasketti":
        return {cid(r["path"]): r["hyp"] for r in map(json.loads, open(PASKETTI, encoding="utf-8"))}
    return {cid(r["path"]): r["greedy"] for r in map(json.loads, open(os.path.join(src, "hyps_val_ours.jsonl"), encoding="utf-8"))}


def ops(ref, hyp):
    """Plain edit distance split into (sub, del, ins), one optimal path."""
    c = collections.Counter(op.tag for op in Levenshtein.editops(ref, hyp))
    return c["replace"], c["delete"], c["insert"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="name=run_dir")
    ap.add_argument("--sys", action="append", default=[], help="name=run_dir or name=pasketti")
    ap.add_argument("--config", default="RUNS/sft-r16-plain-word-s0-prov2/config.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    val = T.load_items(argparse.Namespace(**cfg))[0]["val"]
    C = PD.cost_matrix(T.VOCAB)
    ID = {p: i for i, p in enumerate(T.VOCAB)}
    specs = [a.base] + a.sys
    names = [s.split("=", 1)[0] for s in specs]
    hyps = {s.split("=", 1)[0]: load_hyps(s.split("=", 1)[1]) for s in specs}
    missing = {n: sum(cid(it["path"]) not in h for it in val) for n, h in hyps.items()}
    if any(missing.values()):
        raise SystemExit(f"val clips missing from outputs: {missing}")

    # per child sums: [ref_len, edits, weighted edits, sub, del, ins]
    kid = {n: collections.defaultdict(lambda: np.zeros(6)) for n in names}
    for it in val:
        ref = list(it["labels"])
        for n in names:
            h = [ID[p] for p in hyps[n][cid(it["path"])]]   # unknown symbol -> KeyError: stop
            s, d, k = ops(ref, h)
            kid[n][it["child"]] += [len(ref), s + d + k, PD.weighted_distance(ref, h, C), s, d, k]
    kids = sorted(kid[names[0]])
    M = {n: np.array([kid[n][c] for c in kids]) for n in names}   # children x 6

    def rates(x):
        t = x.sum(0)
        return {"per": t[1] / t[0], "fper": t[2] / t[0], "sub": t[3] / t[0], "del": t[4] / t[0], "ins": t[5] / t[0]}

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(kids), size=(a.n_boot, len(kids)))
    base = names[0]
    print(f"val: {len(val)} clips, {len(kids)} children; baseline = {base}")
    print(f"{'system':12s} " + " ".join(f"{k:>7s}" for k in ("per", "fper", "sub", "del", "ins")))
    for n in names:
        r = rates(M[n])
        print(f"{n:12s} " + " ".join(f"{r[k]:7.4f}" for k in ("per", "fper", "sub", "del", "ins")))
    for n in names[1:]:
        out = []
        for k, col in (("per", 1), ("fper", 2), ("del", 4), ("ins", 5)):
            diff = M[n][:, col].sum() / M[n][:, 0].sum() - M[base][:, col].sum() / M[base][:, 0].sum()
            bs = [M[n][i, col].sum() / M[n][i, 0].sum() - M[base][i, col].sum() / M[base][i, 0].sum() for i in idx]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(f"{k} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"{n} - {base}: " + "; ".join(out))


if __name__ == "__main__":
    main()

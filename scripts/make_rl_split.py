"""Split the train children into an SFT part and an RL-only part (by child, never by clip).

Within each corpus x group cell: sort children by age, take evenly spaced positions for the RL part
(random offset from the seed), then swap neighbours in age order while that brings the RL part's hours
closer to its target share. Sex balance is reported, not forced. Val / test / sealed children are untouched:
only the split's train.txt is read. Writes <out>/sft.txt and <out>/rl.txt (child IDs, read-only) and
<out>/summary.json (counts only, no IDs).

  python -I scripts/make_rl_split.py --rl-share 0.3 --name rl30
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402


def pick(kids, share, rng):
    """kids: list of (child, age, hours) in one cell -> set of RL children."""
    n = len(kids)
    n_rl = int(round(share * n))
    if n >= 3:
        n_rl = max(1, n_rl)
    if n_rl == 0:
        return set()
    kids = sorted(kids, key=lambda k: (k[1] is None, k[1] if k[1] is not None else 0, k[0]))
    u = rng.random()
    pos = sorted({min(n - 1, int((j + u) * n / n_rl)) for j in range(n_rl)})
    rl = set(pos)
    target = share * sum(k[2] for k in kids)

    def dev(s):
        return abs(sum(kids[i][2] for i in s) - target)

    for _ in range(100):  # swap a RL child with an age neighbour if hours get closer to target
        best, best_d = None, dev(rl)
        for i in sorted(rl):
            for j in (i - 2, i - 1, i + 1, i + 2):
                if 0 <= j < n and j not in rl:
                    s2 = (rl - {i}) | {j}
                    if dev(s2) < best_d - 1e-9:
                        best, best_d = s2, dev(s2)
        if best is None:
            break
        rl = best
    return {kids[i][0] for i in rl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-share", type=float, required=True)
    ap.add_argument("--name", required=True, help="output folder name under splits/phonetic/<splits>/sub/")
    ap.add_argument("--splits", default="v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    split_dir = os.path.join(T.DATA, "splits", "phonetic", a.splits)
    train = set(open(os.path.join(split_dir, "train.txt")).read().split())
    info = {}  # child -> [corpus, group, age_months, sex, hours]  (usable clips only, as SFT loads them)
    cfg = argparse.Namespace(splits=a.splits, audio="plain")
    items, _ = T.load_items(cfg)
    for it in items["train"]:
        info.setdefault(it["child"], [it["corpus"], it["group"], None, None, 0.0])[4] += it["dur"] / 3600
    for corpus in T.TRAIN_CORPORA:
        for line in open(os.path.join(T.MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["child_id"] in info:
                assert not r["sip_sealed"], "sealed child in train: stop"
                info[r["child_id"]][2] = r.get("age_months")
                info[r["child_id"]][3] = r.get("sex")
    assert set(info) <= train

    rng = np.random.default_rng(a.seed)
    cells = collections.defaultdict(list)
    for c, (corpus, group, age, _, h) in info.items():
        cells[(corpus, group)].append((c, age, h))
    rl = set()
    for key in sorted(cells):
        rl |= pick(cells[key], a.rl_share, rng)
    sft = set(info) - rl

    out = os.path.join(split_dir, "sub", a.name)
    os.makedirs(out, exist_ok=True)
    for fn, ids in (("sft.txt", sft), ("rl.txt", rl)):
        p = os.path.join(out, fn)
        if os.path.exists(p):
            os.chmod(p, 0o644)
        open(p, "w").write("\n".join(sorted(ids)) + "\n")
        os.chmod(p, 0o444)

    rows = []
    for key in sorted(cells):
        ks = cells[key]
        r_ = [k for k in ks if k[0] in rl]
        ages = lambda s: [k[1] for k in s if k[1] is not None]  # noqa: E731
        rows.append({"corpus": key[0], "group": key[1], "children": len(ks), "rl_children": len(r_),
                     "hours": round(sum(k[2] for k in ks), 2), "rl_hours": round(sum(k[2] for k in r_), 2),
                     "age_mean_sft": round(float(np.mean(ages([k for k in ks if k[0] not in rl]) or [np.nan])), 1),
                     "age_mean_rl": round(float(np.mean(ages(r_) or [np.nan])), 1)})
    sex = {s: collections.Counter(info[c][3] for c in ids) for s, ids in (("sft", sft), ("rl", rl))}
    summary = {"rl_share": a.rl_share, "seed": a.seed, "splits": a.splits, "children_sft": len(sft),
               "children_rl": len(rl), "hours_sft": round(sum(info[c][4] for c in sft), 1),
               "hours_rl": round(sum(info[c][4] for c in rl), 1), "sex": {k: dict(v) for k, v in sex.items()},
               "cells": rows}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}))
    print(f"{'corpus':14s}{'group':14s}{'kids':>5s}{'rl':>4s}{'hours':>8s}{'rl_h':>7s}{'age_sft':>8s}{'age_rl':>7s}")
    for r in rows:
        print(f"{r['corpus']:14s}{r['group']:14s}{r['children']:5d}{r['rl_children']:4d}{r['hours']:8.1f}"
              f"{r['rl_hours']:7.1f}{r['age_mean_sft']:8.1f}{r['age_mean_rl']:7.1f}")


if __name__ == "__main__":
    main()

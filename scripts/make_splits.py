"""Make the phonetic splits (DATA.md §3): by child, never by clip.

    $PY -I scripts/make_splits.py --version v1          # writes splits/phonetic/v1/, refuses to overwrite

Rules (DATA.md §3):
  Train   training-pool corpora minus the children below
  Val     held-out children from each training corpus + all of Preston + 15 SPROUT children
  Test A  more held-out children from each training corpus
  Test B  TorringtonEaton, McAllister (JIBO added when its manifest exists)
  Test C  the 40 sealed SPROUT children + 18 more, chosen so SSD and typical are balanced
  Providence under age 3 is held back: in no split.
Within each corpus, held-out children are drawn per group (typical / SSD / ...) so
each split keeps the corpus's mix. Fixed seed. Corpora with under 10 children
(Providence) give their two smallest children (by hours) to val and Test A.
A child in both PERCEPT corpora is drawn once (with PERCEPT-R).

Writes <split>.txt (one child_id per line), SPLITS.json (rules, counts, hours, sha256).
These files hold child IDs: they stay in the data folder, never in git or W&B.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import INFO_FLAGS, OUT  # noqa: E402

TRAIN_POOL = ["SPROUT", "ACAD", "Cummings", "PERCEPT-R", "PERCEPT-GFTA", "Penney", "Providence"]
TEST_B = ["TorringtonEaton", "McAllister"]
VAL_ONLY = ["Preston"]
HELD_FRAC = 0.10        # per training corpus, for val and again for Test A
SPROUT_VAL = 15
SPROUT_TESTC_EXTRA = 18
SEED = 0
FEW_CHILDREN = 10     # corpora with fewer children: hold out the smallest ones, to keep training hours


def usable(r):
    if set(r["flags"]) - INFO_FLAGS:
        return False
    if r["corpus"] == "Providence" and (r["age_months"] or 0) < 36:
        return False
    return True


def load():
    kids = {}  # child_id -> info
    for c in TRAIN_POOL + TEST_B + VAL_ONLY:
        path = os.path.join(OUT, "phonetic", f"{c}.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            k = kids.setdefault(r["child_id"], {"corpora": set(), "group": r["group"], "hours": 0.0,
                                                "sealed": r["sip_sealed"]})
            k["corpora"].add(c)
            if usable(r):
                k["hours"] += (r["duration"] or 0) / 3600
    return kids


def draw(ids, n, rng, kids):
    """Take n children spread across groups in proportion (at least 1 per group if n allows)."""
    by_g = collections.defaultdict(list)
    for i in sorted(ids):
        by_g[kids[i]["group"]].append(i)
    for g in by_g:
        rng.shuffle(by_g[g])
    picked, total = [], len(ids)
    for g, lst in sorted(by_g.items(), key=lambda x: -len(x[1])):
        k = max(1, round(n * len(lst) / total)) if n >= len(by_g) else 0
        picked += lst[:k]
    rest = [i for g in sorted(by_g) for i in by_g[g] if i not in picked]
    rng.shuffle(rest)
    picked += rest[: max(0, n - len(picked))]
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(OUT.rstrip("/")), "splits", "phonetic", args.version)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} exists. Frozen splits are never edited; use a new --version.")
    rng = random.Random(SEED)
    kids = load()
    split = {}

    # a child seen in a training corpus and an outside set would leak: stop
    for cid, k in kids.items():
        if k["corpora"] & set(TRAIN_POOL) and k["corpora"] & set(TEST_B + VAL_ONLY):
            sys.exit(f"{cid} is in both a training corpus and an outside set: {sorted(k['corpora'])}")

    for cid, k in kids.items():
        if k["corpora"] & set(TEST_B):
            split[cid] = "testB"
        elif k["corpora"] & set(VAL_ONLY):
            split[cid] = "val"

    # SPROUT: sealed -> Test C; then 18 extra for Test C (balanced SSD / typical); 15 for val
    sprout = [c for c, k in kids.items() if "SPROUT" in k["corpora"]]
    for c in sprout:
        if kids[c]["sealed"]:
            split[c] = "testC"
    free = [c for c in sprout if c not in split]
    sealed_g = collections.Counter(kids[c]["group"] for c in sprout if kids[c]["sealed"])
    need = {g: 0 for g in ("SSD", "typical")}
    for _ in range(SPROUT_TESTC_EXTRA):  # add to whichever of SSD / typical is smaller
        g = min(need, key=lambda x: sealed_g[x] + need[x])
        need[g] += 1
    for g, n in need.items():
        pool = sorted(c for c in free if kids[c]["group"] == g)
        rng.shuffle(pool)
        for c in pool[:n]:
            split[c] = "testC"
    free = [c for c in sprout if c not in split]
    for c in draw(free, SPROUT_VAL, rng, kids):
        split[c] = "val"

    # other training corpora: 10% of children to val, 10% to Test A (at least 1 each)
    for corpus in TRAIN_POOL:
        if corpus == "SPROUT":
            continue
        # a child in PERCEPT-R and PERCEPT-GFTA is drawn once, with PERCEPT-R
        ids = [c for c, k in kids.items() if corpus in k["corpora"] and c not in split and k["hours"] > 0
               and not (corpus == "PERCEPT-GFTA" and "PERCEPT-R" in k["corpora"])]
        n = max(1, round(HELD_FRAC * len(ids)))
        if len(ids) < FEW_CHILDREN:  # e.g. Providence: hold out the children with the fewest hours
            ids.sort(key=lambda c: kids[c]["hours"])
            split[ids[0]], split[ids[1]] = "val", "testA"
            continue
        for c in draw(ids, n, rng, kids):
            split[c] = "val"
        ids = [c for c in ids if c not in split]
        for c in draw(ids, n, rng, kids):
            split[c] = "testA"
    # SPROUT Test A: same 10% rule on what is left
    free = [c for c in sprout if c not in split]
    for c in draw(free, max(1, round(HELD_FRAC * len(sprout))), rng, kids):
        split[c] = "testA"

    for cid, k in kids.items():
        split.setdefault(cid, "train")

    # checks
    assert all(split[c] == "testC" for c, k in kids.items() if k["sealed"]), "sealed child outside Test C"
    assert not any(k["sealed"] for c, k in kids.items() if split[c] in ("train", "val")), "sealed child in train/val"

    os.makedirs(outdir)
    summary = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    files = {}
    for s in ("train", "val", "testA", "testB", "testC"):
        ids = sorted(c for c in split if split[c] == s)
        p = os.path.join(outdir, f"{s}.txt")
        with open(p, "w") as f:
            f.write("\n".join(ids) + "\n")
        files[s] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        for c in ids:
            for corpus in kids[c]["corpora"]:
                cell = summary[s][f"{corpus}/{kids[c]['group']}"]
                cell[0] += 1
                cell[1] += kids[c]["hours"] / len(kids[c]["corpora"])
    meta = {"version": args.version, "seed": SEED, "few_children": FEW_CHILDREN, "held_frac": HELD_FRAC, "sprout_val": SPROUT_VAL,
            "sprout_testC_extra": SPROUT_TESTC_EXTRA, "train_pool": TRAIN_POOL, "test_b": TEST_B,
            "val_only": VAL_ONLY, "providence_min_age_months": 36, "sha256": files,
            "summary": {s: {k: {"children": v[0], "usable_hours": round(v[1], 2)} for k, v in d.items()}
                        for s, d in summary.items()}}
    json.dump(meta, open(os.path.join(outdir, "SPLITS.json"), "w"), indent=1)
    for p in os.listdir(outdir):
        os.chmod(os.path.join(outdir, p), 0o444)

    print(f"{'split':7}{'corpus/group':32}{'children':>9}{'usable h':>9}")
    for s in ("train", "val", "testA", "testB", "testC"):
        tot_k = tot_h = 0
        for k, (n, h) in sorted(summary[s].items()):
            print(f"{s:7}{k:32}{n:>9}{h:>9.1f}")
            tot_k += n
            tot_h += h
        print(f"{s:7}{'TOTAL':32}{tot_k:>9}{tot_h:>9.1f}\n")
    print("sha256:", json.dumps(files, indent=1))
    print(f"frozen (read-only) in {outdir}")


if __name__ == "__main__":
    main()

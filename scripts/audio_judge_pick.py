"""Audio-judge step 2: pick train clips for the judge check.

Stratified: SSD / typical x age band (3, 4, 5, 6-7, 8+). Train children only (split v2). Never val/test, never sealed children.
Writes a private JSONL (mode 600) outside the repo:
    clip path, child (hashed), corpus, age, label phones (words joined by ' | ').

    python -I scripts/audio_judge_pick.py --out <scratch dir>/clips.jsonl --per-cell 30
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sft import DATA, MANIFESTS, TRAIN_CORPORA, INFO_FLAGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=6.0)
    ap.add_argument("--min-phones", type=int, default=3)
    ap.add_argument("--max-phones", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="v2")
    a = ap.parse_args()

    split_dir = os.path.join(DATA, "splits", "phonetic", a.splits)
    train = set(open(os.path.join(split_dir, "train.txt")).read().split())
    path_of = {}
    for p in sorted(os.listdir(os.path.join(DATA, "processed", "index"))):
        for line in open(os.path.join(DATA, "processed", "index", p)):
            r = json.loads(line)
            path_of[r["clip_id"]] = r["denoised_path"]

    def band(m):
        y = int((m or -12) // 12)
        return None if y < 3 else str(y) if y <= 5 else "6-7" if y <= 7 else "8+"

    cells = defaultdict(lambda: defaultdict(list))   # (group, age band) -> child -> clips
    for corpus in TRAIN_CORPORA:
        for line in open(os.path.join(MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["child_id"] not in train or set(r["flags"]) - INFO_FLAGS or r["clip_id"] not in path_of:
                continue
            if r["sip_sealed"]:
                raise RuntimeError("sealed child in train split: stop")
            b = band(r["age_months"])
            if b is None or r["group"] not in ("SSD", "typical"):   # under 3 (as in SFT), unknown group
                continue
            n = sum(1 for p in r["ipa_normalized"].split() if p != "|")
            if not (a.min_dur <= r["duration"] <= a.max_dur and a.min_phones <= n <= a.max_phones):
                continue
            cells[(r["group"], b)][r["child_id"]].append(r)

    rng = random.Random(a.seed)
    out = []
    for key in sorted(cells):
        kids = sorted(cells[key])
        rng.shuffle(kids)
        pools = {k: rng.sample(cells[key][k], len(cells[key][k])) for k in kids}
        picked, rnd = [], 0
        while len(picked) < a.per_cell and any(rnd < len(pools[k]) for k in kids):
            for k in kids:   # one clip per child per round: as many children as possible
                if len(picked) < a.per_cell and rnd < len(pools[k]):
                    picked.append(pools[k][rnd])
            rnd += 1
        for r in picked:
            out.append({"path": path_of[r["clip_id"]], "corpus": r["corpus"],
                        "child": hashlib.sha256(r["child_id"].encode()).hexdigest()[:12],
                        "age_months": r["age_months"], "group": r["group"], "age_band": key[1],
                        "duration": r["duration"], "phones": r["ipa_normalized"]})
        corp = sorted({r["corpus"] for r in picked})
        print(f"{key[0]:8s} {key[1]:4s}: {len(picked)} clips, {len({r['child_id'] for r in picked})} children, corpora {corp}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} clips")


if __name__ == "__main__":
    main()

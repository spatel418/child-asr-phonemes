"""Is there one direction in the model's last hidden layer that makes it drop phones? (weight-space reading, local)

Idea (MIDTRAINING-style one-change test; see RESEARCH_LOG 2026-11-17): if deletions come from a single "say blank"
direction that overrides phones the model does hear, removing that direction should cut deletions without adding
junk (an obstacle, like the Qwen loop direction). If deleted-phone frames can't be told apart from true-blank frames,
the model just doesn't hear those phones (missing ability; editing can't help).

Data use: train children only for finding and tuning; val once at the end; test never. Children are split in two
(seed 0): A = find the direction, B = pick the edit strength. Everything stays on Quest; output is numbers only.

Steps
 1. For each clip: hidden vectors h [T, 1024] (input of the CTC head), log-probs, greedy path, and a CTC forced
    alignment of the human labels (Viterbi). Frame groups:
      dropped  - frame of a reference phone (its best frame) where the whole aligned segment is greedy-blank
      blank_ok - forced-aligned blank and greedy blank (sampled 1 in 10)
      phone_ok - aligned phone p and greedy p (sampled 1 in 10)
 2. Direction d = unit(mean h[dropped] - mean h[blank_ok]) on A. Separability on B: AUC of h.d, AUC of the
    blank-minus-best-phone logit margin (what a blank penalty uses), and AUC of a full linear probe (logistic, on A).
 3. Edit: h' = h - alpha * ((h - c) . d) d with c = mean h[blank_ok] (the direction's value is pulled toward its
    typical true-blank level). Same as changing the head weights (W' = W (I - alpha d d^T), b' = b + alpha W d d^T c).
    alpha on a grid; pick on B (pooled PER); report val at that alpha (all val + the two val halves of blank_penalty).

    $PY -I scripts/deletion_direction.py --run RUNS/sft-r16-plain-word-s0-prov2 [--rl RUNS/grpo-m1-prov2] --out RUNS/deldir-...
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
import blank_penalty as BP  # noqa: E402
import rl_diagnose as D  # noqa: E402
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def ctc_align(lp, labels):
    """Viterbi CTC forced alignment. lp [T, V] log-probs (numpy), labels list of ids (no blanks).
    Returns per frame the label index it belongs to (0..L-1) or -1 for blank; None if impossible."""
    L = len(labels)
    S = 2 * L + 1
    ext = np.zeros(S, dtype=np.int64)
    ext[1::2] = labels
    Tn = lp.shape[0]
    NEG = -1e30
    skip = np.zeros(S, dtype=bool)  # s-2 -> s allowed: non-blank and different from two back
    skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    dp = np.full(S, NEG)
    dp[0] = lp[0, 0]
    if S > 1:
        dp[1] = lp[0, ext[1]]
    bp = np.zeros((Tn, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, Tn):
        c0 = dp
        c1 = np.concatenate([[NEG], dp[:-1]])
        c2 = np.where(skip, np.concatenate([[NEG, NEG], dp[:-2]]), NEG)
        stack = np.stack([c0, c1, c2])
        arg = stack.argmax(0)
        best = stack.max(0)
        dp = np.where(best > NEG / 2, best + lp[t, ext], NEG)
        bp[t] = arg
    cand = [S - 1] + ([S - 2] if S > 1 else [])
    s = max(cand, key=lambda e: dp[e])
    if dp[s] < NEG / 2:
        return None
    path = [0] * Tn
    for t in range(Tn - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return [(p - 1) // 2 if p % 2 == 1 else -1 for p in path]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def encode(model, items, ids, device):
    """Yield (item index, hidden [T,1024] float32 cpu, n frames) per clip."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ds = T.Clips(items)
    for b in G.batches_by_seconds(ids, items, 200.0):
        x, mask = G.load_batch(ds, b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = base.wavlm(input_values=x, attention_mask=mask)[0]
        n = base._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        hid = hid.float()
        for k, i in enumerate(b):
            yield i, hid[k, : int(n[k])], int(n[k])


def head_fn(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.lm_head


def collect(model, items, ids, device, rng):
    head = head_fn(model)
    groups = collections.defaultdict(list)
    n_drop_phones = n_phones = 0
    for i, h, n in encode(model, items, ids, device):
        with torch.no_grad():
            logits = head(h)
        lp = torch.log_softmax(logits, -1).cpu().numpy()
        greedy = lp.argmax(-1)
        lab = items[i]["labels"]
        al = ctc_align(lp, lab)
        if al is None:
            continue
        hc = h.cpu().numpy()
        seg = collections.defaultdict(list)
        for t, j in enumerate(al):
            seg[j].append(t)
        for j in range(len(lab)):
            fr = seg.get(j, [])
            if not fr:
                continue
            n_phones += 1
            if all(greedy[t] == 0 for t in fr):
                t = max(fr, key=lambda t: lp[t, lab[j]])
                groups["dropped"].append(hc[t])
                groups["dropped_margin"].append(float(lp[t, 0] - lp[t, 1:].max()))
                n_drop_phones += 1
        for t, j in enumerate(al):
            if rng.random() >= 0.1:
                continue
            if j == -1 and greedy[t] == 0:
                groups["blank_ok"].append(hc[t])
                groups["blank_ok_margin"].append(float(lp[t, 0] - lp[t, 1:].max()))
            elif j >= 0 and greedy[t] == lab[j]:
                groups["phone_ok"].append(hc[t])
    out = {k: np.asarray(v, dtype=np.float32) for k, v in groups.items()}
    out["stats"] = {"phones": n_phones, "dropped_phones": n_drop_phones}
    return out


@torch.no_grad()
def score(model, items, ids, device, d, c, alphas, keyfn=None):
    """Pooled edit counts per alpha (and per key from keyfn(item) if given)."""
    head = head_fn(model)
    d_t, c_t = torch.tensor(d, device=device), torch.tensor(c, device=device)
    acc = {a: collections.defaultdict(lambda: np.zeros(4)) for a in alphas}
    for i, h, n in encode(model, items, ids, device):
        proj = (h - c_t) @ d_t
        for a in alphas:
            hh = h - a * proj[:, None] * d_t[None, :]
            hyp, _ = T.greedy(head(hh)[None], [n])
            o = BP.ops(items[i]["labels"], hyp[0])
            for key in ["all"] + (keyfn(items[i]) if keyfn else []):
                acc[a][key] += (len(items[i]["labels"]), *o)
    return {a: {k: {"per": float(v[1:].sum() / max(v[0], 1)), "sub": float(v[1] / max(v[0], 1)),
                    "del": float(v[2] / max(v[0], 1)), "ins": float(v[3] / max(v[0], 1))} for k, v in acc[a].items()}
            for a in alphas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SFT run dir")
    ap.add_argument("--rl", default=None, help="optional RL run on top (its best.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips-a", type=int, default=4000)
    ap.add_argument("--clips-b", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    model, cfg = D.load(a.run, a.rl, device)
    items = T.load_items(argparse.Namespace(**cfg))[0]
    train, val = items["train"], items["val"]
    kids = sorted({it["child"] for it in train})
    rng = random.Random(a.seed)
    rng.shuffle(kids)
    half = {k: "A" if j % 2 == 0 else "B" for j, k in enumerate(kids)}
    ids_a = [i for i, it in enumerate(train) if half[it["child"]] == "A"]
    ids_b = [i for i, it in enumerate(train) if half[it["child"]] == "B"]
    ids_a, ids_b = rng.sample(ids_a, min(a.clips_a, len(ids_a))), rng.sample(ids_b, min(a.clips_b, len(ids_b)))

    gA = collect(model, train, ids_a, device, rng)
    gB = collect(model, train, ids_b, device, rng)
    d = gA["dropped"].mean(0) - gA["blank_ok"].mean(0)
    d = d / np.linalg.norm(d)
    c = gA["blank_ok"].mean(0)
    res = {"run": a.run, "rl": a.rl, "counts_A": {k: len(v) for k, v in gA.items() if k != "stats"}, "stats_A": gA["stats"],
           "counts_B": {k: len(v) for k, v in gB.items() if k != "stats"}, "stats_B": gB["stats"]}
    # separability on B (direction and probe fit on A)
    from sklearn.linear_model import LogisticRegression
    pa = (gB["dropped"] - c) @ d
    pn = (gB["blank_ok"] - c) @ d
    X = np.concatenate([gA["dropped"], gA["blank_ok"]])
    y = np.concatenate([np.ones(len(gA["dropped"])), np.zeros(len(gA["blank_ok"]))])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
    sB = clf.decision_function(np.concatenate([gB["dropped"], gB["blank_ok"]]))
    nd = len(gB["dropped"])
    res["separability_B"] = {"auc_direction": auc(pa, pn), "auc_linear_probe": auc(sB[:nd], sB[nd:]),
                             "auc_blank_margin": auc(-np.asarray(gB["dropped_margin"]), -np.asarray(gB["blank_ok_margin"])),
                             "dropped_vs_phone_ok_cosine_to_d": float(((gB["phone_ok"] - c) @ d).mean())}
    # pick alpha on B, then val once
    sb = score(model, train, ids_b, device, d, c, ALPHAS)
    best = min(ALPHAS, key=lambda x: sb[x]["all"]["per"])
    res["train_B"] = {str(x): sb[x]["all"] for x in ALPHAS}
    res["picked_alpha_on_B"] = best
    sv = score(model, val, list(range(len(val))), device, d, c, sorted({0.0, best}),
               keyfn=lambda it: [f"corpus:{it['corpus']}", f"tag:{it['tag']}"])
    res["val"] = {str(x): sv[x] for x in sv}
    tags = [t for t in T.SCORE_TAGS]
    for x in sv:
        res["val"][str(x)]["select_score"] = float(np.mean([sv[x][f"tag:{t}"]["per"] for t in tags if f"tag:{t}" in sv[x]]))
    np.save(os.path.join(a.out, "direction.npy"), np.stack([d, c]))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("counts_A", "stats_A", "separability_B", "train_B", "picked_alpha_on_B")}, indent=1))
    for x in sv:
        r = res["val"][str(x)]
        print(f"val alpha={x}: select {r['select_score']:.4f}  all PER {r['all']['per']:.4f} del {r['all']['del']:.3f} ins {r['all']['ins']:.3f}"
              + "  " + " ".join(f"{k[7:]} {v['per']:.3f}" for k, v in r.items() if k.startswith("corpus:")))


if __name__ == "__main__":
    main()

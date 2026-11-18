"""MWER step 1: is there room for MWER? No training.

For each clip: CTC prefix beam search -> the N most likely distinct phone strings, each with its CTC
log p(y|x). Reports, on train and val clips:
  greedy PER, beam-top-1 PER, oracle PER (best of top 4 / 8 / 16 beam guesses),
  best-of-16 random sampling at T=1.0 (what GRPO sees), the expected PER under the model's own odds
  renormalized over the top 8 (what MWER would push down), and how often the best beam guess is not
  already the top one.
If oracle-of-8 is far below greedy, MWER has something to move toward; if it is close, it does not.

    $PY -I scripts/nbest_ceiling.py --init RUNS/sft-r16-plain-word-s0 --out RUNS/nbest-sft100 --clips 3000
"""
import argparse
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

NEG = -1e30


def prefix_beam(lp, beam=16, prune=-10.0, max_tok=8):
    """CTC prefix beam search on one clip. lp: [T, V] log-probs (numpy), blank = 0.
    Returns [(prefix tuple, log p(prefix | x))] best first, up to `beam` entries.
    Per frame only tokens with lp > prune (at most max_tok, always the best one) are expanded."""
    beams = {(): (0.0, NEG)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    for t in range(lp.shape[0]):
        row = lp[t]
        cand = np.argsort(-row)[:max_tok]
        cand = [int(c) for k, c in enumerate(cand) if k == 0 or row[c] > prune]
        nxt = {}

        def add(pfx, b, nb):
            ob, onb = nxt.get(pfx, (NEG, NEG))
            nxt[pfx] = (np.logaddexp(ob, b), np.logaddexp(onb, nb))

        for pfx, (pb, pnb) in beams.items():
            tot = np.logaddexp(pb, pnb)
            last = pfx[-1] if pfx else None
            for c in cand:
                p = float(row[c])
                if c == 0:
                    add(pfx, tot + p, NEG)
                elif c == last:
                    add(pfx, NEG, pnb + p)          # repeat without blank: same prefix
                    add(pfx + (c,), NEG, pb + p)    # blank in between: new phone
                else:
                    add(pfx + (c,), NEG, tot + p)
        beams = dict(sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:beam])
    out = [(list(p), float(np.logaddexp(*v))) for p, v in beams.items() if np.logaddexp(*v) > -1e29]  # drop impossible
    return sorted(out, key=lambda x: -x[1])


@torch.no_grad()
def run_set(model, ds, items, ids, args, device):
    ref_n = 0
    err = {"greedy": 0, "beam1": 0, "oracle4": 0, "oracle8": 0, "oracle16": 0, "sample16": 0}
    exp_per, n_hyp, best_not_top, greedy_not_beam1, top_mass = [], [], 0, 0, []
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for b in G.batches_by_seconds(ids, items, 200.0):
        x, mask = G.load_batch(ds, b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, n = G.frame_logits(model, x, mask)
        logits = logits.float()
        g, _ = T.greedy(logits, n.tolist())
        samp = G.collapse(G.sample_paths(logits, n, 16, 1.0, generator=gen))
        lp = torch.log_softmax(logits, -1).cpu().numpy()
        for k, i in enumerate(b):
            ref = items[i]["labels"]
            ref_n += len(ref)
            hyps = prefix_beam(lp[k, : int(n[k])], beam=args.beam)
            e = [Levenshtein.distance(ref, h) for h, _ in hyps]
            err["greedy"] += Levenshtein.distance(ref, g[k])
            err["beam1"] += e[0]
            for m in (4, 8, 16):
                err[f"oracle{m}"] += min(e[:m])
            err["sample16"] += min(Levenshtein.distance(ref, h) for h in samp[k])
            s = np.array([sc for _, sc in hyps[:8]])
            w = np.exp(s - s.max())
            w /= w.sum()
            exp_per.append(float((w * np.array(e[:8])).sum() / max(len(ref), 1)))
            top_mass.append(float(w[0]))
            n_hyp.append(len(hyps))
            best_not_top += int(min(e[:8]) < e[0])
            greedy_not_beam1 += int(hyps[0][0] != g[k])
    c = max(len(ids), 1)
    out = {f"{k}_per": v / max(ref_n, 1) for k, v in err.items()}
    out.update({"clips": len(ids), "expected_per_top8_mean": float(np.mean(exp_per)),
                "top1_share_of_top8_mass": float(np.mean(top_mass)), "mean_beam_hyps": float(np.mean(n_hyp)),
                "share_best_of_8_beats_top1": best_not_top / c, "share_greedy_differs_from_beam1": greedy_not_beam1 / c})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=3000, help="per split")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--train-children", default=None, help="train clips from these children (default: SFT's)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _, sft_cfg, use_ema = G.load_sft(a.init, device)
    model.eval()
    cfg = argparse.Namespace(**sft_cfg)
    if a.train_children:
        cfg.train_children = a.train_children
    items, _ = T.load_items(cfg)
    train, val = items["train"], items["val"]
    rng = random.Random(a.seed)
    tr_ids = rng.sample(range(len(train)), min(a.clips, len(train)))
    va_ids = G.val_subset(val, a.clips, rng)
    res = {"init": a.init, "init_weights": "ema" if use_ema else "live", "beam": a.beam,
           "train_children": a.train_children or "sft",
           "train": run_set(model, T.Clips(train), train, tr_ids, a, device),
           "val": run_set(model, T.Clips(val), val, va_ids, a, device)}
    json.dump(res, open(os.path.join(a.out, "nbest.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

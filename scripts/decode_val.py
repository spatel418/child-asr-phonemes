"""Greedy val outputs for an SFT run or an RL checkpoint on top of it (val only). Same format as
error_report.py --decode (greedy + beam1 as phone symbols, keyed by audio path), written to <out>/hyps_val_ours.jsonl.

    $PY -I scripts/decode_val.py --sft RUNS/sft-r16-plain-word-s0-prov2 --rl RUNS/grpo-m1-prov2 --out RUNS/grpo-m1-prov2
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbest_ceiling as NB  # noqa: E402
import rl_diagnose as D  # noqa: E402
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--rl", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    model, cfg = D.load(a.sft, a.rl, device)
    val = T.load_items(argparse.Namespace(**cfg))[0]["val"]
    ds = T.Clips(val)
    with open(os.path.join(a.out, "hyps_val_ours.jsonl"), "w") as f:
        for b in G.batches_by_seconds(range(len(val)), val, 200.0):
            x, mask = G.load_batch(ds, b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, n = G.frame_logits(model, x, mask)
            logits = logits.float()
            g, _ = T.greedy(logits, n.tolist())
            lp = torch.log_softmax(logits, -1).cpu().numpy()
            for k, i in enumerate(b):
                beam = NB.prefix_beam(lp[k, : int(n[k])], beam=16)
                f.write(json.dumps({"path": val[i]["path"], "greedy": [T.VOCAB[t] for t in g[k]],
                                    "beam1": [T.VOCAB[t] for t in beam[0][0]] if beam else []}, ensure_ascii=False) + "\n")
    os.chmod(os.path.join(a.out, "hyps_val_ours.jsonl"), 0o600)


if __name__ == "__main__":
    main()

"""Re-score a saved SFT checkpoint on val, with every breakdown train_sft.evaluate makes.

    $PY -I scripts/eval_ckpt.py RUNS/sft-r16-s0            # best.pt, EMA or live as chosen in training
    $PY -I scripts/eval_ckpt.py RUNS/sft-r16-s0 --ckpt last.pt --weights live

Val split only: test splits are never read (load_items reads train and val only).
Writes <run-dir>/eval_val_<ckpt>.json and prints a table by group and corpus.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--weights", default="auto", choices=["auto", "ema", "live"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(a.run_dir, a.ckpt), map_location=device, weights_only=False)
    cfg = argparse.Namespace(**ck["config"])
    model, head, lora = T.build_model(cfg, device)
    use_ema = ck["state"].get("best_is_ema", False) if a.weights == "auto" else a.weights == "ema"
    src = ck["ema"] if use_ema else ck["params"]
    for p, v in zip(head + lora, src):
        p.data.copy_(v.to(p.dtype))

    items, _ = T.load_items(cfg)
    val = items["val"]
    order = sorted(range(len(val)), key=lambda i: val[i]["dur"])
    batches, cur, longest = [], [], 0.0
    for i in order:
        if cur and max(longest, val[i]["dur"]) * (len(cur) + 1) > cfg.batch_seconds:
            batches.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, val[i]["dur"])
    batches.append(cur)
    loader = torch.utils.data.DataLoader(T.Clips(val), batch_sampler=batches, collate_fn=T.collate,
                                         num_workers=a.workers)
    m = T.evaluate(model, loader, val, device)
    m.update({"ckpt": a.ckpt, "weights": "ema" if use_ema else "live",
              "step": ck["state"]["step"], "n_val_clips": len(val)})
    out = os.path.join(a.run_dir, f"eval_val_{os.path.splitext(a.ckpt)[0]}.json")
    json.dump(m, open(out, "w"), indent=1)

    print(f"{a.ckpt} step {m['step']} ({m['weights']} weights), {len(val)} val clips")
    print(f"select_score {m['val/select_score']:.4f}  PER all {m['val/per/all']:.4f}  "
          f"sub {m['val/sub_rate']:.3f} del {m['val/del_rate']:.3f} ins {m['val/ins_rate']:.3f}  "
          f"len_ratio {m['val/len_ratio']:.3f}  blank {m['val/blank_rate']:.3f}")
    for k in sorted(k for k in m if k.startswith("val/per/") and k != "val/per/all"):
        key = k[len("val/per/"):]
        print(f"  {key:28s} PER {m[k]:.3f}  empty {m.get('val/empty_rate/' + key, 0):.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()

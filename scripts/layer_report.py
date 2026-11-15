"""Which encoder layers did training change most?  Weights only, no audio, runs on CPU.

    $PY -I scripts/layer_report.py RUNS/sft-r16-plain-s0 [--ckpt best.pt] [--weights ema|live]

For each transformer layer: relative weight change ||dW|| / ||W|| (Frobenius), per module and summed.
  LoRA run:  dW = (alpha/r) * B @ A for each adapted matrix (q, k, v, out).
  Full run:  dW = trained weight - pretrained weight, for every trainable matrix in the layer.
A bigger number means training moved that layer further from pretrained WavLM. It is not proof that the
layer matters more for PER; for that you need an ablation (switch one layer's change off, re-score val).
Writes <run>/layer_report_<ckpt>.json and, with --wandb, logs a bar chart to the run's W&B group.
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402

LAYER = re.compile(r"encoder\.layers\.(\d+)\.(.+?)\.(?:lora_[AB]\.default\.weight|weight)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--weights", default="ema", choices=["ema", "live"])
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    ck = torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu", weights_only=False, mmap=True)  # mmap: full-FT ckpts are 5 GB
    cfg = argparse.Namespace(**ck["config"])
    model, head, rest = T.build_model(cfg, torch.device("cpu"))
    names = {id(p): n for n, p in model.named_parameters()}
    saved = ck["ema"] if a.weights == "ema" else ck["params"]
    trained = {names[id(p)]: v.float() for p, v in zip(head + rest, saved)}
    base = dict(model.named_parameters())

    per = {}  # (layer, module) -> (||dW||^2, ||W||^2)
    if cfg.finetune == "lora":
        scale = cfg.lora_alpha / cfg.lora_r
        for n, A in trained.items():
            if ".lora_A." not in n:
                continue
            B = trained[n.replace(".lora_A.", ".lora_B.")]
            W = base[n.replace("lora_A.default.weight", "base_layer.weight")].detach().float()
            m = LAYER.search(n)
            per[(int(m.group(1)), m.group(2))] = (float((scale * B @ A).pow(2).sum()), float(W.pow(2).sum()))
    else:
        for n, v in trained.items():
            m = LAYER.search(n)
            if not m or v.dim() < 2:
                continue
            W = base[n].detach().float()
            per[(int(m.group(1)), m.group(2))] = (float((v - W).pow(2).sum()), float(W.pow(2).sum()))

    layers = sorted({k[0] for k in per})
    mods = sorted({k[1] for k in per})
    rows = []
    for L in layers:
        d2 = sum(per[(L, m)][0] for m in mods if (L, m) in per)
        w2 = sum(per[(L, m)][1] for m in mods if (L, m) in per)
        rows.append({"layer": L, "rel_change": (d2 / w2) ** 0.5,
                     **{m: (per[(L, m)][0] / per[(L, m)][1]) ** 0.5 for m in mods if (L, m) in per}})
    top = max(r["rel_change"] for r in rows)
    short = [m.split(".")[-1] for m in mods]
    print(f"{a.run_dir} ({cfg.finetune}, {a.ckpt}, {a.weights}): relative weight change per layer")
    print("layer  total   " + " ".join(f"{s:>8}" for s in short))
    for r in rows:
        bar = "#" * round(30 * r["rel_change"] / top)
        print(f"{r['layer']:>5}  {r['rel_change']:.4f}  " + " ".join(f"{r.get(m, 0):8.4f}" for m in mods) + "  " + bar)
    out = os.path.join(a.run_dir, f"layer_report_{os.path.splitext(a.ckpt)[0]}.json")
    json.dump({"finetune": cfg.finetune, "weights": a.weights, "rows": rows}, open(out, "w"), indent=1)
    print("written", out)

    if a.wandb:
        import wandb
        run = os.path.basename(os.path.normpath(a.run_dir))
        w = wandb.init(project="child-asr-phonemes", name=f"{run}:layers", group=run, job_type="analysis",
                       id="layers-" + run, resume="allow", config={"source_run": run, "ckpt": a.ckpt})
        t = wandb.Table(columns=["layer", "rel_change"], data=[[r["layer"], r["rel_change"]] for r in rows])
        w.log({"layers/rel_change": wandb.plot.bar(t, "layer", "rel_change", title=f"{run}: weight change per layer")})
        w.finish()


if __name__ == "__main__":
    main()

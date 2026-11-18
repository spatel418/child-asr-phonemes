"""Why does RL help so little? Val split (+ a train sample) only; test splits are never read.

For each clip, for the SFT start and for an RL checkpoint on top of it:
  greedy edits (sub/del/ins), 64 samples at T=1.0 (GRPO uses 16; best-of-4..64 shows what more samples would add): best-of-16 edits, mean edits,
  share of samples better than greedy, and whether ANY sample beats greedy (= the clip gives RL a signal).
Breakdowns: corpus, score tag, task, age band, reference length, and greedy error type.
Writes <out>/clips.jsonl (per clip: numbers + audio path, stays on Quest) and <out>/summary.json / summary.md.

    $PY -I scripts/rl_diagnose.py --sft RUNS/sft-r16-plain-word-s0-prov2 --rl RUNS/grpo-m1-prov2 --out RUNS/rl-diagnose-prov2
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
import error_report as E  # noqa: E402
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402


def ops(r, y):
    s = d = i = 0
    for o in Levenshtein.editops(r, y):
        s += o.tag == "replace"
        d += o.tag == "delete"
        i += o.tag == "insert"
    return s, d, i


def load(sft, rl, device):
    model, head, lora, whead, cfg, _ = G.load_sft(sft, device)
    if rl:
        ck = torch.load(os.path.join(rl, "best.pt"), map_location="cpu", weights_only=False)
        src = ck["params"]
        assert len(src) == len(head + lora + whead)
        for p, v in zip(head + lora + whead, src):
            p.data.copy_(v.to(p.dtype))
    model.eval()
    return model, cfg


@torch.no_grad()
def run(model, items, ids, device, seed):
    ds = T.Clips(items)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for b in G.batches_by_seconds(ids, items, 200.0):
        x, mask = G.load_batch(ds, b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, n = G.frame_logits(model, x, mask)
        logits = logits.float()
        g, _ = T.greedy(logits, n.tolist())
        samp = G.collapse(G.sample_paths(logits, n, 64, 1.0, generator=gen))
        for k, i in enumerate(b):
            ref = items[i]["labels"]
            ge = ops(ref, g[k])
            se = [Levenshtein.distance(ref, h) for h in samp[k]]
            s16 = se[:16]
            out[i] = {"greedy": ge, "best16": min(s16), "mean16": float(np.mean(s16)),
                      "better_share": float(np.mean([e < sum(ge) for e in s16])), "glen": len(g[k]),
                      **{f"best{m}": min(se[:m]) for m in (4, 8, 32, 64)},
                      **{f"signal{m}": int(min(se[:m]) < sum(ge)) for m in (4, 8, 16, 32, 64)}}
    return out


def table(rows, keyfn, items):
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for i, r in rows.items():
        it = items[i]
        ref = len(it["labels"])
        for key in ["all"] + keyfn(i):
            a = acc[key]
            a["clips"] += 1
            a["ref"] += ref
            for m in ("sft", "rl"):
                s, d, n_ins = r[m]["greedy"]
                a[f"{m}_err"] += s + d + n_ins
                a[f"{m}_del"] += d
                a[f"{m}_sub"] += s
                a[f"{m}_ins"] += n_ins
                a[f"{m}_best16"] += r[m]["best16"]
                a[f"{m}_signal"] += r[m]["better_share"] > 0
                a[f"{m}_better_share"] += r[m]["better_share"]
                for q in (4, 8, 32, 64):
                    a[f"{m}_best{q}"] += r[m][f"best{q}"]
                for q in (4, 8, 16, 32, 64):
                    a[f"{m}_signal{q}"] += r[m][f"signal{q}"]
    res = {}
    for k, a in acc.items():
        R, C = max(a["ref"], 1), max(a["clips"], 1)
        res[k] = {"clips": int(a["clips"]),
                  **{f"{m}_{q}": a[f"{m}_{q}"] / R for m in ("sft", "rl") for q in ("err", "del", "sub", "ins", "best16")},
                  **{f"{m}_signal_clips": a[f"{m}_signal"] / C for m in ("sft", "rl")},
                  **{f"{m}_better_share": a[f"{m}_better_share"] / C for m in ("sft", "rl")},
                  **{f"{m}_best{q}": a[f"{m}_best{q}"] / R for m in ("sft", "rl") for q in (4, 8, 32, 64)},
                  **{f"{m}_signal{q}": a[f"{m}_signal{q}"] / C for m in ("sft", "rl") for q in (4, 8, 16, 32, 64)}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--rl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-clips", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    sft, cfg = load(a.sft, None, device)
    items, _ = T.load_items(argparse.Namespace(**cfg))
    val, train = items["val"], items["train"]
    tr_ids = random.Random(a.seed).sample(range(len(train)), min(a.train_clips, len(train)))
    res = {"val": {"sft": run(sft, val, range(len(val)), device, a.seed)},
           "train": {"sft": run(sft, train, tr_ids, device, a.seed)}}
    del sft
    torch.cuda.empty_cache()
    rl, _ = load(a.sft, a.rl, device)
    res["val"]["rl"] = run(rl, val, range(len(val)), device, a.seed)
    res["train"]["rl"] = run(rl, train, tr_ids, device, a.seed)

    meta = E.clip_meta()
    path_task = {}  # audio path -> corpus:task (manifest field)
    idx = {}
    for p in sorted(os.listdir(os.path.join(T.DATA, "processed", "index"))):
        for line in open(os.path.join(T.DATA, "processed", "index", p)):
            r = json.loads(line)
            idx[r["clip_id"]] = r["plain_path"]
    for corpus in T.VAL_CORPORA:
        for line in open(os.path.join(T.MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["clip_id"] in idx:
                path_task[idx[r["clip_id"]]] = f"{corpus}:{r.get('task')}"

    summary = {}
    for split, its in (("val", val), ("train", train)):
        rows = {i: {"sft": res[split]["sft"][i], "rl": res[split]["rl"][i]} for i in res[split]["sft"]}

        def keys(i, its=its, rows=rows):
            it = its[i]
            age = meta.get(it["path"], (None, None))[0]
            s, d, n_ins = rows[i]["sft"]["greedy"]
            kind = "correct" if s + d + n_ins == 0 else ("mostly_del" if d > s + n_ins else "mostly_sub" if s >= d + n_ins else "mixed")
            return [f"corpus:{it['corpus']}", f"tag:{it['tag']}", f"task:{path_task.get(it['path'], '?')}",
                    f"age:{E.age_band(age)}", f"len:{E.len_band(len(it['labels']))}", f"sft_error:{kind}"]
        summary[split] = table(rows, keys, its)
        if split == "val":
            with open(os.path.join(a.out, "clips.jsonl"), "w") as f:
                for i, r in rows.items():
                    f.write(json.dumps({"path": its[i]["path"], "corpus": its[i]["corpus"], "tag": its[i]["tag"],
                                        "ref_len": len(its[i]["labels"]), **r}) + "\n")
            os.chmod(os.path.join(a.out, "clips.jsonl"), 0o600)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    md = ["# RL diagnosis", "", f"SFT {a.sft} vs RL {a.rl}. Pooled per-phone rates. best16 = best of 16 samples at T=1. "
          "signal = share of clips where at least one sample beats greedy.", ""]
    for split in ("val", "train"):
        md += [f"## {split}", "", "| cell | clips | SFT | RL | change | SFT del | RL del | SFT best16 | RL best16 | SFT signal | RL signal |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        S = summary[split]
        for k in sorted(S, key=lambda k: (k.split(":")[0], -S[k]["clips"])):
            r = S[k]
            md.append(f"| {k} | {r['clips']} | {r['sft_err']:.3f} | {r['rl_err']:.3f} | {r['rl_err'] - r['sft_err']:+.3f} | "
                      f"{r['sft_del']:.3f} | {r['rl_del']:.3f} | {r['sft_best16']:.3f} | {r['rl_best16']:.3f} | "
                      f"{r['sft_signal_clips']:.2f} | {r['rl_signal_clips']:.2f} |")
        md.append("")
        md += [f"### {split}: more samples (SFT model). PER of best-of-k / share of clips where some sample beats greedy", "",
               "| cell | greedy | k=4 | 8 | 16 | 32 | 64 | signal 4 | 8 | 16 | 32 | 64 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(S, key=lambda k: (k.split(":")[0], -S[k]["clips"])):
            if not (k == "all" or k.startswith("corpus:") or k.startswith("sft_error:")):
                continue
            r = S[k]
            md.append(f"| {k} | {r['sft_err']:.3f} | " + " | ".join(f"{r[f'sft_best{q}']:.3f}" for q in (4, 8, 16, 32, 64)) + " | "
                      + " | ".join(f"{r[f'sft_signal{q}']:.2f}" for q in (4, 8, 16, 32, 64)) + " |")
        md.append("")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

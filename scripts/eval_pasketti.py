"""Score drivendata/pasketti-phonetic on our val split, the same way as our models.

    $PY -I scripts/eval_pasketti.py [--audio plain|denoised]

Same clips, same phone set (pasketti's IPA output goes through phoneset.normalize),
same PER and breakdowns as train_sft.evaluate. Val only; test splits are never read.

Caveat printed with the results: pasketti-phonetic was trained on SPROUT, Arizona (ACAD),
PERCEPT-R and PERCEPT-GFTA, probably including some of our val children. On those corpora
its numbers are optimistic. Cummings, Penney, Preston and Providence are unseen by it.

Writes RUNS/pasketti-phonetic/eval_val_<audio>.json and hyps_val_<audio>.jsonl (stays on Quest).
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402
from phoneset import UnknownSymbol, normalize  # noqa: E402

MODEL_ID = "drivendata/pasketti-phonetic"
IN_PASKETTI_TRAIN = {"SPROUT", "ACAD", "PERCEPT-R", "PERCEPT-GFTA"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RUNS", "pasketti-phonetic")


def to_tokens(text, unknown):
    toks = []
    for w in text.split():
        try:
            words, _ = normalize(w)
        except UnknownSymbol as e:
            unknown[str(e)] += 1
            continue
        toks += [T.TOK[p] for ww in words for p in ww]
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="plain", choices=["plain", "denoised"])
    ap.add_argument("--batch-seconds", type=float, default=200)
    a = ap.parse_args()
    from transformers import Wav2Vec2Processor, WavLMForCTC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = WavLMForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()  # fp32 only (card)

    items, _ = T.load_items(argparse.Namespace(splits="v2", audio=a.audio))
    val = items["val"]
    order = sorted(range(len(val)), key=lambda i: val[i]["dur"])
    batches, cur, longest = [], [], 0.0
    for i in order:
        if cur and max(longest, val[i]["dur"]) * (len(cur) + 1) > a.batch_seconds:
            batches.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, val[i]["dur"])
    batches.append(cur)

    acc = collections.defaultdict(lambda: np.zeros(6))  # ref, sub, del, ins, empty, clips
    unknown = collections.Counter()
    hyp_len = ref_len = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    hyp_f = open(os.path.join(OUT_DIR, f"hyps_val_{a.audio}.jsonl"), "w")
    t0 = time.time()
    for b, idx in enumerate(batches):
        wavs = [sf.read(val[i]["path"], dtype="float32")[0] for i in idx]
        inp = proc(wavs, sampling_rate=16000, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model(**inp).logits.argmax(-1)
        n = model._get_feat_extract_output_lengths(inp["attention_mask"].sum(-1))
        ids[torch.arange(ids.shape[1], device=device) >= n[:, None]] = proc.tokenizer.pad_token_id
        for i, text in zip(idx, proc.batch_decode(ids.cpu())):
            it = val[i]
            h = to_tokens(text, unknown)
            s, de, ins = T.edit_ops(it["labels"], h)
            seen = "seen" if it["corpus"] in IN_PASKETTI_TRAIN else "unseen"
            for key in (it["tag"], f"corpus_{it['corpus']}", f"pasketti_{seen}", "all"):
                acc[key] += (len(it["labels"]), s, de, ins, len(h) == 0, 1)
            hyp_len += len(h)
            ref_len += len(it["labels"])
            hyp_f.write(json.dumps({"path": it["path"], "raw": text, "hyp": [T.VOCAB[t] for t in h]}) + "\n")
        if b % 100 == 0:
            print(f"batch {b}/{len(batches)}, {time.time() - t0:.0f}s", flush=True)
    hyp_f.close()

    m = {"model": MODEL_ID, "audio": a.audio, "n_val_clips": len(val), "val/len_ratio": hyp_len / ref_len,
         "unknown_symbols": dict(unknown)}
    for key, (r, s, de, ins, emp, k) in acc.items():
        m[f"val/per/{key}"] = (s + de + ins) / max(r, 1)
        m[f"val/empty_rate/{key}"] = emp / max(k, 1)
        if key == "all":
            m.update({"val/sub_rate": s / r, "val/del_rate": de / r, "val/ins_rate": ins / r})
    present = [t for t in T.SCORE_TAGS if f"val/per/{t}" in m]
    m["val/select_score"] = float(np.mean([m[f"val/per/{t}"] for t in present]))
    out = os.path.join(OUT_DIR, f"eval_val_{a.audio}.json")
    json.dump(m, open(out, "w"), indent=1)

    print(f"pasketti-phonetic on val ({a.audio} audio), {len(val)} clips")
    print(f"select_score {m['val/select_score']:.4f}  PER all {m['val/per/all']:.4f}  sub {m['val/sub_rate']:.3f} "
          f"del {m['val/del_rate']:.3f} ins {m['val/ins_rate']:.3f}  len_ratio {m['val/len_ratio']:.3f}")
    for k in sorted(k for k in m if k.startswith("val/per/") and k != "val/per/all"):
        print(f"  {k[8:]:28s} PER {m[k]:.3f}")
    print("unknown symbols (words dropped):", dict(unknown.most_common(20)))
    print("NOTE: pasketti trained on SPROUT, ACAD, PERCEPT-R, PERCEPT-GFTA -> optimistic there")
    print("wrote", out)


if __name__ == "__main__":
    main()

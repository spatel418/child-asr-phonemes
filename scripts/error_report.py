"""Where do the val errors come from? Val split only (test splits are never read).

Step 1 (GPU): save our model's val outputs (greedy and beam top-1), keyed by audio path. Stays on Quest.
    $PY -I scripts/error_report.py --decode RUNS/sft-r16-plain-word-s0
Step 2 (CPU): breakdowns, ours next to pasketti-phonetic on the same clips.
    $PY -I scripts/error_report.py --report RUNS/sft-r16-plain-word-s0 --out RUNS/error-report-sft100

Breakdowns: corpus x group, age (years), sex, reference length, per child (how concentrated the errors are),
plus substitution / deletion / insertion rates and the most common phone confusions.
PER here = total edits / total reference phones in the cell (pooled), not the 5-group select score.
Output: <out>/report.json and <out>/report.md (counts and phone symbols only, no IDs or transcripts).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbest_ceiling as NB  # noqa: E402
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402

PASKETTI = "RUNS/pasketti-phonetic/hyps_val_plain.jsonl"


@torch.no_grad()
def decode(run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _, cfg, _ = G.load_sft(run_dir, device)
    model.eval()
    val = T.load_items(argparse.Namespace(**cfg))[0]["val"]
    ds = T.Clips(val)
    out = open(os.path.join(run_dir, "hyps_val_ours.jsonl"), "w")
    for b in G.batches_by_seconds(range(len(val)), val, 200.0):
        x, mask = G.load_batch(ds, b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, n = G.frame_logits(model, x, mask)
        logits = logits.float()
        g, _ = T.greedy(logits, n.tolist())
        lp = torch.log_softmax(logits, -1).cpu().numpy()
        for k, i in enumerate(b):
            beam = NB.prefix_beam(lp[k, : int(n[k])], beam=16)
            out.write(json.dumps({"path": val[i]["path"], "greedy": [T.VOCAB[t] for t in g[k]],
                                  "beam1": [T.VOCAB[t] for t in beam[0][0]] if beam else []}, ensure_ascii=False) + "\n")
    out.close()


def clip_meta():
    """audio path -> (age_months, sex) from the phonetic manifests (val children only are used)."""
    path_of = {}
    for p in sorted(os.listdir(os.path.join(T.DATA, "processed", "index"))):
        for line in open(os.path.join(T.DATA, "processed", "index", p)):
            r = json.loads(line)
            path_of[r["clip_id"]] = r["plain_path"]
    meta = {}
    for corpus in T.VAL_CORPORA:
        for line in open(os.path.join(T.MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["clip_id"] in path_of:
                meta[path_of[r["clip_id"]]] = (r.get("age_months"), r.get("sex"))
    return meta


def age_band(m):
    if m is None:
        return "unknown"
    y = int(m // 12)
    return "<3" if y < 3 else ("9+" if y >= 9 else str(y))


def len_band(n):
    return "1-2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


class Acc:
    def __init__(self):
        self.v = collections.defaultdict(lambda: np.zeros(7))  # ref, sub, del, ins, clips, empty, children-set size later
        self.kids = collections.defaultdict(set)

    def add(self, key, ref, ops, empty, child):
        self.v[key] += (ref, ops[0], ops[1], ops[2], 1, empty, 0)
        self.kids[key].add(child)


def ops(ref, hyp):
    s = d = i = 0
    for op in Levenshtein.editops(ref, hyp):
        s += op.tag == "replace"
        d += op.tag == "delete"
        i += op.tag == "insert"
    return s, d, i


def report(run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    val = T.load_items(argparse.Namespace(**cfg))[0]["val"]
    train_corpora = set(cfg.get("train_corpora") or T.TRAIN_CORPORA)
    ours = {r["path"]: r for r in map(json.loads, open(os.path.join(run_dir, "hyps_val_ours.jsonl"), encoding="utf-8"))}
    pk = {r["path"]: r["hyp"] for r in map(json.loads, open(PASKETTI, encoding="utf-8"))}
    meta = clip_meta()
    systems = {"ours_greedy": lambda p: ours[p]["greedy"], "ours_beam1": lambda p: ours[p]["beam1"],
               "pasketti": lambda p: pk[p]}
    acc = {s: Acc() for s in systems}
    conf = collections.Counter()
    dels, ins = collections.Counter(), collections.Counter()
    conf_c = collections.defaultdict(collections.Counter)
    child_err = collections.defaultdict(lambda: np.zeros(2))
    missing = collections.Counter()
    for it in val:
        p = it["path"]
        if p not in ours or p not in pk:
            missing["ours" if p not in ours else "pasketti"] += 1
            continue
        ref = [T.VOCAB[t] for t in it["labels"]]
        age, sex = meta.get(p, (None, None))
        cell = f"{it['corpus']} | {it['group']}"
        keys = ["all", f"cell:{cell}", f"corpus:{it['corpus']}", f"group:{it['group']}",
                f"age:{age_band(age)}", f"agegroup:{age_band(age)} | {it['group']}", f"sex:{sex}",
                f"len:{len_band(len(ref))}", f"tag:{it['tag']}"]
        for s, f in systems.items():
            h = f(p)
            o = ops(ref, h)
            for k in keys:
                acc[s].add(k, len(ref), o, len(h) == 0, it["child"])
            if s == "ours_greedy":
                child_err[it["child"]] += (sum(o), len(ref))
                for op in Levenshtein.editops(ref, h):
                    if op.tag == "replace":
                        conf[(ref[op.src_pos], h[op.dest_pos])] += 1
                        conf_c[it["corpus"]][(ref[op.src_pos], h[op.dest_pos])] += 1
                    elif op.tag == "delete":
                        dels[ref[op.src_pos]] += 1
                    else:
                        ins[h[op.dest_pos]] += 1

    tot_err = acc["ours_greedy"].v["all"][1:4].sum()
    rows = {}
    for k in acc["ours_greedy"].v:
        row = {}
        for s in systems:
            r, su, de, i_, n, emp, _ = acc[s].v[k]
            row[s] = {"per": (su + de + i_) / max(r, 1), "sub": su / max(r, 1), "del": de / max(r, 1),
                      "ins": i_ / max(r, 1), "empty": emp / max(n, 1)}
        r, su, de, i_, n, _, _ = acc["ours_greedy"].v[k]
        row.update({"clips": int(n), "children": len(acc["ours_greedy"].kids[k]), "ref_phones": int(r),
                    "share_of_errors": float((su + de + i_) / max(tot_err, 1))})
        rows[k] = row

    ce = np.array(sorted([e / max(r, 1) for e, r in child_err.values()]))
    err_sorted = sorted(child_err.values(), key=lambda v: -v[0] / max(v[1], 1))
    tot = sum(v[0] for v in err_sorted)
    top10 = sum(v[0] for v in err_sorted[: max(1, len(err_sorted) // 10)]) / max(tot, 1)
    children = {"n": len(ce), "per_median": float(np.median(ce)), "per_p90": float(np.quantile(ce, 0.9)),
                "share_children_per_over_0.5": float((ce > 0.5).mean()), "worst_10pct_children_share_of_errors": float(top10)}
    res = {"run": run_dir, "pasketti": PASKETTI, "train_corpora": sorted(train_corpora), "missing": dict(missing),
           "rows": rows, "children": children,
           "top_confusions": [[a, b, c] for (a, b), c in conf.most_common(30)],
           "top_deleted": dels.most_common(20), "top_inserted": ins.most_common(20),
           "top_confusions_by_corpus": {c: [[a, b, n] for (a, b), n in v.most_common(10)] for c, v in conf_c.items()}}
    json.dump(res, open(os.path.join(out_dir, "report.json"), "w"), indent=1, ensure_ascii=False)

    md = [f"# Val error report: {run_dir}", "", f"Pooled PER (edits / ref phones). Missing clips: {dict(missing)}.", ""]
    for title, pre in (("Corpus x group", "cell:"), ("Corpus", "corpus:"), ("Group", "group:"), ("Age (years)", "age:"),
                       ("Age x group", "agegroup:"), ("Sex", "sex:"), ("Reference length (phones)", "len:"), ("Score tag", "tag:")):
        md += [f"## {title}", "", "| cell | in train | clips | kids | ours | beam | pasketti | sub | del | ins | empty | share of errors |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ks = [k for k in rows if k.startswith(pre)]
        for k in sorted(ks, key=lambda k: -rows[k]["share_of_errors"]):
            r, o = rows[k], rows[k]["ours_greedy"]
            name = k[len(pre):]
            intr = ("yes" if name.split(" | ")[0] in train_corpora else "no") if pre in ("cell:", "corpus:") else ""
            md.append(f"| {name} | {intr} | {r['clips']} | {r['children']} | {o['per']:.3f} | {r['ours_beam1']['per']:.3f} | "
                      f"{r['pasketti']['per']:.3f} | {o['sub']:.3f} | {o['del']:.3f} | {o['ins']:.3f} | {o['empty']:.3f} | "
                      f"{r['share_of_errors']:.1%} |")
        md.append("")
    a = rows["all"]
    md += [f"All: ours {a['ours_greedy']['per']:.3f}, beam {a['ours_beam1']['per']:.3f}, pasketti {a['pasketti']['per']:.3f}.", "",
           "## Per child (ours, greedy)", "", json.dumps(children), "",
           "## Top confusions (ref -> ours)", "", ", ".join(f"{x}->{y} {c}" for x, y, c in res["top_confusions"]), "",
           "## Most deleted", "", ", ".join(f"{x} {c}" for x, c in res["top_deleted"]), "",
           "## Most inserted", "", ", ".join(f"{x} {c}" for x, c in res["top_inserted"]), ""]
    for c, v in sorted(res["top_confusions_by_corpus"].items()):
        md.append(f"- {c}: " + ", ".join(f"{x}->{y} {n}" for x, y, n in v))
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.decode:
        decode(a.decode)
    if a.report:
        report(a.report, a.out or os.path.join(a.report, "error_report"))


if __name__ == "__main__":
    main()

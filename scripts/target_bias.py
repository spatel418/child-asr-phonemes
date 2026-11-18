"""Target bias and error detection on val (test splits are never read). CPU only; reads saved model outputs.

The clinical question: where the child's production differs from the target word, does the model hear what the
child said, or does it "hear" the target (target bias)? PER treats those few positions like any other phone.

For each val clip with a word transcript whose words are all in CMUdict:
  target T = dictionary pronunciation of the words (per word, the variant closest to the human IPA)
  human  H = the clinician / transcriber IPA (our labels)
  model  M = each system's greedy output
Consonants only (clinical PCC practice). Typical American variants count as the target: t/d -> [ɾ], t -> [ʔ].
At every target consonant (T aligned to H and to M by edit distance):
  child error     H differs from T
  detected        child error and M also differs from T            (recall of child errors)
  target bias     child error and M equals T  (= 1 - detected)       the model "corrects" the child
  exact           child error and M equals H (same substitute, or both deleted)
  false alarm     H equals T but M differs                          (rate among correct consonants)
Per child: PCC (percent consonants correct) from H vs from M, their correlation, and SSD-vs-typical AUC.
Breakdowns: corpus, group, age, sex, consonant class, child error kind, and for SPROUT: race, Hispanic/Latino,
income group, site. Output: <out>/report.json + report.md (counts and rates only; stays on Quest).

    $PY -I scripts/target_bias.py --out RUNS/target-bias \
        --sys SFT-100=RUNS/sft-r16-plain-word-s0 --sys age2=RUNS/sft-r16-plain-word-s0-prov2 ...
    (a --sys value is a run dir with hyps_val_ours.jsonl, or 'pasketti')
"""
import argparse
import collections
import csv
import itertools
import json
import os
import re
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402
from wordset import _BRACKET, _BULLET, _finish, normalize_plain  # noqa: E402

CMUDICT = os.path.join(os.path.dirname(T.DATA.rstrip("/")), "lexicon", "cmudict", "cmudict.dict")
PASKETTI = "RUNS/pasketti-phonetic/hyps_val_plain.jsonl"
SPROUT_TSV = os.path.join(T.DATA, "raw", "SPROUT_v2", "participants.tsv")
ARPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AH0": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɚ", "EY": "e",
        "IH": "ɪ", "IY": "i", "OW": "o", "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
        "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
        "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
CLASS = {**{c: "stop" for c in ["p", "b", "t", "d", "k", "ɡ"]}, **{c: "fricative" for c in ["f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"]},
         **{c: "affricate" for c in ["tʃ", "dʒ"]}, **{c: "nasal" for c in ["m", "n", "ŋ"]},
         **{c: "liquid" for c in ["l", "ɹ"]}, **{c: "glide" for c in ["w", "j"]}}
RACE = {"3": "Black", "6": "White"}  # others pooled (small groups); multiple codes = "multiple"


def cid(path):
    return os.path.splitext(os.path.basename(path))[0]


def ok(target, real):
    """Does a realization count as the target consonant?"""
    return real == target or (target in ("t", "d") and real == "ɾ") or (target == "t" and real == "ʔ")


def load_lexicon():
    lex = collections.defaultdict(list)
    for line in open(CMUDICT, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        w, *ph = line.split()
        w = re.sub(r"\(\d+\)$", "", w)
        out = []
        for p in ph:
            base = re.sub(r"\d", "", p)
            out.append(ARPA["AH0"] if p == "AH0" else ARPA[base])
        if out not in lex[w]:
            lex[w].append(out)
    return lex


def chat_target_words(main):
    """Intended words from a CHAT main tier: like wordset.normalize_chat, but '(be)cause' -> 'because'."""
    flags = set()
    s = _BULLET.sub(" ", main)
    s = re.sub(r"^\*[A-Z]+:\s*", "", s)
    s = _BRACKET.sub(" ", s).replace("<", " ").replace(">", " ")
    words = []
    for tok in s.split():
        lo = tok.lower()
        if lo.startswith(("xxx", "yyy", "www")):
            return None
        if tok.startswith(("&", "0", "+")) or re.fullmatch(r"\(\.+\)", tok) or re.fullmatch(r"[.,!?;‡„]+", tok):
            continue
        if "@" in tok:  # child form / babble / letters: no dictionary target
            return None
        words.append(tok.replace("(", "").replace(")", "").replace(":", "").replace("^", ""))
    w, flags = _finish(words, flags)
    return None if flags else w.split()


def target_for(words, lex, human):
    vars_ = [lex[w] for w in words]
    if any(not v for v in vars_):
        return None
    combos = itertools.islice(itertools.product(*vars_), 64)
    best = min(combos, key=lambda c: Levenshtein.distance([p for w in c for p in w], human))
    return [p for w in best for p in w]


def align(src, dst):
    """For each src position: the aligned dst token, or None if deleted."""
    out, i, j = [None] * len(src), 0, 0
    for op in Levenshtein.editops(src, dst):
        while i < op.src_pos and j < op.dest_pos:
            out[i] = dst[j]
            i, j = i + 1, j + 1
        if op.tag == "replace":
            out[i] = dst[j]
            i, j = i + 1, j + 1
        elif op.tag == "delete":
            i += 1
        else:
            j += 1
    while i < len(src) and j < len(dst):
        out[i] = dst[j]
        i, j = i + 1, j + 1
    return out


def age_band(m):
    if m is None:
        return "unknown"
    y = int(m // 12)
    return "<3" if y < 3 else "9+" if y >= 9 else str(y)


def sprout_demo():
    out = {}
    for r in csv.DictReader(open(SPROUT_TSV), delimiter="\t"):
        codes = [c.strip() for c in r["ss_child_race"].split(",") if c.strip()]
        race = "unknown" if not codes else "multiple" if len(codes) > 1 else RACE.get(codes[0], "other")
        hisp = {"1.0": "yes", "2.0": "no"}.get(r["ss_child_hisp_latx"], "unknown")
        inc = {"Low Income": "low", "Not Low income": "not low"}.get(r["ss_demographic_groups"], "unknown")
        out["SPROUT_" + r["participant_id"].replace("sub-", "")] = {"race": race, "hispanic": hisp, "income": inc,
                                                                     "site": r["data_collection_site"]}
    return out


def auc(scores, labels):
    """P(score of a positive > score of a negative); positives = SSD. Lower PCC = more SSD-like, so pass -PCC."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sys", action="append", required=True, help="name=run_dir or name=pasketti")
    ap.add_argument("--config", default="RUNS/sft-r16-plain-word-s0-prov2/config.json", help="for the val clip list")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lex = load_lexicon()
    cfg = argparse.Namespace(**json.load(open(a.config)))
    val = T.load_items(cfg)[0]["val"]
    by_path = {it["path"]: it for it in val}
    # manifest fields (words, age, sex) for val clips, keyed by audio path
    idx = {}
    for p in sorted(os.listdir(os.path.join(T.DATA, "processed", "index"))):
        for line in open(os.path.join(T.DATA, "processed", "index", p)):
            r = json.loads(line)
            idx[r["clip_id"]] = r[f"{cfg.audio}_path"]
    meta = {}
    for corpus in T.VAL_CORPORA:
        for line in open(os.path.join(T.MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            p = idx.get(r["clip_id"])
            if p in by_path:
                meta[p] = r
    systems = {}
    for s in a.sys:
        name, src = s.split("=", 1)
        if src == "pasketti":
            systems[name] = {cid(r["path"]): r["hyp"] for r in map(json.loads, open(PASKETTI, encoding="utf-8"))}
        else:  # keyed by clip id, so plain- and denoised-audio runs line up
            systems[name] = {cid(r["path"]): r["greedy"] for r in map(json.loads, open(os.path.join(src, "hyps_val_ours.jsonl"), encoding="utf-8"))}
    demo = sprout_demo()

    cov = collections.Counter()
    acc = {n: collections.defaultdict(lambda: np.zeros(6)) for n in systems}  # n_cons, child_err, detected, exact, n_correct, false_alarm
    kid = {n: collections.defaultdict(lambda: np.zeros(3)) for n in systems}   # n_cons, human_ok, model_ok
    kid_group = {}
    for p, it in by_path.items():
        r = meta.get(p)
        cov[(it["corpus"], "clips")] += 1
        if r is None or any(cid(p) not in h for h in systems.values()):
            continue
        words = normalize_plain(r["orthographic"])[0].split() if it["corpus"] == "SPROUT" else chat_target_words(r["orthographic"] or "")
        if not words:
            continue
        human = [T.VOCAB[t] for t in it["labels"]]
        tgt = target_for(words, lex, human)
        if tgt is None:
            continue
        cov[(it["corpus"], "with_target")] += 1
        h_al = align(tgt, human)
        d = demo.get(it["child"], {}) if it["corpus"] == "SPROUT" else {}
        base = ["all", f"corpus:{it['corpus']}", f"group:{it['group']}", f"cell:{it['corpus']} | {it['group']}",
                f"age:{age_band(r.get('age_months'))}", f"sex:{r.get('sex')}"]
        if d:
            base += [f"sprout_race:{d['race']}", f"sprout_hispanic:{d['hispanic']}", f"sprout_income:{d['income']}",
                     f"sprout_site:{d['site']}", f"sprout_race_group:{d['race']} | {it['group']}"]
        kid_group[it["child"]] = (it["corpus"], it["group"], d.get("race"))
        for name, hyps in systems.items():
            m_al = align(tgt, list(hyps[cid(p)]))
            for j, t in enumerate(tgt):
                if t not in CLASS:
                    continue
                child_err = not (h_al[j] is not None and ok(t, h_al[j]))
                model_dev = not (m_al[j] is not None and ok(t, m_al[j]))
                kind = "correct" if not child_err else ("deletion" if h_al[j] is None else "substitution")
                row = (1, child_err, child_err and model_dev, child_err and m_al[j] == h_al[j],
                       not child_err, (not child_err) and model_dev)
                for k in base + [f"class:{CLASS[t]}", f"child_error_kind:{kind}", f"class_group:{CLASS[t]} | {it['group']}"]:
                    acc[name][k] += row
                kid[name][it["child"]] += (1, not child_err, not model_dev)

    res = {"coverage": {c: {"clips": cov[(c, "clips")], "with_target": cov[(c, "with_target")]} for c in sorted({c for c, _ in cov})},
           "systems": list(systems), "cells": {}, "children": {}}
    for name in systems:
        for k, (n, ce, det, ex, nc, fa) in acc[name].items():
            res["cells"].setdefault(k, {})[name] = {
                "consonants": int(n), "child_errors": int(ce), "human_pcc": 1 - ce / max(n, 1),
                "detected": det / max(ce, 1), "target_bias": 1 - det / max(ce, 1), "exact": ex / max(ce, 1),
                "false_alarm": fa / max(nc, 1), "precision": det / max(det + fa, 1)}
        # per child: PCC human vs model
        ks = [k for k in kid[name] if kid[name][k][0] >= 20]
        hp = np.array([kid[name][k][1] / kid[name][k][0] for k in ks])
        mp = np.array([kid[name][k][2] / kid[name][k][0] for k in ks])
        out = {"children": len(ks), "pcc_corr": float(np.corrcoef(hp, mp)[0, 1]), "pcc_mean_abs_diff": float(np.mean(np.abs(hp - mp))),
               "pcc_model_minus_human_mean": float(np.mean(mp - hp))}
        for corp in sorted({kid_group[k][0] for k in ks}):
            sub = [i for i, k in enumerate(ks) if kid_group[k][0] == corp]
            labs = [kid_group[ks[i]][1] == "SSD" for i in sub]
            grp_ok = [kid_group[ks[i]][1] in ("SSD", "typical") for i in sub]
            s_idx = [i for i, g in zip(sub, grp_ok) if g]
            out[f"{corp}"] = {"children": len(sub),
                              "pcc_corr": float(np.corrcoef(hp[sub], mp[sub])[0, 1]) if len(sub) > 2 else None,
                              "auc_ssd_human": auc([-hp[i] for i in s_idx], [kid_group[ks[i]][1] == "SSD" for i in s_idx]),
                              "auc_ssd_model": auc([-mp[i] for i in s_idx], [kid_group[ks[i]][1] == "SSD" for i in s_idx])}
            del labs
        res["children"][name] = out
    json.dump(res, open(os.path.join(a.out, "report.json"), "w"), indent=1, ensure_ascii=False)

    names = list(systems)
    md = ["# Target bias and error detection (val, consonants)", "",
          "child err = share of target consonants the child got wrong (by the human IPA). detected = share of those the "
          "model also marks as wrong (higher is better). bias = model wrote the target instead (lower is better). "
          "exact = model wrote the same thing as the human. false alarm = model marks a correct consonant wrong.", "",
          "Coverage: " + ", ".join(f"{c} {v['with_target']}/{v['clips']}" for c, v in res["coverage"].items()), ""]
    groups = [("Overall and corpus", ("all", "corpus:")), ("Group", ("group:",)), ("Corpus x group", ("cell:",)),
              ("Age", ("age:",)), ("Sex", ("sex:",)), ("Consonant class", ("class:",)), ("Class x group", ("class_group:",)),
              ("Child error kind", ("child_error_kind:",)), ("SPROUT race", ("sprout_race:",)),
              ("SPROUT race x group", ("sprout_race_group:",)), ("SPROUT Hispanic/Latino", ("sprout_hispanic:",)),
              ("SPROUT income", ("sprout_income:",)), ("SPROUT site", ("sprout_site:",))]
    for title, pres in groups:
        ks = [k for k in res["cells"] if k == "all" and "all" in pres or any(k.startswith(p) for p in pres if p != "all")]
        if not ks:
            continue
        md += [f"## {title}", "", "| cell | cons. | child err | " + " | ".join(f"{n} det / bias / exact / FA" for n in names) + " |",
               "|---|---|---|" + "---|" * len(names)]
        for k in sorted(ks, key=lambda k: -res["cells"][k][names[0]]["consonants"]):
            c0 = res["cells"][k][names[0]]
            md.append(f"| {k} | {c0['consonants']} | {1 - c0['human_pcc']:.3f} | " + " | ".join(
                f"{res['cells'][k][n]['detected']:.2f} / {res['cells'][k][n]['target_bias']:.2f} / {res['cells'][k][n]['exact']:.2f} / "
                f"{res['cells'][k][n]['false_alarm']:.3f}" for n in names) + " |")
        md.append("")
    md += ["## Per child PCC (children with >= 20 target consonants)", ""]
    for n in names:
        md.append(f"- {n}: " + json.dumps(res["children"][n]))
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

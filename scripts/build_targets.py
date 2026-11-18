"""Dictionary targets for train and val clips, for the false-alarm RL reward (train_grpo.py --reward per_fa).

Same method as target_bias.py: CMUdict pronunciation of the transcript words, per-utterance variant closest to
the human IPA. For each target position: is it a consonant the child said correctly (per the human IPA)?
Those are the positions a model output must not drop or change (a "false alarm").
Clips without a dictionary target (babble, child forms, unknown words, xxx) are left out: they keep the plain
PER reward. Test splits are never read.

    $PY -I scripts/build_targets.py
Writes <data>/targets/v1/targets.jsonl ({clip id, target phones, correct mask}) + summary.json, mode 640.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import target_bias as B  # noqa: E402
import train_sft as T  # noqa: E402
from wordset import normalize_plain  # noqa: E402

OUT = os.path.join(T.DATA, "targets", "v1")
CONFIG = "RUNS/sft-r16-plain-word-s0-prov2/config.json"


def main():
    cfg = argparse.Namespace(**json.load(open(CONFIG)))
    items = T.load_items(cfg)[0]
    lex = B.load_lexicon()
    idx = {}
    for p in sorted(os.listdir(os.path.join(T.DATA, "processed", "index"))):
        for line in open(os.path.join(T.DATA, "processed", "index", p)):
            r = json.loads(line)
            idx[r["clip_id"]] = r
    words_of = {}
    for corpus in T.VAL_CORPORA:
        for line in open(os.path.join(T.MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["clip_id"] in idx:
                words_of[r["clip_id"]] = (corpus, r["orthographic"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "targets.jsonl")
    n = collections.Counter()
    with open(path, "w") as f:
        for split in ("train", "val"):
            for it in items[split]:
                c = B.cid(it["path"])
                n[(split, it["corpus"], "clips")] += 1
                if c not in words_of:
                    continue
                corpus, text = words_of[c]
                words = normalize_plain(text)[0].split() if corpus == "SPROUT" else B.chat_target_words(text or "")
                if not words:
                    continue
                human = [T.VOCAB[t] for t in it["labels"]]
                tgt = B.target_for(words, lex, human)
                if tgt is None:
                    continue
                h_al = B.align(tgt, human)
                mask = [t in B.CLASS and h is not None and B.ok(t, h) for t, h in zip(tgt, h_al)]
                n[(split, it["corpus"], "with_target")] += 1
                f.write(json.dumps({"cid": c, "split": split, "target": tgt, "correct": mask}, ensure_ascii=False) + "\n")
    os.chmod(path, 0o640)
    summ = {f"{s} | {c}": {"clips": n[(s, c, "clips")], "with_target": n[(s, c, "with_target")]}
            for s, c, k in sorted(n) if k == "clips"}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

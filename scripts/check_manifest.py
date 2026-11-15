"""Check phonetic manifests before they are used (phone-data skill, "Validation checks").

    $PY -I scripts/check_manifest.py                 # all manifests
    $PY -I scripts/check_manifest.py SPROUT ACAD     # some
    $PY -I scripts/check_manifest.py --audio 300     # also open 300 random clips per corpus

Fails (exit 1) on: missing audio, unmapped phones, missing child_id, a child with
two groups or two sexes, wrong number of sealed children, a clip past the end of
its audio, or a duration that does not match the audio file.
Prints a summary by corpus and group (counts and hours only).
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import OUT, SEALED  # noqa: E402
from phoneset import PHONES, WORD_SEP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*")
    ap.add_argument("--audio", type=int, default=0, help="open this many random clips per corpus")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(OUT, "phonetic", "*.jsonl")))
    if args.corpora:
        files = [f for f in files if os.path.basename(f)[:-6] in args.corpora]
    ok_phones = set(PHONES) | {WORD_SEP}
    errors, warns = collections.Counter(), collections.Counter()
    examples = {}
    child_group, child_sex = collections.defaultdict(set), collections.defaultdict(set)
    summary = collections.defaultdict(lambda: [0, 0.0, set()])
    sealed_seen = set()
    random.seed(0)

    def err(kind, row):
        errors[kind] += 1
        examples.setdefault(kind, row.get("clip_id"))

    for path in files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        for r in rows:
            if not r.get("child_id"):
                err("no_child_id", r)
            if r["source_path"] and not os.path.exists(r["source_path"]):
                err("audio_missing", r)
            if set(r["ipa_normalized"].split()) - ok_phones:
                err("unmapped_phone", r)
            child_group[r["child_id"]].add(r["group"])
            if r["sex"]:
                child_sex[r["child_id"]].add(r["sex"])
            if r["sip_sealed"]:
                sealed_seen.add(r["child_id"])
            s = summary[(r["corpus"], r["group"])]
            s[0] += 1
            s[2].add(r["child_id"])
            if not r["flags"]:
                s[1] += r["duration"] or 0
        if args.audio:
            import soundfile as sf
            for r in random.sample(rows, min(args.audio, len(rows))):
                if not r["source_path"] or not os.path.exists(r["source_path"]):
                    continue
                if r["source_path"].endswith((".m4a", ".mp3")):
                    warns["needs_conversion (m4a/mp3)"] += 1  # converted in preprocessing
                    continue
                try:
                    info = sf.info(r["source_path"])
                except Exception:
                    err("audio_unreadable", r)
                    continue
                if r["start"] is None:
                    if r["duration"] and abs(info.duration - r["duration"]) > 0.05:
                        err("duration_mismatch", r)
                elif r["end"] > info.duration + 0.5:
                    err("clip_past_end_of_audio", r)
                elif r["end"] > info.duration + 0.05:
                    warns["clip_ends_<0.5s_past_audio (cut stops at end)"] += 1

    for c, g in child_group.items():
        if len(g) > 1:
            errors["child_two_groups"] += 1
            examples.setdefault("child_two_groups", f"{c}: {sorted(g)}")
    for c, g in child_sex.items():
        if len(g) > 1:
            errors["child_two_sexes"] += 1
            examples.setdefault("child_two_sexes", c)
    n_sealed = sum(1 for l in open(SEALED) if l.strip())
    if any("SPROUT" in f for f in files) and len(sealed_seen) != n_sealed:
        errors["sealed_count"] += 1
        examples["sealed_count"] = f"found {len(sealed_seen)}, list has {n_sealed}"

    print(f"{'corpus':16}{'group':15}{'children':>9}{'clips':>8}{'usable h':>9}")
    for (c, g), (n, h, kids) in sorted(summary.items()):
        print(f"{c:16}{g:15}{len(kids):>9}{n:>8}{h / 3600:>9.1f}")
    print(f"sealed SPROUT children marked: {len(sealed_seen)} of {n_sealed}")
    for k, v in warns.items():
        print(f"warning: {k}: {v}")
    if errors:
        print("\nFAILED:")
        for k, v in errors.items():
            print(f"  {k}: {v}   e.g. {examples[k]}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()

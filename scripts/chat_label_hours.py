"""Count child utterances and IPA hours in CHAT transcripts.

For each corpus folder (unzipped TalkBank/PhonBank transcripts), counts:
  - child (*CHI) lines
  - child lines with a %pho tier (IPA of what was said) and their total hours
  - child lines with a %mod tier (target pronunciation)

Durations come from the utterance time marks (start_end in ms). Lines without
time marks count toward the totals but add no hours.

Usage:
  python scripts/chat_label_hours.py EXTRACTED_DIR            # one row per corpus folder
  python scripts/chat_label_hours.py EXTRACTED_DIR --by-age   # also split IPA hours by child age (years)
"""
import argparse
import collections
import glob
import os
import re

BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")
AGE = re.compile(r"^@ID:\s*[^|]*\|[^|]*\|CHI\|(\d+);", re.M)


def count_corpus(folder):
    n = n_pho = n_mod = 0
    pho_sec = 0.0
    by_age = collections.Counter()
    files = glob.glob(os.path.join(folder, "**", "*.cha"), recursive=True)
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = AGE.search(text)
        age = int(m.group(1)) if m else None
        for block in re.split(r"\n(?=\*)", text):
            if not block.startswith("*CHI:"):
                continue
            n += 1
            if re.search(r"^%mod:", block, re.M):
                n_mod += 1
            if re.search(r"^%pho:", block, re.M):
                n_pho += 1
                t = BULLET.search(block)
                if t:
                    sec = (int(t.group(2)) - int(t.group(1))) / 1000
                    pho_sec += sec
                    by_age[age] += sec
    return len(files), n, n_pho, pho_sec / 3600, n_mod, by_age


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extracted_dir", help="folder holding one subfolder of .cha files per corpus")
    ap.add_argument("--by-age", action="store_true", help="also print IPA hours per child age")
    args = ap.parse_args()

    print(f"{'corpus':16}{'cha':>7}{'CHI':>9}{'%pho':>9}{'IPA h':>8}{'%mod':>9}")
    for name in sorted(os.listdir(args.extracted_dir)):
        folder = os.path.join(args.extracted_dir, name)
        if not os.path.isdir(folder):
            continue
        n_files, n, n_pho, hours, n_mod, by_age = count_corpus(folder)
        print(f"{name:16}{n_files:>7}{n:>9}{n_pho:>9}{hours:>8.1f}{n_mod:>9}")
        if args.by_age and by_age:
            for age in sorted(by_age, key=lambda a: (a is None, a)):
                label = "?" if age is None else str(age)
                print(f"{'':16}  age {label:>3}: {by_age[age] / 3600:7.1f} h")


if __name__ == "__main__":
    main()

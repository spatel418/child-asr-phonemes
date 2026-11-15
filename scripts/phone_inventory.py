"""Count every symbol used in each corpus's IPA labels.

This is the first "look at the data" step before building one shared phone set.
It counts Unicode characters (so diacritics and tie bars show up on their own),
not phones, so nothing is hidden by a guess about how to split phones.

Sources it understands:
  chat    folder of .cha files: %pho tier of child (*CHI) lines
  sprout  SPROUT v2 transcript JSONs: phonetic_text
  jibo    JIBO jsonl files: phonetic_text
  timit   TIMIT .PHN files: phone labels (TIMIT's own symbols, not IPA)

Usage:
  python scripts/phone_inventory.py --out REPORT.tsv \
      chat:Cummings=/path/to/Cummings sprout:SPROUT=/path/to/transcripts \
      jibo:JIBO=/path/to/jibo_transcripts timit:TIMIT=/path/to/TIMIT

Writes one row per (corpus, symbol): count, share of labels containing it,
code point and Unicode name. Prints a short summary per corpus.
"""
import argparse
import collections
import glob
import json
import os
import re
import unicodedata


def chat_labels(folder):
    for path in glob.glob(os.path.join(folder, "**", "*.cha"), recursive=True):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for block in re.split(r"\n(?=\*)", text):
            if not block.startswith("*CHI:"):
                continue
            m = re.search(r"^%pho:\t?(.*(?:\n\t.*)*)", block, re.M)
            if m:
                yield " ".join(m.group(1).split())


def sprout_labels(folder):
    for path in glob.glob(os.path.join(folder, "**", "*.json"), recursive=True):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("phonetic_text"):
            yield d["phonetic_text"]


def jibo_labels(folder):
    for path in glob.glob(os.path.join(folder, "**", "*.jsonl"), recursive=True):
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("phonetic_text"):
                    yield d["phonetic_text"]


def timit_labels(folder):
    for path in glob.glob(os.path.join(folder, "**", "*.PHN"), recursive=True):
        with open(path, encoding="ascii", errors="replace") as f:
            yield [line.split()[2] for line in f if line.strip()]


READERS = {"chat": chat_labels, "sprout": sprout_labels, "jibo": jibo_labels, "timit": timit_labels}


def describe(sym):
    if len(sym) != 1:  # TIMIT phone names
        return "", "TIMIT phone"
    return f"U+{ord(sym):04X}", unicodedata.name(sym, "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="kind:NAME=folder")
    ap.add_argument("--out", required=True, help="TSV report path")
    args = ap.parse_args()

    rows = []
    for spec in args.sources:
        kind, rest = spec.split(":", 1)
        name, folder = rest.split("=", 1)
        counts, has = collections.Counter(), collections.Counter()
        n_labels = 0
        for label in READERS[kind](folder):
            n_labels += 1
            symbols = label if kind == "timit" else unicodedata.normalize("NFC", label)
            counts.update(symbols)
            has.update(set(symbols))
        print(f"{name:16} labels={n_labels:<8} distinct symbols={len(counts)}")
        for sym, c in counts.most_common():
            cp, uname = describe(sym)
            rows.append((name, sym, c, has[sym] / max(n_labels, 1), cp, uname))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("corpus\tsymbol\tcount\tshare_of_labels\tcodepoint\tname\n")
        for name, sym, c, share, cp, uname in rows:
            shown = repr(sym) if sym.isspace() or unicodedata.category(sym[0]).startswith("M") else sym
            f.write(f"{name}\t{shown}\t{c}\t{share:.4f}\t{cp}\t{uname}\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

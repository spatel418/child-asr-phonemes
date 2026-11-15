"""Word-only manifests (no IPA) for the auxiliary word head.

    $PY -I scripts/build_word_manifest.py            # all: Cameron EllisWeismer OCSC MyST
    $PY -I scripts/build_word_manifest.py MyST

One row per child utterance, flags instead of deletion (like build_manifest.py).
Writes manifests/orthographic/<corpus>.jsonl. Word text rules: scripts/wordset.py.
Not here: JIBO (Test B), SEED (excluded), CSLU Kids (no utterance timing).
"""
import glob
import json
import os
import re
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import BULLET, CHAT_ROOT, CHI_ID, MEDIA, MEDIA_EXT, OUT, RAW, age_months, clip_id  # noqa: E402
from wordset import BLOCKING, VERSION, normalize_chat, normalize_myst  # noqa: E402

MIN_SEC, MAX_SEC = 0.3, 20.0
MAX_CHARS_PER_SEC = 40  # WavLM: 50 frames/s; CTC needs room for blanks between repeats
GROUPS = {"Cameron": lambda p: "typical", "OCSC": lambda p: "typical",
          "EllisWeismer": lambda p: {"LT": "late_talker", "TD": "typical"}.get(p[0], "unknown")}


def _common_flags(r, words):
    f = set()
    d = r["duration"]
    if d is None:
        f.add("no_time")
    elif d < MIN_SEC:
        f.add("too_short")
    elif d > MAX_SEC:
        f.add("too_long")
    elif len(words) / d > MAX_CHARS_PER_SEC:
        f.add("too_fast")
    if not r["source_path"]:
        f.add("no_audio")
    return f


def read_chat_words(corpus):
    root = os.path.join(CHAT_ROOT, corpus)
    media_root = os.path.join(RAW, corpus, "media")
    for cha in sorted(glob.glob(os.path.join(root, "**", "*.cha"), recursive=True)):
        text = open(cha, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(cha, root)
        parts = rel.split(os.sep)
        m = MEDIA.search(text)
        audio = None
        if m:
            for ext in MEDIA_EXT:
                cand = os.path.join(media_root, os.path.dirname(rel), m.group(1) + ext)
                if os.path.exists(cand):
                    audio = cand
                    break
        idm = CHI_ID.search(text)
        age = age_months(idm.group(3)) if idm else None
        sex = {"male": "M", "female": "F"}.get(idm.group(4)) if idm else None
        child = os.path.splitext(parts[-1])[0]
        for block in re.split(r"\n(?=\*)", text):
            if not block.startswith("*CHI:"):
                continue
            main = block.split("\n%")[0]
            t = BULLET.search(main)
            start = end = dur = None
            if t:
                start, end = int(t.group(1)) / 1000, int(t.group(2)) / 1000
                dur = round(end - start, 3)
            words, flags = normalize_chat(main)
            r = {"clip_id": clip_id(corpus, rel, start, end), "corpus": corpus,
                 "child_id": f"{corpus}_{child}", "age_months": age, "sex": sex,
                 "group": GROUPS[corpus](parts), "task": parts[0],
                 "source_path": audio, "start": start, "end": end, "duration": dur,
                 "words": words, "words_original": " ".join(BULLET.sub("", main[5:]).split())}
            r["flags"] = sorted(flags | _common_flags(r, words))
            yield r


def read_myst():
    root = os.path.join(RAW, "MyST", "myst_child_conv_speech", "data")
    for part in ("train", "development", "test"):
        for trn in sorted(glob.glob(os.path.join(root, part, "*", "*", "*.trn"))):
            flac = trn[:-4] + ".flac"
            raw = open(trn, encoding="utf-8", errors="replace").read()
            words, flags = normalize_myst(raw)
            ok = os.path.exists(flac)
            r = {"clip_id": clip_id("MyST", os.path.relpath(flac, root)), "corpus": "MyST",
                 "child_id": f"MyST_{trn.split(os.sep)[-3]}", "age_months": None, "sex": None,
                 "group": "unknown", "task": "science_tutor", "myst_part": part,
                 "source_path": flac if ok else None, "start": None, "end": None,
                 "duration": round(sf.info(flac).duration, 3) if ok else None,
                 "words": words, "words_original": " ".join(raw.split())}
            r["flags"] = sorted(flags | _common_flags(r, words))
            yield r


READERS = {"Cameron": lambda: read_chat_words("Cameron"), "EllisWeismer": lambda: read_chat_words("EllisWeismer"),
           "OCSC": lambda: read_chat_words("OCSC"), "MyST": read_myst}


def usable(r):
    return not (set(r["flags"]) & (BLOCKING | {"no_time", "too_short", "too_long", "too_fast", "no_audio"}))


def main():
    names = sys.argv[1:] or list(READERS)
    out_dir = os.path.join(OUT, "orthographic")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'corpus':14}{'utts':>8}{'children':>9}{'hours':>8}{'usable h':>9}  top flags")
    for name in names:
        path = os.path.join(out_dir, f"{name}.jsonl")
        n, hours, use_h, kids, flagc, odd = 0, 0.0, 0.0, set(), {}, {}
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for r in READERS[name]():
                r["wordset"] = VERSION
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                kids.add(r["child_id"])
                hours += r["duration"] or 0
                if usable(r):
                    use_h += r["duration"]
                for fl in r["flags"]:
                    flagc[fl] = flagc.get(fl, 0) + 1
        os.replace(path + ".tmp", path)
        top = ", ".join(f"{k} {v}" for k, v in sorted(flagc.items(), key=lambda x: -x[1])[:5])
        print(f"{name:14}{n:>8}{len(kids):>9}{hours / 3600:>8.1f}{use_h / 3600:>9.1f}  {top}", flush=True)
    print("written to", out_dir)


if __name__ == "__main__":
    main()

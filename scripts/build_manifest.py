"""Build phonetic manifests: one JSON row per child clip with an IPA label.

    PY=/projects/p32576/simron/envs/pasketti-phonetic/bin/python
    $PY -I scripts/build_manifest.py SPROUT Cummings ACAD PERCEPT-R      # some corpora
    $PY -I scripts/build_manifest.py all                                 # every corpus below

Reads raw data only (read-only). Writes $CHILD_MANIFESTS/phonetic/<corpus>.jsonl
(git-ignored; holds child IDs and transcripts, never commit or upload it).

Row fields:
  clip_id        stable hash of (corpus, source file, start, end)
  corpus, child_id, age_months, sex, group, task
  source_path    audio file; for session corpora the clip is [start, end] of it
  start, end     seconds in source_path (None = whole file)
  duration       seconds
  orthographic   word transcript as given (may be None)
  ipa_original   label exactly as in the source
  ipa_normalized phones from scripts/phoneset.py, words joined by ' | '
  n_phones
  flags          list; training code decides what to skip
  sip_sealed     True for the sealed SIP test children (never train on them)
  phoneset       phone set version used

Child IDs: one per child across corpora (PERCEPT-R and PERCEPT-GFTA share names).
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phoneset import VERSION, UnknownSymbol, normalize, to_string  # noqa: E402


def load_dotenv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
RAW = os.environ.get("CHILD_ASR_RAW", "/projects/p32992/simron/data/child_asr/raw")
CHAT_ROOT = os.environ.get("CHAT_EXTRACTED", "/projects/p32576/simron/scratch/tb_check/extracted")
OUT = os.environ.get("CHILD_MANIFESTS", os.path.join(os.path.dirname(RAW.rstrip("/")), "manifests"))
SEALED = os.environ.get("SIP_SEALED_IDS", "/projects/p32576/simron/sip_v2/protocol/test_ids.txt")

MIN_SEC, MAX_SEC = 0.3, 20.0
INFO_FLAGS = {"sex_fixed", "group_fixed"}  # notes only; the clip is still usable
MAX_PHONES_PER_SEC = 25  # WavLM gives 50 frames/s; CTC needs room for blanks and repeats

# ---------------------------------------------------------------- helpers


def clip_id(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


UNKNOWN = {}  # symbol -> [count, example]; any entry makes the build fail


def label_fields(ipa, duration):
    try:
        words, flags = normalize(ipa if isinstance(ipa, str) else "")  # NaN/None -> empty_label
    except UnknownSymbol as e:
        key = str(e).split(" in ")[0]
        UNKNOWN.setdefault(key, [0, str(e)])[0] += 1
        words, flags = [], {"unknown_symbol"}
    n = sum(len(w) for w in words)
    if duration is not None:
        if duration < MIN_SEC:
            flags.add("too_short")
        if duration > MAX_SEC:
            flags.add("too_long")
        if duration > 0 and n / duration > MAX_PHONES_PER_SEC:
            flags.add("label_too_long")
    return {"ipa_normalized": to_string(words), "n_phones": n}, flags


def age_months(chat_age):
    m = re.match(r"(\d+);(\d*)", chat_age or "")
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2) or 0)


# ---------------------------------------------------------------- SPROUT

SPROUT_GROUP = {"1": "language-only", "2": "SSD", "3": "SSD", "4": "typical", "5": "unknown"}


def read_sprout():
    root = os.path.join(RAW, "SPROUT_v2")
    part = {r["participant_id"]: r for r in csv.DictReader(open(os.path.join(root, "participants.tsv")), delimiter="\t")}
    sealed = {l.strip() for l in open(SEALED) if l.strip()}
    for js in sorted(glob.glob(os.path.join(root, "sub-*", "beh", "*desc-utt*.json"))):
        d = json.load(open(js, encoding="utf-8"))
        cid = os.path.basename(js).split("_")[0].replace("sub-", "")
        p = part.get(cid, {})
        risk = (p.get("at_risk_decision") or "").split(".")[0]
        age = p.get("ss_child_age")
        dur = d.get("audio_duration_sec")
        lab, flags = label_fields(d.get("phonetic_text"), dur)
        if d.get("should_scrub"):
            flags.add("scrub")
        if d.get("tier") != "CHI":
            flags.add("not_child")
        if "desc-uttX" in js:
            flags.add("unintelligible")
        wav = js[:-5] + ".wav"
        yield {
            "clip_id": clip_id("SPROUT", wav),
            "corpus": "SPROUT",
            "child_id": f"SPROUT_{cid}",
            "age_months": float(age) if age else None,
            "sex": {"1.0": "M", "2.0": "F"}.get(p.get("ss_child_biosex", "")),
            "group": SPROUT_GROUP.get(risk, "unknown"),
            "task": os.path.basename(js).split("_task-")[1].split("_")[0],
            "source_path": wav, "start": None, "end": None, "duration": dur,
            "orthographic": d.get("orthographic_text_normalized") if isinstance(d.get("orthographic_text_normalized"), str) else None,
            "ipa_original": d.get("phonetic_text") if isinstance(d.get("phonetic_text"), str) else None,
            **lab, "flags": sorted(flags), "sip_sealed": cid in sealed,
        }


# ---------------------------------------------------------------- TalkBank / PhonBank (CHAT)

BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")
CHI_ID = re.compile(r"^@ID:\s*([^|]*)\|([^|]*)\|CHI\|([^|]*)\|([^|]*)\|([^|]*)\|", re.M)
MEDIA = re.compile(r"^@Media:\s*([^,\s]+)", re.M)
MEDIA_EXT = [".wav", ".flac", ".m4a", ".mp3"]


def _child_from_path(corpus, rel):
    parts = rel.split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    if corpus == "ACAD":
        return parts[0]
    if corpus in ("Cummings", "Providence", "McAllister", "PERCEPT-R"):
        return parts[1]  # CHAT/<child>/<session>.cha
    return stem          # PERCEPT-GFTA, Penney, Preston, TorringtonEaton: one file per child/task


def _task(corpus, rel):
    parts = rel.split(os.sep)
    if corpus == "Penney" or corpus == "TorringtonEaton":
        return parts[1]
    if corpus == "PERCEPT-R":
        return os.path.splitext(parts[-1])[0].split("-")[0]
    return {"PERCEPT-GFTA": "GFTA", "Cummings": "tests", "Preston": "naming",
            "ACAD": "elicited", "Providence": "free_play", "McAllister": "pictures"}.get(corpus)


def _percept_groups():
    """PERCEPT-GFTA folders say each child's group; PERCEPT-R uses the same names."""
    out = {}
    for cha in glob.glob(os.path.join(CHAT_ROOT, "PERCEPT-GFTA", "**", "*.cha"), recursive=True):
        folder = cha.split(os.sep)[-2]
        g = "typical" if folder.startswith("TD") else "SSD" if folder.startswith("Suspected") else "unknown"
        out[os.path.splitext(os.path.basename(cha))[0]] = g
    return out


FIXED_GROUP = {"ACAD": "typical", "Penney": "typical", "Providence": "typical",
               "Cummings": "SSD", "Preston": "SSD", "McAllister": "SSD"}
TE_GROUP = {"SSD": "SSD", "typical_speech": "typical"}


def read_chat(corpus):
    root = os.path.join(CHAT_ROOT, corpus)
    media_root = os.path.join(RAW, corpus, "media")
    pgroups = _percept_groups() if corpus.startswith("PERCEPT") else {}
    for cha in sorted(glob.glob(os.path.join(root, "**", "*.cha"), recursive=True)):
        text = open(cha, encoding="utf-8", errors="replace").read()
        if "%pho:" not in text:
            continue
        rel = os.path.relpath(cha, root)
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
        child = _child_from_path(corpus, rel)
        if corpus.startswith("PERCEPT"):
            child_id, group = f"PERCEPT_{child}", pgroups.get(child, "unknown")
        else:
            child_id = f"{corpus}_{child}"
            group = FIXED_GROUP.get(corpus) or TE_GROUP.get(idm.group(5) if idm else "", "unknown")
        task = _task(corpus, rel)
        for block in re.split(r"\n(?=\*)", text):
            if not block.startswith("*CHI:"):
                continue
            pm = re.search(r"^%pho:\t?(.*(?:\n\t.*)*)", block, re.M)
            if not pm:
                continue
            ipa = " ".join(pm.group(1).split())
            main = block.split("\n%")[0]
            ortho = " ".join(BULLET.sub("", main[len("*CHI:"):]).split())
            t = BULLET.search(main)
            start = end = dur = None
            if t:
                start, end = int(t.group(1)) / 1000, int(t.group(2)) / 1000
                dur = round(end - start, 3)
            lab, flags = label_fields(ipa, dur)
            if t is None:
                flags.add("no_time")
            if audio is None:
                flags.add("no_audio")
            if age is not None and age >= 18 * 12:
                flags.add("adult")
            yield {
                "clip_id": clip_id(corpus, rel, start, end),
                "corpus": corpus, "child_id": child_id, "age_months": age, "sex": sex,
                "group": group, "task": task,
                "source_path": audio, "start": start, "end": end, "duration": dur,
                "orthographic": ortho, "ipa_original": ipa,
                **lab, "flags": sorted(flags), "sip_sealed": False,
            }


def fix_child_metadata(rows):
    """Headers sometimes disagree across a child's files (e.g. sex in 3 ACAD children).
    Use the most common value per child, and flag the rows that were changed."""
    import collections
    for field in ("sex", "group"):
        votes = collections.defaultdict(collections.Counter)
        for r in rows:
            if r[field] not in (None, "unknown"):
                votes[r["child_id"]][r[field]] += 1
        for r in rows:
            v = votes.get(r["child_id"])
            if v and len(v) > 1:
                best = v.most_common(1)[0][0]
                if r[field] != best:
                    r[field] = best
                    r["flags"] = sorted(set(r["flags"]) | {f"{field}_fixed"})


CHAT_CORPORA = ["ACAD", "Cummings", "PERCEPT-R", "PERCEPT-GFTA", "Penney", "Providence",
                "Preston", "TorringtonEaton", "McAllister"]
READERS = {"SPROUT": read_sprout, **{c: (lambda c=c: read_chat(c)) for c in CHAT_CORPORA}}


def main():
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(READERS)
    os.makedirs(os.path.join(OUT, "phonetic"), exist_ok=True)
    print(f"{'corpus':16}{'clips':>8}{'children':>9}{'hours':>8}{'usable h':>9}  top flags")
    for name in names:
        path = os.path.join(OUT, "phonetic", f"{name}.jsonl")
        n, hours, usable, children, flagc = 0, 0.0, 0.0, set(), {}
        UNKNOWN.clear()
        all_rows = list(READERS[name]())
        fix_child_metadata(all_rows)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for row in all_rows:
                row["phoneset"] = VERSION
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                children.add(row["child_id"])
                d = row["duration"] or 0
                hours += d
                if not set(row["flags"]) - INFO_FLAGS:
                    usable += d
                for fl in row["flags"]:
                    flagc[fl] = flagc.get(fl, 0) + 1
        if UNKNOWN:
            os.remove(path + ".tmp")
            print(f"{name}: UNKNOWN SYMBOLS, nothing written. Add them to scripts/phoneset.py:")
            for k, (c, ex) in sorted(UNKNOWN.items(), key=lambda x: -x[1][0]):
                print(f"   {c:6}  {ex}")
            continue
        os.replace(path + ".tmp", path)
        top = ", ".join(f"{k} {v}" for k, v in sorted(flagc.items(), key=lambda x: -x[1])[:4])
        print(f"{name:16}{n:>8}{len(children):>9}{hours / 3600:>8.1f}{usable / 3600:>9.1f}  {top}")
    print(f"written to {os.path.join(OUT, 'phonetic')}")


if __name__ == "__main__":
    main()

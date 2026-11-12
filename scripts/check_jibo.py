"""Check the JIBO Kids download: audio files, transcripts, and totals."""
import csv, json, pathlib, collections, sys
import soundfile as sf

R = pathlib.Path("/projects/p32576/simron/data/child_asr/raw")
AUD = R / "jibo_audio/extracted/jibo_release/data"
TR = R / "jibo_transcripts/extracted"

meta = list(csv.DictReader(open(TR / "metadata.csv")))
wavs = {str(p.relative_to(AUD)): p for p in AUD.rglob("*.wav")}
print(f"metadata rows {len(meta)}, wav files {len(wavs)}, transcript files {len(list((TR/'transcripts').glob('*.jsonl')))}")

# audio properties
srs, chans, bad, total_audio_s = collections.Counter(), collections.Counter(), [], 0.0
info = {}
for rel, p in wavs.items():
    try:
        i = sf.info(str(p)); info[rel] = i
        srs[i.samplerate] += 1; chans[i.channels] += 1; total_audio_s += i.duration
    except Exception as e:
        bad.append((rel, str(e)))
print("sample rates", dict(srs), "channels", dict(chans), "unreadable", len(bad), bad[:3])
print(f"total audio {total_audio_s/3600:.1f} h over {len(wavs)} files")

# children
def child(rel): return rel.split("/")[1]          # e.g. T1027235
kids_all = {child(r) for r in wavs}
kid_num = {k[-3:] for k in kids_all}               # ZZZ = unique child
print(f"recording IDs {len(kids_all)}, unique children (by last 3 digits) {len(kid_num)}")
print("tasks (all wavs):", dict(collections.Counter(r.split('/')[0] for r in wavs)))

# untranscribed wavs
transcribed = {m["raw_audio_path"] for m in meta}
untr = sorted(set(wavs) - transcribed)
print(f"wavs without transcript {len(untr)}:", dict(collections.Counter(r.split('/')[0] for r in untr)))

# transcripts vs metadata
utt_total = 0; mism_count = 0; mism_dur = []; out_of_range = 0; ipa_missing = 0; inexact = 0; xxx = 0
speech_s = 0.0; speech_exact_s = 0.0; neg = 0; overlap = 0
kids_tr = set()
for m in meta:
    lines = [json.loads(l) for l in open(TR / m["transcript_path"])]
    utt_total += len(lines)
    if len(lines) != int(m["utterance_count"]): mism_count += 1
    i = info.get(m["raw_audio_path"])
    if i and abs(i.duration/60 - float(m["raw_file_duration_minutes"])) > 0.05:
        mism_dur.append((m["raw_file_name"], round(i.duration/60, 2), m["raw_file_duration_minutes"]))
    kids_tr.add(m["raw_audio_path"].split("/")[1][-3:])
    prev_end = -1
    for u in lines:
        s, e = u["start_seconds"], u["end_seconds"]
        if e <= s: neg += 1
        if i and e > i.duration + 0.05: out_of_range += 1
        if s < prev_end - 0.01: overlap += 1
        prev_end = max(prev_end, e)
        d = max(e - s, 0); speech_s += d
        if not (u.get("phonetic_text_normalized") or "").strip(): ipa_missing += 1
        if u.get("inexact_transcription"): inexact += 1
        else: speech_exact_s += d
        if "xxx" in (u.get("orthographic_text_normalized") or ""): xxx += 1
print(f"utterances {utt_total}; files where count != metadata {mism_count}; audio length != metadata {len(mism_dur)} {mism_dur[:3]}")
print(f"utterances ending past audio end {out_of_range}; end<=start {neg}; overlapping previous {overlap}")
print(f"missing IPA {ipa_missing}; inexact {inexact}; contain xxx {xxx}")
print(f"speech in utterances {speech_s/3600:.2f} h (exact only {speech_exact_s/3600:.2f} h); children with transcripts {len(kids_tr)}")

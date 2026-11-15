"""Audio prep (DATA.md §4): denoise, cut, normalize, 16 kHz mono. Two copies per clip.

    $PY -I scripts/prep_audio.py --shard 0 --nshards 64        # one Slurm array task
    $PY -I scripts/prep_audio.py --limit 5 --corpus Cummings   # quick local test

For every usable clip in the manifests (no blocking flags; Providence age 3+):
  1. load the whole source file at its own rate, mono
     (session corpora: the whole session; SPROUT: the clip itself)
  2. denoised copy: noisereduce spectral gating on the whole file (PedzSTAR QC
     setting: nr.reduce_noise defaults), then cut
     plain copy:    cut only
  3. each clip, both copies: loudness to -16 LUFS (pyloudnorm). Clips under the
     0.4 s BS.1770 block: peak-normalize to PEAK_SHORT instead (DATA.md §4)
  4. resample to 16 kHz with soxr (band-limited, high quality)
Writes processed/<corpus>/{denoised,plain}/<clip_id>.wav (32-bit float, so -16 LUFS never clips) and
processed/index/shard-XXXX.jsonl (clip_id, both paths, duration, how normalized).
Same steps for train and test audio. Re-running skips clips already written.

--words: word-only clips (manifests/orthographic, for the word head). Same steps, but saved as
16-bit FLAC at the level before loudness normalization, with the gain in the index (about 7x
smaller). Training rescales each clip to unit variance anyway, so the gain changes nothing there.
Index: processed/index/words-shard-XXXX.jsonl.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import warnings
import zlib

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import INFO_FLAGS, OUT  # noqa: E402
from build_word_manifest import usable as word_usable  # noqa: E402

warnings.filterwarnings("ignore", message="Possible clipped samples")  # float output: nothing clips
TARGET_SR = 16000
TARGET_LUFS = -16.0
PEAK_SHORT = 0.5      # linear peak for clips under 0.4 s (about what -16 LUFS speech peaks at)
PROCESSED = os.path.join(os.path.dirname(OUT.rstrip("/")), "processed")
WORD_CORPORA = ["Cameron", "EllisWeismer", "OCSC", "MyST"]
CORPORA = ["SPROUT", "ACAD", "Cummings", "PERCEPT-R", "PERCEPT-GFTA", "Penney", "Providence",
           "Preston", "TorringtonEaton", "McAllister"]


def usable(r):
    if set(r["flags"]) - INFO_FLAGS or not r["source_path"]:
        return False
    return not (r["corpus"] == "Providence" and (r["age_months"] or 0) < 36)


def load_mono(path):
    """Float32 mono at the file's own rate. m4a/mp3 go through ffmpeg."""
    if path.endswith((".m4a", ".mp3")):
        sr = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                 "stream=sample_rate", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout.strip())
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy(), sr
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def normalize(y, sr, meter_cache={}):
    import pyloudnorm as pyln
    meter = meter_cache.setdefault(sr, pyln.Meter(sr))
    if len(y) >= int(meter.block_size * sr):
        lufs = meter.integrated_loudness(y)
        if np.isfinite(lufs):
            out = pyln.normalize.loudness(y, lufs, TARGET_LUFS)
            how = "lufs"
        else:  # digital silence
            out, how = y, "silent"
    else:
        peak = float(np.max(np.abs(y))) or 1.0
        out, how = y * (PEAK_SHORT / peak), "peak"
    return out.astype(np.float32), how  # saved as float: peaks over 1.0 are kept, never clipped


def to16k(y, sr):
    return y if sr == TARGET_SR else soxr.resample(y, sr, TARGET_SR, quality="HQ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--corpus", action="append", help="only these corpora")
    ap.add_argument("--limit", type=int, default=0, help="only this many source files (testing)")
    ap.add_argument("--words", action="store_true", help="word-only manifests, FLAC output")
    args = ap.parse_args()
    import noisereduce as nr

    by_source = collections.defaultdict(list)
    kind_dir, ok = ("orthographic", word_usable) if args.words else ("phonetic", usable)
    for c in args.corpus or (WORD_CORPORA if args.words else CORPORA):
        for line in open(os.path.join(OUT, kind_dir, f"{c}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if ok(r) and zlib.crc32(r["source_path"].encode()) % args.nshards == args.shard:
                by_source[r["source_path"]].append(r)
    sources = sorted(by_source)
    if args.limit:
        sources = sources[: args.limit]
    os.makedirs(os.path.join(PROCESSED, "index"), exist_ok=True)
    index_path = os.path.join(PROCESSED, "index", f"{'words-' if args.words else ''}shard-{args.shard:04d}.jsonl")
    done = set()
    if os.path.exists(index_path):
        done = {json.loads(l)["clip_id"] for l in open(index_path)}
    n_new = n_fail = 0
    with open(index_path, "a") as idx:
        for i, src in enumerate(sources):
            rows = [r for r in by_source[src] if r["clip_id"] not in done]
            if not rows:
                continue
            try:
                y, sr = load_mono(src)
                y_dn = nr.reduce_noise(y=y, sr=sr).astype(np.float32)
            except Exception as e:  # report and go on; the index shows what is missing
                print(f"FAIL {src}: {type(e).__name__}: {e}", flush=True)
                n_fail += len(rows)
                continue
            for r in rows:
                a = 0 if r["start"] is None else int(round(r["start"] * sr))
                b = len(y) if r["end"] is None else min(len(y), int(round(r["end"] * sr)))
                if b - a < int(0.05 * sr):
                    print(f"SKIP {r['clip_id']}: clip outside audio", flush=True)
                    n_fail += 1
                    continue
                out = {"clip_id": r["clip_id"], "corpus": r["corpus"], "source_sr": sr}
                for kind, sig in (("denoised", y_dn), ("plain", y)):
                    raw_clip = sig[a:b]
                    clip, how = normalize(raw_clip, sr)
                    if args.words:  # save the un-normalized level as 16-bit FLAC, keep the gain
                        peak = float(np.max(np.abs(raw_clip))) or 1.0
                        gain = float(np.max(np.abs(clip))) / peak if how != "silent" else 1.0
                        clip = np.clip(to16k(raw_clip, sr), -1.0, 1.0)
                        p = os.path.join(PROCESSED, r["corpus"], kind, f"{r['clip_id']}.flac")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        sf.write(p, clip, TARGET_SR, subtype="PCM_16")
                        out[f"{kind}_gain"] = round(gain, 6)
                    else:
                        clip = to16k(clip, sr)
                        p = os.path.join(PROCESSED, r["corpus"], kind, f"{r['clip_id']}.wav")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        sf.write(p, clip, TARGET_SR, subtype="FLOAT")
                    out[f"{kind}_path"], out[f"{kind}_norm"] = p, how
                out["duration"] = round(len(clip) / TARGET_SR, 3)
                idx.write(json.dumps(out) + "\n")
                n_new += 1
            idx.flush()
            if i % 50 == 0:
                print(f"[{i + 1}/{len(sources)}] files, {n_new} clips written", flush=True)
    print(f"shard {args.shard}: {len(sources)} source files, {n_new} new clips, {n_fail} failed/skipped")


if __name__ == "__main__":
    main()

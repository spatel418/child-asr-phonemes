"""Audio-judge step 2: can TTS + an encoder tell the real label from a wrong one?

For each train clip (from audio_judge_pick.py):
  - make 3 corruptions of the clinician label at one random position:
    near swap (1 phonetic feature apart), far swap (all features apart), deletion
  - synthesize all 4 strings with StyleTTS 2, voice copied from the real clip,
    no random style draw, durations scaled to the real clip (so only the phones differ)
  - embed real and synthetic audio with each encoder, align with DTW,
    distance = mean cosine distance along the path
  - "win" = label is closer to the real audio than the corruption

Pre-set rules are in the research log (2026-11-17, audio-judge entry).
Per-clip results go to a private file (mode 600); the summary has numbers only.
Runs in the tts-judge env:  python -I scripts/audio_judge_check.py --clips ... --out-dir ...
"""
import argparse
import json
import os
import random
import sys

import numpy as np

# ---------------- phonetic features (our 44-phone set, phoneset v1) ----------------
CONS = {  # phone: (voiced, place, manner)
    "p": (0, "bilabial", "stop"), "b": (1, "bilabial", "stop"),
    "t": (0, "alveolar", "stop"), "d": (1, "alveolar", "stop"),
    "k": (0, "velar", "stop"), "ɡ": (1, "velar", "stop"), "ʔ": (0, "glottal", "stop"),
    "m": (1, "bilabial", "nasal"), "n": (1, "alveolar", "nasal"), "ŋ": (1, "velar", "nasal"),
    "f": (0, "labiodental", "fricative"), "v": (1, "labiodental", "fricative"),
    "θ": (0, "dental", "fricative"), "ð": (1, "dental", "fricative"),
    "s": (0, "alveolar", "fricative"), "z": (1, "alveolar", "fricative"),
    "ʃ": (0, "postalveolar", "fricative"), "ʒ": (1, "postalveolar", "fricative"),
    "h": (0, "glottal", "fricative"), "ç": (0, "palatal", "fricative"),
    "ɬ": (0, "alveolar", "lateral-fricative"),
    "tʃ": (0, "postalveolar", "affricate"), "dʒ": (1, "postalveolar", "affricate"),
    "l": (1, "alveolar", "lateral"), "ɹ": (1, "alveolar", "approximant"),
    "w": (1, "labial-velar", "approximant"), "j": (1, "palatal", "approximant"),
    "ɾ": (1, "alveolar", "tap"),
}
VOW = {  # phone: (height 1=high..5=low, backness 0=front 1=central 2=back, round)
    "i": (1, 0, 0), "ɪ": (2, 0, 0), "e": (3, 0, 0), "ɛ": (4, 0, 0), "æ": (5, 0, 0),
    "u": (1, 2, 1), "ʊ": (2, 2, 1), "o": (3, 2, 1), "ɔ": (4, 2, 1), "ɑ": (5, 2, 0),
    "ʌ": (4, 1, 0), "ə": (3, 1, 0),
}
# not used as swap targets or sources: ɚ (rhotic), diphthongs aɪ aʊ ɔɪ
OTHER = {"ɚ", "aɪ", "aʊ", "ɔɪ"}
ALL = set(CONS) | set(VOW) | OTHER


def near_far(ph):
    """Candidate near and far replacements for one phone (empty lists if none)."""
    if ph in CONS:
        v, pl, mn = CONS[ph]
        near, far = [], []
        for q, (v2, pl2, mn2) in CONS.items():
            if q == ph:
                continue
            diff = (v != v2) + (pl != pl2) + (mn != mn2)
            if diff == 1:
                near.append(q)
            elif diff == 3:
                far.append(q)
        return near, far
    if ph in VOW:
        h, b, r = VOW[ph]
        near, far = [], []
        for q, (h2, b2, r2) in VOW.items():
            if q == ph:
                continue
            d = abs(h - h2) + abs(b - b2)
            if d == 1:
                near.append(q)
            elif d >= 4:
                far.append(q)
        return near, far
    return [], []


def corrupt(words, rng):
    """words: list of phone lists. Returns dict of variant -> words, plus the position info."""
    flat = [(wi, pi, p) for wi, w in enumerate(words) for pi, p in enumerate(w)]
    for _, _, p in flat:
        if p not in ALL:
            raise ValueError(f"unknown phone {p!r}: stop")
    ok = [(wi, pi, p) for wi, pi, p in flat if all(near_far(p))]
    if not ok:
        return None
    wi, pi, p = rng.choice(ok)
    near, far = near_far(p)
    out = {"label": words}
    for name, rep in (("near", rng.choice(near)), ("far", rng.choice(far)), ("del", None)):
        w2 = [list(w) for w in words]
        if rep is None:
            del w2[wi][pi]
        else:
            w2[wi][pi] = rep
        w2 = [w for w in w2 if w]
        out[name] = w2
    info = {"src": p, "near": out["near"], "far": out["far"],
            "kind": "consonant" if p in CONS else "vowel"}
    return out, info


# ---------------- TTS (StyleTTS 2, MIT pip package, LibriTTS checkpoint) ----------------
TTS_MAP = {"e": "eɪ", "o": "oʊ"}  # our merged diphthongs -> how the TTS was trained to see them


def to_tts_string(words):
    return " ".join("".join(TTS_MAP.get(p, p) for p in w) for w in words) + "."


class TTS:
    def __init__(self, device):
        import torch
        from styletts2 import tts as st
        from styletts2.text_utils import dicts
        self.torch, self.dicts = torch, dicts
        self.m = st.StyleTTS2()
        self.m.device = device
        self.length_to_mask = st.length_to_mask

    def tokens(self, s):
        ids = [0]
        for ch in s:
            if ch not in self.dicts:
                raise ValueError(f"TTS has no symbol {ch!r} (U+{ord(ch):04X}): stop")
            ids.append(self.dicts[ch])
        return ids

    def style(self, path):
        """Voice/style vector from the real clip. Very short clips are tiled to >= 1 s
        (the style encoder's conv needs a minimum length)."""
        import librosa
        torch, m = self.torch, self.m
        from styletts2.tts import preprocess
        wave, _ = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if len(audio) < 24000:
            audio = np.tile(audio, int(np.ceil(24000 / max(len(audio), 1))))
        mel = preprocess(audio).to(m.device)
        with torch.no_grad():
            ref_s = m.model.style_encoder(mel.unsqueeze(1))
            ref_p = m.model.predictor_encoder(mel.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def synth(self, s, ref_s, target_sec=None):
        """Like styletts2 inference(), but:
        - tokens come straight from our IPA (no phonemizer)
        - no diffusion style sample (alpha = beta = 0): style is the child's clip, so output is deterministic
          (random style draws made the same word last 0.6 s or 5 s; smoke test 2026-11-17)
        - phone durations rescaled so the total matches the real clip (target_sec)"""
        torch, m = self.torch, self.m
        tokens = torch.LongTensor(self.tokens(s)).to(m.device).unsqueeze(0)
        with torch.no_grad():
            L = torch.LongTensor([tokens.shape[-1]]).to(m.device)
            mask = self.length_to_mask(L).to(m.device)
            t_en = m.model.text_encoder(tokens, L, mask)
            bert_dur = m.model.bert(tokens, attention_mask=(~mask).int())
            d_en = m.model.bert_encoder(bert_dur).transpose(-1, -2)
            ref, sty = ref_s[:, :128], ref_s[:, 128:]
            d = m.model.predictor.text_encoder(d_en, sty, L, mask)
            x, _ = m.model.predictor.lstm(d)
            dur = torch.sigmoid(m.model.predictor.duration_proj(x)).sum(axis=-1).squeeze(0)
            if target_sec:
                dur = dur * (target_sec * 40.0 / float(dur.sum()))   # 40 alignment frames/s (decoder makes 600 samples at 24 kHz per frame; measured)
            pred_dur = torch.round(dur).clamp(min=1)
            aln = torch.zeros(int(L), int(pred_dur.sum()))
            c = 0
            for i in range(aln.size(0)):
                aln[i, c:c + int(pred_dur[i])] = 1
                c += int(pred_dur[i])
            aln = aln.unsqueeze(0).to(m.device)
            en = d.transpose(-1, -2) @ aln
            asr = t_en @ aln
            if m.model_params.decoder.type == "hifigan":
                en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)
                asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)
            F0, N = m.model.predictor.F0Ntrain(en, sty)
            out = m.model.decoder(asr, F0, N, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50]  # 24 kHz


# ---------------- encoders ----------------
class MFCC:
    name = "mfcc"

    def __call__(self, wav16):
        import librosa
        m = librosa.feature.mfcc(y=wav16, sr=16000, n_mfcc=13, n_fft=400, hop_length=320)
        d = librosa.feature.delta(m)
        f = np.concatenate([m, d]).T
        return (f - f.mean(0)) / (f.std(0) + 1e-5)  # per-utterance normalization


class HF:
    def __init__(self, repo, layer, device, name):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self.torch, self.device, self.layer, self.name = torch, device, layer, name
        self.fe = AutoFeatureExtractor.from_pretrained(repo)
        m, info = AutoModel.from_pretrained(repo, output_loading_info=True)
        missing = [k for k in info["missing_keys"]]
        if missing:
            # old checkpoints store the weight-normed position conv as weight_g / weight_v;
            # this transformers version expects parametrizations.original0/1. Load them by hand.
            from huggingface_hub import hf_hub_download
            try:
                from safetensors.torch import load_file
                sd = load_file(hf_hub_download(repo, "model.safetensors"))
            except Exception:
                sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu")
            conv = m.encoder.pos_conv_embed.conv
            g = next(v for k, v in sd.items() if k.endswith("pos_conv_embed.conv.weight_g"))
            v = next(v for k, v in sd.items() if k.endswith("pos_conv_embed.conv.weight_v"))
            with torch.no_grad():
                conv.parametrizations.weight.original0.copy_(g)
                conv.parametrizations.weight.original1.copy_(v)
            left = [k for k in missing if "pos_conv_embed.conv.parametrizations" not in k]
            if left:
                raise RuntimeError(f"{repo}: weights still missing: {left}")
            print(f"{repo}: loaded pos_conv weight_g/weight_v by hand")
        self.m = m.to(device).eval()

    def __call__(self, wav16):
        x = self.fe(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with self.torch.no_grad():
            h = self.m(x, output_hidden_states=True).hidden_states[self.layer][0]
        f = h.float().cpu().numpy()
        return f - f.mean(0)  # remove per-utterance mean (speaker/channel offset)


def dtw_dist(a, b):
    import librosa
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    C = 1.0 - a @ b.T
    D, wp = librosa.sequence.dtw(C=C)
    return float(D[-1, -1] / len(wp))


def boot_ci(rows, key, n=2000, seed=0):
    """95% CI of the win rate, resampling children."""
    by = {}
    for r in rows:
        by.setdefault(r["child"], []).append(r[key])
    kids = list(by)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.choice(len(kids), len(kids))
        v = [x for i in pick for x in by[kids[i]]]
        stats.append(np.mean(v))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoders", default="mfcc,hubert")
    ap.add_argument("--hubert-layer", type=int, default=18)   # fixed before results
    ap.add_argument("--babyhubert-repo", default="labicquette/BabyHuBERT")
    ap.add_argument("--babyhubert-layer", type=int, default=9)  # middle of a base model, fixed before results
    ap.add_argument("--save-wavs", type=int, default=5, help="save synthetic audio for every Nth clip (private); 30 = one per cell")
    a = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    clips = [json.loads(l) for l in open(a.clips, encoding="utf-8")]
    if a.limit:
        clips = clips[:a.limit]

    tts = TTS(device)
    encs = []
    for e in a.encoders.split(","):
        if e == "mfcc":
            encs.append(MFCC())
        elif e == "hubert":
            encs.append(HF("facebook/hubert-large-ll60k", a.hubert_layer, device, "hubert"))
        elif e == "babyhubert":
            encs.append(HF(a.babyhubert_repo, a.babyhubert_layer, device, "babyhubert"))
        else:
            raise ValueError(e)

    os.makedirs(a.out_dir, mode=0o700, exist_ok=True)
    wav_dir = os.path.join(a.out_dir, "wavs")
    rows, skipped = [], 0
    for ci, c in enumerate(clips):
        words = [w.split() for w in c["phones"].split("|")]
        words = [w for w in words if w]
        got = corrupt(words, rng)
        if got is None:
            skipped += 1
            continue
        variants, info = got
        real16, _ = librosa.load(c["path"], sr=16000)
        real16, _ = librosa.effects.trim(real16, top_db=30)
        ref_s = tts.style(c["path"])
        syn16 = {}
        for name, w in variants.items():
            y = tts.synth(to_tts_string(w), ref_s, target_sec=len(real16) / 16000)
            y = librosa.resample(y, orig_sr=24000, target_sr=16000)
            y, _ = librosa.effects.trim(y, top_db=30)
            syn16[name] = y
            if a.save_wavs and ci % a.save_wavs == 0:
                os.makedirs(wav_dir, mode=0o700, exist_ok=True)
                sf.write(os.path.join(wav_dir, f"{ci:03d}_{name}.wav"), y, 16000)
                os.chmod(os.path.join(wav_dir, f"{ci:03d}_{name}.wav"), 0o600)
        row = {"i": ci, "child": c["child"], "corpus": c["corpus"], "age_months": c["age_months"],
               "group": c["group"], "kind": info["kind"]}
        for enc in encs:
            fr = enc(real16)
            d = {n: dtw_dist(fr, enc(y)) for n, y in syn16.items()}
            for n in ("near", "far", "del"):
                row[f"{enc.name}_win_{n}"] = float(d["label"] < d[n])
            row[f"{enc.name}_d"] = d
        rows.append(row)
        if ci % 25 == 0:
            print(f"{ci}/{len(clips)}", {k: v for k, v in row.items() if "_win_" in k}, flush=True)

    fd = os.open(os.path.join(a.out_dir, "per_clip.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"n_clips": len(rows), "skipped_no_swap_position": skipped, "tts": "styletts2-libritts",
               "hubert_layer": a.hubert_layer, "encoders": {}}
    for enc in encs:
        s = {}
        allw = [r[f"{enc.name}_win_{n}"] for r in rows for n in ("near", "far", "del")]
        s["win_all"] = float(np.mean(allw))
        for n in ("near", "far", "del"):
            s[f"win_{n}"] = float(np.mean([r[f"{enc.name}_win_{n}"] for r in rows]))
            s[f"win_{n}_ci"] = boot_ci(rows, f"{enc.name}_win_{n}")
        s["pass"] = bool(s["win_all"] >= 0.75 and s["win_near_ci"][0] > 0.5)
        s["by_corpus"] = {}
        for corp in sorted({r["corpus"] for r in rows}):
            rr = [r for r in rows if r["corpus"] == corp]
            s["by_corpus"][corp] = {"n": len(rr), **{n: round(float(np.mean([r[f"{enc.name}_win_{n}"] for r in rr])), 3)
                                                      for n in ("near", "far", "del")}}
        s["by_kind"] = {k: {n: round(float(np.mean([r[f"{enc.name}_win_{n}"] for r in rows if r["kind"] == k])), 3)
                            for n in ("near", "far", "del")} for k in ("consonant", "vowel")}
        summary["encoders"][enc.name] = s
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

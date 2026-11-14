# Data: sources, splits and preprocessing

Last updated 2026-11-13. Status: **downloads done; splits and preprocessing proposed, not built.**

This file says what data we use, where it comes from, and how we split and process it. Background on every corpus we looked at is in `notes/corpora_inventory.md`. Decisions are logged in the (private) research log.

## Goal and scope

- First model: **phoneme ASR**. Audio in, IPA out. A CTC model (e.g. WavLM-Large) is fine-tuned on audio→IPA pairs, then trained further with RL.
- End use: help flag children with speech sound disorders (SSD). Turn the target word into expected phonemes, compare them with the phonemes the model hears, and look at the differences.
- So every training clip needs a person's IPA transcription of what the child **actually said**, not the expected pronunciation.
- pasketti-phonetic (DrivenData's released model) is only a **benchmark**. We do not train from it, and we do not use the DrivenData competition data.

## Where the data lives

All downloads go to the folder set by `CHILD_ASR_RAW` in `.env` (see `.env.example`). Layout of the data root:

```
child_asr/
  raw/<corpus>/          as downloaded, read-only
  processed/<corpus>/    16 kHz mono (denoised, plus a plain copy)
  manifests/phonetic/    <corpus>.jsonl: clips with IPA of what was said
  manifests/orthographic/
  splits/phonetic/       frozen train / val / test lists
  splits/orthographic/
adult_asr/               same layout (empty until adult data arrives)
```

TalkBank and PhonBank corpora are downloaded with `scripts/tb_download.py` (needs a free TalkBank account).

## 1. Sources

"IPA h" = hours of child speech that has an IPA transcription of what was said. For TalkBank corpora it is counted from the CHAT files with `scripts/chat_label_hours.py` (child lines with a `%pho` tier).

### 1a. Training pool

| Corpus | Who | IPA h | Notes |
|---|---|---|---|
| SPROUT (v2, our lab) | 216 children, ~4 y; 63 SSD, 20 language-only, 108 typical, 25 unknown | 29.7 | Already cut into child-only clips (all six tasks). Some children go to val and Test C (§3) |
| Providence | 6 children, recorded over time | 22.6 (ages 3–4) | Free play at home. **Ages 0–2 (126 h) held back for now** |
| ACAD | Ages 4–7, typical | 15.1 | One short file per transcript, child only. `%pho` but no target tier |
| Cummings | Ages 3–7, SSD | 13.2 | Long sessions; `%pho` + `%mod` (target). Many segments are exactly 3.000 s: alignment to be checked |
| PERCEPT-R | Ages 6–24, mostly residual /ɹ/ errors + peers | 11.9 | Only 40,127 of 105,280 child lines have `%pho` on TalkBank. Drop the adults |
| PERCEPT-GFTA | Ages 7+, SSD + peers | 4.3 | Many children also in PERCEPT-R: one ID per child |
| Penney | Ages 5–6 | 1.1 | |

About **98 h** in all (75 h without Providence). For comparison, pasketti's phonetic model used about 160 h. The gap is almost entirely data we don't have (§1d).

### 1b. Outside sets (never trained on)

| Corpus | Who | IPA h | Use |
|---|---|---|---|
| Preston | 44 children, ages 4–6, all SSD | 2.0 | **Proposed: part of the val set** (§3) |
| TorringtonEaton | 51 SSD children, ages 4–6, 3 tasks | 2.6 | Test B |
| McAllister | Age 4 | 2.1 | Test B |
| JIBO Kids | Pre-K–1st grade, talking with a robot | not counted yet | Test B (pasketti never trained on it) |
| SEED (our lab) | Adults and children, typical and SSD | none | **No human IPA**, only target lists. Used for target-vs-heard checks, not PER. Its `pipeline_processing/` folder duplicates adult files and is excluded |
| PaidoEnglish | Ages 2–6 | none | SLP scored each target consonant right or wrong. Used to test error detection |
| Chiat | Ages 5–6 | partial | Only some words transcribed |

Never train on TinyVox. It repackages these PhonBank sets.

### 1c. Audio without IPA

Cameron, Ellis Weismer and OCSC have word transcripts only. So do these, which are for a future word model, not the phoneme model:

| Corpus | Who | Hours | Labels |
|---|---|---|---|
| MyST (LDC2021S05) | 1,371 children, grades 3–5, talking to a science tutor | 473 h audio; 102,433 of 227,567 utterances transcribed | Words. Comes split into train / development / test by student; keep that split |
| CSLU Kids (LDC2007S18) | ~1,100 children, K–grade 10 | 100 h audio | Spontaneous: word transcripts (30.6 h, one long file per child). Scripted: single prompted words, each marked only for whether the target word was said, not how. **No phone labels** |
| CHILDES-Aligned | 49 corpora, ages 0;6–14;3 | 413 h | Words (raw CHAT and normalized) |

TIMIT (LDC93S1) is adult read speech with hand-labeled phones: 3.94 h train, 1.44 h test, 630 speakers. Used for the adult forgetting test and as a small adult replay share in SFT. Its 61 phone symbols get mapped to our IPA set.

Providence ages 0–2 has IPA but is held back. The word-only audio above could feed a self-supervised "mid-training" stage on unlabeled child audio (postponed until after the SFT baseline).

### 1d. Not available now

We plan as if these never arrive, and add them as they are if they do.

| What | Hours | Why missing |
|---|---|---|
| DrivenData's new IPA labels for Ellis Weismer, Cameron, ENNI and the rest of PERCEPT-R | ~70 | Announced as "forthcoming" on TalkBank, not posted. Audio for all but ENNI is already downloaded |
| ReadNet | ~25 | By request to the PI; waiting |
| CMU Kids (LDC97S63) | ~5k read sentences | Not covered by our LDC access; to request |

Looked at and not used: UltraSuite (phone labels follow the expected pronunciation, not what SSD children said), Korean SSD sets (wrong language for this model). speechocean762 is a maybe (L2 adult and child phone errors, not SSD).

## 2. What we record for every clip

One manifest row per clip:

`path, duration, corpus, child_id, age, sex, task, group, target_word, ipa_original, ipa_normalized, flags, split_tag`

- **group** (per child): `typical`, `SSD`, `language-only`, `unknown`. For SPROUT: at risk for speech, or speech + language → `SSD`; language only → `language-only`; not at risk → `typical`.
- **target_word**: known for picture-naming tasks (SPROUT GFTA and PCT, PERCEPT-GFTA, Preston, TorringtonEaton). Needed for target vs actual.
- **ipa_normalized**: one shared IPA symbol set for all corpora. Labs write IPA differently, so every corpus is mapped. Unknown symbols stop the script instead of being dropped.
- **flags**: low SNR, too short, empty or too-long label, unintelligible, and so on. Clips are flagged, not deleted.

## 3. Splits (proposed)

All splits are **by child**, never by clip. A child in two corpora gets one ID and lands in one split.

| Split | Who | Used for |
|---|---|---|
| **Train** | The training pool minus the children below | SFT, then RL |
| **Val** (one set) | Held-out children from the training corpora + Preston + ~15 SPROUT children | Choosing checkpoints, early stopping, tracking progress |
| **Test A** | Held-out children from the training corpora | Same kind of data as training |
| **Test B** | TorringtonEaton, McAllister, JIBO | New labs and labelers |
| **Test C** | 40 SPROUT children sealed for another study + ~15–20 more, balanced SSD / typical | Main clinical test |
| **Adult test** | TIMIT test set (core test as the headline) | Forgetting |

**Val must be a realistic mirror of the tests, not an easy win.** So it mixes three kinds of children: same corpora as training, a new lab (Preston), and SPROUT clinical children. Each clip carries a tag for its kind and for typical / SSD.
- The checkpoint score averages PER **within each tag group first**, then across groups. Pooling all clips would let the biggest, easiest group dominate.
- Sub-scores are logged too. If in-domain PER falls while Preston PER rises, the model is fitting our corpora's quirks.

**SPROUT notes:**
- The 40 sealed children are never trained on. Using them in Test C costs no training data.
- pasketti trained on 50 SPROUT children. That matters only for the benchmark comparison.

**SSD balance:** SSD children make up about 30–40% of training examples (sampled by child, with a cap on clips per child).

**Rules:** freeze the split files once made and log their checksums. Look at test results only at the end.

## 4. Preprocessing (proposed)

**Cutting:**
- Corpora that come as clips are used as they are: SPROUT (child-only clips), ACAD, SEED. They are not re-cut from sessions.
- TalkBank session corpora (Cummings, PERCEPT, Preston, TorringtonEaton, Providence, ...) are cut into clips at the CHAT utterance times, child lines only. Cutting removes adult speech, pairs each clip with its own IPA, and keeps inputs short.

**Denoise and normalize** with the PedzSTAR QC pipeline (`noisereduce` spectral gating, then −16 LUFS with `pyloudnorm`; clips under 0.4 s use peak normalization instead):
- Session corpora: denoise the whole session, then cut. The filter estimates noise from the file, and a whole session gives it more to work with than a short clip.
- Clip corpora: denoise each clip.

**Resample to 16 kHz mono:**
- The base models (WavLM, wav2vec2, HuBERT, Whisper) were pretrained on 16 kHz audio, and every corpus must share one rate.
- Use a proper band-limited resampler (low-pass, then resample; e.g. soxr or `scipy.signal.resample_poly`), not sample dropping, to avoid aliasing.
- Cost: nothing above 8 kHz survives, which trims some /s/ detail in child speech.

**Keep a plain 16 kHz copy** (no denoise) and compare PER once. Spectral gating can soften /s/ and /ʃ/, which are common SSD error sounds.

The same steps apply to every corpus and to test audio, so the model cannot learn "this corpus sounds denoised."

# Data: sources, access and splits

Pasketti RLVR project. Last updated 2026-11-09. Status: **plan**. Nothing new has been downloaded yet.

**All data now lives in `/projects/p32576/simron/data/`** (`child_asr/`, `adult_asr/`). The live have/missing list is `/projects/p32576/simron/data/CHECKLIST.md`.

This file is the single place for what data we use, where each piece comes from, and how we split it. Background notes are in `notes/corpora_inventory.md` (all corpora we looked at) and `notes/2026-11-05_competitor_pipelines.md`. Decisions are logged in `RESEARCH_LOG.md`.

## Goal and scope

- First model: **phoneme ASR**. Audio in, IPA out. A CTC model (e.g. WavLM-Large) is fine-tuned (SFT), then trained further with RL (RLVR).
- The end use is to help flag children with speech sound disorders (SSD): turn the target word into expected phonemes, compare with the phonemes the model hears, and look at the differences.
- So every training clip needs a transcript of what the child **actually said**, in IPA, made by a person.
- pasketti-phonetic (DrivenData's released model) is only a **benchmark**. We do not train from it.

## 1. Data sources

### 1a. DrivenData corpora

DrivenData does not own this data. After the challenge, each corpus is shared by its owner: TalkBank, LDC, or the research group. Hours are DrivenData's numbers for the final pasketti dataset (training plus their held-out evaluation).

**With phoneme labels (used to train the phoneme model):**

| Corpus | Children | Phonetic hours | Source | Status |
|---|---|---|---|---|
| SPROUT | 216 children, ~4 y; 63 SSD, 20 language-only, 108 typical, 25 unknown | 29.7 (our copy) | Ours (PedzSTAR lab) | Have. **On hold** (user, 2026-11-11): which copy to use and its condition still to settle |
| ACAD (Arizona) | Ages 2–7, isolated words | 15.7 | TalkBank | Account ready; not downloaded |
| PERCEPT-R | Ages 6–24, SSD and typical | 29.7 | TalkBank | Account ready; **DD phonetic labels "forthcoming"** |
| PERCEPT-GFTA | Ages ~7–17, SSD (some suspected CAS) and typical | 4.2 | TalkBank | Account ready; **DD labels "forthcoming"** (some older labels exist) |
| ENNI | Ages 4–10, ~1 in 5 with language impairment | 27.8 | TalkBank | Account ready; **DD labels "forthcoming"** |
| Ellis Weismer | Ages 2.5–5.5, late talkers and typical | 16.4 | TalkBank | Account ready; **DD labels "forthcoming"** |
| Cameron | Ages 3–5, African American English and General American English | 7.3 | TalkBank | Account ready; **DD labels "forthcoming"** |
| ReadNet | K–3rd grade, literacy tasks | 25.1 | By request to the PI | **Requested 2026-11-09**, waiting |
| JIBO Kids | Pre-K–1st grade, child–robot talk; test only (see §3) | 5.3 | Audio: GitHub balaji1312/Jibo_Kids releases (Zenodo 19604408 mirror). Transcripts: platform.k12-ai-infrastructure.org/datasets/13/jibo (public, CC BY 4.0, no account) | **Downloaded 2026-11-09** to `/projects/p32576/simron/data/child_asr/raw/jibo_*` |

**"Forthcoming" matters.** About 85 of the ~160 phonetic hours depend on DrivenData's new labels being posted on TalkBank. If they are not up yet, we can start training on SPROUT + ACAD + ReadNet (~69 h; JIBO is test only) and add the rest later.

**Words only (not used for the phoneme model; kept for the word model and maybe as unlabeled audio):**

| Corpus | Word hours | Source | Status |
|---|---|---|---|
| MyST | 160.6 | LDC (LDC2021S05) | Waiting on LDC approval |
| Ohio (OCSC) | 86.7 | TalkBank | Account ready |
| CSLU Kids | 62.8 | LDC (LDC2007S18) | Waiting on LDC approval |
| CMU Kids | 3.7 | LDC (LDC97S63) | Waiting on LDC approval |

### 1b. Outside DrivenData: children (test only)

These are never used for training. Neither our model nor pasketti has seen them, so they test how well a model works on new sites and tasks.

| Corpus | Children | Labels | Source | Status |
|---|---|---|---|---|
| TorringtonEaton | 51, ages 4–5 (42 typical, 9 SSD) | SLP IPA, picture naming | PhonBank (same TalkBank account) | Not downloaded |
| Preston | 44, ages 4–5, all SSD | SLP IPA with error types, GFTA-2 items | PhonBank (same TalkBank account) | Not downloaded |
| SEED | Ours, SSD and typical | In-house | Ours | Have. Use only if no child is also in SPROUT |

Never train on TinyVox. It repackages these PhonBank sets.

### 1c. Outside DrivenData: adults (so the model does not forget adult speech)

Only adult data with **hand-made** phone labels. Labels made from a pronouncing dictionary give the textbook pronunciation and would teach the model to write what *should* be said.

| Corpus | What | Source | Status |
|---|---|---|---|
| **TIMIT** | 630 speakers, ~5 h, read speech, hand-labeled phones | LDC (LDC93S1) | Waiting on LDC approval. **First choice** |
| Buckeye | 40 speakers, ~40 h, conversation, hand-corrected phones | buckeyecorpus.osu.edu, free; the **PI signs** the license | Optional |
| L2-ARCTIC | 24 non-native speakers; part hand-labeled, with mispronunciations marked | Texas A&M, free research license | Optional |

pasketti used no adult data. No pasketti team measured forgetting.

**Citations required by licenses:** JIBO = Shankar et al. 2024, JASA Express Letters 4(11):115201, doi 10.1121/10.0034195; balaji1312 (2026) JIBO Kids Corpus v1.0.0, doi 10.5281/zenodo.19604408; DrivenData (2026) JIBO Kids Corpus Transcripts, K-12 AI Infrastructure Program. TalkBank corpora: cite each corpus reference listed on TalkBank.

## 2. What we record for every clip

One manifest row per clip:

`path, duration, corpus, child_id, age, sex, task, group, target_word, ipa_original, ipa_normalized, flags`

- **group** (per child): `typical`, `SSD`, `language-only`, `unknown`.
  - From DrivenData's pathology field where present.
  - SPROUT: at risk for speech, or speech + language → `SSD`; language only → `language-only`; not at risk → `typical`.
  - From the corpus description otherwise (Preston → SSD; ENNI and Ellis Weismer → language-only unless marked otherwise).
- **target_word**: known for picture-naming tasks (SPROUT GFTA and PCT, PERCEPT-GFTA, Preston, TorringtonEaton). Needed for the target-vs-actual diagnostic.
- **ipa_normalized**: one shared IPA symbol set for all corpora (DrivenData's scoring rules). Unknown symbols stop the script instead of being dropped.
- **flags**: low SNR, too short, empty or too-long label, and so on. Clips are flagged, not deleted.

## 3. Splits

All splits are made **by child**, never by clip. A child who appears in two corpora (e.g. PERCEPT-R and PERCEPT-GFTA) gets one ID and lands in one split.

| Split | Who | Used for |
|---|---|---|
| **Train** | Most children in the corpora with phoneme labels, plus adult data (20–30% of examples) | SFT, then RL |
| **Validation** | ~5–8% of children per corpus, balanced by group and age, with SSD children included | Choosing checkpoints and settings |
| **Test A: in-domain** | ~10–15% of children per DrivenData corpus, balanced by group and age | How well it works on familiar corpora |
| **Test B: outside** | TorringtonEaton, Preston, JIBO Kids (pasketti never trained on JIBO) | New sites and tasks; fair vs pasketti |
| **Test C: clinical** | SPROUT children held out, balanced by at-risk group | Main diagnostic test (target vs actual, intelligibility) |
| **Adult test** | Held-out TIMIT speakers (standard 24-speaker core test) | Forgetting |

**SPROUT notes:**
- pasketti trained on 50 SPROUT children. For a fair pasketti comparison, Test C should use only the other 166. That needs DrivenData's list of the 50. It matters only for the benchmark, not for our own results.
- **40 SPROUT children are sealed for the SIP paper** (`/projects/p32576/simron/sip_v2/protocol/test_ids.txt`). We never train on them. Whether this project tests on them is a separate decision.

**Few SSD children:**
- Tests need enough SSD children to measure error detection: at least ~50 across Tests B and C. Preston (44) and TorringtonEaton (9) give most of these, so few SSD children have to leave training.
- Checkpoints are chosen by the **average of typical PER and SSD PER**. Pooled PER would mostly reflect typical children.
- In training, SSD children make up ~30–40% of examples (sampled by child), with a cap on clips per child.

**Rules:**
- Freeze the split files once made and log their checksums.
- Do not look at test results while developing.

## 4. Preprocessing

Decided: denoise, using our PedzSTAR QC pipeline (`repos/sprout-derivatives/src/pedzstar_qc/extractor.py`, `denoise_audio`).

1. `noisereduce` spectral gating.
2. Loudness normalization to −16 LUFS (`pyloudnorm`). Clips under 0.4 s skip this step in that code; use peak normalization for them.
3. Mono, 16 kHz.
4. Keep the QC pipeline's SNR and duration checks as manifest flags.

- The same steps apply to **every** corpus and to test audio. Otherwise the model could learn "this corpus sounds denoised."
- Keep the raw audio too. Compare raw vs denoised once, because denoising can blur /s/ and /ʃ/.
- Long clips are split once, offline, so audio and label always match (no random cropping).

## 5. Storage

- p32576: ~174 GB free (850 / 1024 GB used, 2026-11-08).
- Estimate: DrivenData raw ~30–60 GB, the processed copy about the same, SPROUT at 16 kHz ~16 GB, TIMIT < 1 GB.
- Checkpoints go to scratch storage during runs.
- Layout:
  - `data/raw/<corpus>/`: as downloaded, read-only
  - `data/processed/<corpus>/`: denoised, 16 kHz
  - `data/manifests/`
  - `data/splits/`

## 6. Getting the data: step by step

Requests that take days to answer go first, so the waiting happens in parallel.

**Simron**

1. ~~Email Yaacov Petscher for ReadNet~~ Sent 2026-11-09. Waiting for a reply.
2. ~~Make a K-12 AI Infrastructure Platform account~~ Not needed: the JIBO transcripts are public. Claude downloaded them.
3. **Download the original corpora from TalkBank/PhonBank** (logged in, in your browser). No DrivenData account needed. For each: the transcript zip from the corpus page, and the audio from its media folder. Move to Quest with Globus into `/projects/p32576/simron/data/child_asr/raw/<corpus>/`.
   - **First, likely with IPA:** ACAD (talkbank.org/phon/access/Eng-NA/ACAD.html), PERCEPT-R and PERCEPT-GFTA (talkbank.org/phon/access/Clinical/PERCEPT-R.html, .../PERCEPT-GFTA.html).
   - **Then, words only for now:** Ellis Weismer (talkbank.org/childes/access/Clinical-Eng/EllisWeismer.html), Cameron (talkbank.org/childes/access/Eng-AAE/Cameron.html), OCSC (talkbank.org/childes/access/Eng-NA/OCSC.html).
   - **ENNI:** TalkBank has no public audio ("Media Type: not available"). Skip unless DrivenData's labels and audio appear later.
   - Not the DrivenData competition bundle: the competition closed in Apr 2026 and its rules limit that data to the challenge period.
4. **Download Preston and TorringtonEaton from PhonBank** (logged in to TalkBank in your browser):
   - Transcripts: talkbank.org/phon/access/Clinical/Preston.html and .../TorringtonEaton.html → the download (zip) link.
   - Audio: media.talkbank.org/phon/Clinical/Preston and .../TorringtonEaton.
   - Move to Quest into `/projects/p32576/simron/data/child_asr/raw/Preston/` and `.../raw/TorringtonEaton/`.
5. **LDC, once approved:** request **TIMIT first** (adults), then CMU Kids, CSLU Kids and MyST. Send Claude the download links.
6. **Later:** watch for DrivenData's "forthcoming" IPA labels (Cameron, ENNI, Ellis Weismer, PERCEPT) on the public TalkBank pages.
7. **Optional, later:** ask DrivenData which 50 SPROUT children pasketti trained on (benchmark only). Get Buckeye (PI signs) or L2-ARCTIC if TIMIT is not enough.

**Claude, in parallel**

- A. ~~Draft the ReadNet email~~ Not needed; sent.
- B. ~~Download JIBO audio and transcripts~~ Done 2026-11-09.
- C. Build the SPROUT manifest at 16 kHz from our copy.
- D. As each corpus lands: audit it, map its IPA to the shared set, add it to the manifest.
- E. When enough data is in: build and freeze the splits.

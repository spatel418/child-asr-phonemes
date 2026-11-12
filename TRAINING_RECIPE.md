# Training recipe: child phoneme ASR, SFT then RLVR

The working plan for the pasketti-rlvr workstream. It gathers what has been decided and what is only proposed so far.
- `RESEARCH_LOG.md` holds the history and the reasons for each choice.
- `notes/corpora_inventory.md` holds the data details.
- Update this file when a decision changes, and log the change.

Labels used below:
- **[decided]**: agreed with the user.
- **[proposed]**: suggested, not agreed yet.
- **[open]**: needs a decision.

Last updated: 2026-11-05.

---

## 1. Goals

- **[decided]** Make a better child ASR model than pasketti, and use the work to learn RL training and general deep-learning practice.
- **[decided]** Phoneme ASR first. Word ASR later.
- **[decided]** SFT first, then RL (RLVR).
- **[decided]** Train on the DrivenData (On Top of Pasketti) data plus other child corpora we can get.
- **[decided]** The model should not forget adult speech.
- **[decided]** Train on children with intelligibility issues and typical children together.

## 2. Data

Full list, labels and access in `notes/corpora_inventory.md`. What to pull, splits, adults and preprocessing: `DATA.md`.

- **Main source:** the DrivenData bundle. It has 14 datasets (13 named), 520 h and 660K+ utterances.
  - Words + phonemes: ACAD, Cameron, ENNI, Ellis Weismer, JIBO Kids, PERCEPT-GFTA, PERCEPT-R, ReadNet, SPROUT.
  - Words only: CMU, CSLU, MyST, Ohio.
  - Phonetically labeled training data is about 116 h. Word-labeled data is about 372 h.
- **Access:**
  - Most of it is on TalkBank, which needs an account tied to a DrivenData email.
  - CMU, CSLU and MyST are LDC. JIBO is on Zenodo plus the K-12 AI Infrastructure Platform.
  - ReadNet means asking the PI (Yaacov Petscher, FSU). SPROUT is our lab's.
- **Watch:** DrivenData's new transcripts for Cameron, ENNI, Ellis Weismer and both PERCEPT sets are "forthcoming on TalkBank". Check what the download actually contains.
- **Extra child data (small, PhonBank, free):**
  - TorringtonEaton: 51 children, 42 typical + 9 SSD, ages 4–5.
  - Preston: 44 SSD, ages 4–5.
  - Penney, Chiat, McAllister, Goad, Davis.
- **Adult data for replay (phonetic):** **[proposed]** TIMIT (LDC, paid) and Buckeye. Check access and license for both.
- **Storage:**
  - p32576 has about 209 GB free. The full set is about 30–60 GB.
  - Put checkpoints in scratch storage during runs. Copy keepers to `/projects`.

## 3. Pipeline

### Stage 0: Download and audit [proposed]

Problems only show up after download. For each corpus, write an audit report before any training:
- Clip count, total hours, and the duration histogram.
- Sample rates and channels.
- Missing audio, empty transcripts, and unreadable files.
- Per-child counts. Find children who appear in more than one corpus:
  - PERCEPT-R and PERCEPT-GFTA
  - JIBO and Ohio
  - MyST and the CU corpora
  - SEED and SPROUT
- Speaker labels. Are clips child-only, or is adult speech mixed in?
- **Phone symbol inventory per corpus:** every symbol used, with counts. Corpora write the same sound differently (SPROUT writes `ai`; pasketti writes `ɑ͡ɪ`).
- Label sanity spot-checks: listen to 20 random clips per corpus, and check them against the transcripts. On CAAP this found swapped target words (lemonade/helicopter).
- If DrivenData's 10% re-transcribed clips are included, save them. They give the human agreement rate, which is the realistic ceiling.

### Stage 1: Preprocessing [proposed]

**Audio:**
- Resample to 16 kHz mono (both pasketti models expect this). Store as FLAC.
- Cut long recordings into utterances using the corpus time stamps (CHAT time marks).
- Keep **child speech only**, using the speaker tiers. Clips with an adult talking over the child need a rule: drop, or keep and mark.
- Drop clips longer than 20 s, as DrivenData did. Decide a minimum length too.
- Loudness: optional peak or RMS normalization. Check that it doesn't clip.
- **Be careful with silence trimming and denoising.** VAD can cut quiet child onsets. Denoising can remove speech cues and often hurts ASR. Treat both as experiments to test on validation, not defaults. An earlier exploratory CAAP rerun on denoised audio exists in pasketti-inference.

**Phoneme labels:**
- Map every corpus to **one phone inventory**. Follow DrivenData's convention: about one character per phone. The 55-symbol pasketti-phonetic alphabet is a ready reference.
- All three phonetic winners trained on the scorer's own normalization:
  - NFC.
  - Strip tie bars, stress and nasal tildes.
  - ɝ → ɚ; tʃ → ʧ; dʒ → ʤ.
  - Space as word boundary; ː as its own token.
  - That gives about 52 symbols. Code: `external/on-top-of-pasketti/phonetic-track/1st Place/src/models/base.py` L32-56, L230-252.
  - **[open]** Train on that normalized set (matches scoring) or keep tie bars and tildes (as pasketti-phonetic does). Either way, train and score with one shared function.
- Write the mapping as a table in code, with a test for each corpus. Unmapped symbols must fail loudly, not pass silently. This is the lesson from the CAAP diphthong bug.
- Rules for transcription marks (unintelligible `xxx`/`yyy`, uncertain, overlap, non-speech): drop the clip, drop the mark, or map to a token. Decide per mark and log it.
- Keep word boundaries (space symbol) as pasketti-phonetic does.

**Word labels** (for the later word model):
- Whisper English normalizer for scoring.
- Keep the raw text too, so normalizer exploits can be spotted.

**Scoring normalization** (match DrivenData so numbers compare):
- PER on IPA after removing spaces, tie bars, nasal tildes, stress marks and ASCII punctuation.
- Also report PER counting word boundaries.

**Metadata per clip:** child ID, corpus, task, age, sex, group (typical / SSD / language disorder / unknown), duration, source file.

**Augmentation** (training only, test as experiments):
- What the competitors found (`notes/2026-11-05_competitor_pipelines.md`):
  - **Phonetic track:** concat-mix (join 2–8 short clips and labels, p 0.5) gave about -0.01. A consistency loss between clean/augmented copies (CR-CTC) gave about -0.015, the largest single gain. SpecAugment, moderate classroom noise, and pitch/tempo were used too.
  - **Word track:** augmentation mostly did not help. Only light SpecAugment (5% time masking) and the competition's own noise at low probability survived.
- Start with SpecAugment + concat-mix (if CTC) + RealClass noise. Add the others as ablations.

**Flag, don't drop:** clips where existing models and the human label disagree strongly go on a review list. Don't remove them automatically. Disordered speech naturally disagrees with model guesses, and dropping those clips would remove exactly the children we care about.

### Stage 2: Splits [proposed]

- **Split by child, never by clip.** A child in two corpora stays in one split.
- **Train:** shared by SFT and RL.
- **Validation:** about 5% of children per corpus. Used for checkpoint choice and tuning.
- **Test A, in-domain:** about 10–15% of children per corpus, balanced by age and group.
- **Test B, held-out task:** corpora never used in training. Candidates: TorringtonEaton and Preston (picture naming, preschoolers, many SSD). The DrivenData analogues are CSLU and CMU (words only).
- **Adult test set:** held-out adult speakers, to measure forgetting.
- **[open]** If DrivenData shares their child-level train/eval split, use their eval children as Test A. That is the only fair way to compare with pasketti models, which may have trained on any child we pick ourselves.
- Never look at test sets while developing. Freeze the split files and record their checksums in the log.
- Lesson from the competitors: random or overlapping validation splits inflated local scores. 1st place's validation only tracked the leaderboard after grouping by child. 2nd place's child-grouped split gave R² ≈ 0.95 against the leaderboard.

### Stage 3: Baselines [proposed]

Score these before training anything, on validation and test, split by group:
- pasketti-phonetic (WavLM CTC). This is the model to beat.
- The chosen base model, untrained (if it can output IPA at all).
- Sanity runs (Karpathy's recipe):
  - **Overfit one batch.** Train on 2–5 clips until near-zero loss / near-perfect PER. If it fails, there is a pipeline bug.
  - **Input-independent baseline.** Feed silence and zeroed audio, and count the output phones. That measures hallucination from the language model alone.

### Stage 4: SFT [proposed]

- **[open] Model choice:**

  | | Qwen3-ASR-1.7B to IPA | CTC model (WavLM, like pasketti-phonetic) |
  |---|---|---|
  | Faithful to what was said | Risky: the LM pulls toward real words | Strong: output comes from audio |
  | RLVR | Fits standard tools (GRPO, TRL / ms-swift / verl) | No standard RL support; custom loop |
  | Learning goal | Same pipeline later serves the word model | Different pipeline from the word model |

  Earlier recommendation: Qwen3-ASR to IPA, with pasketti-phonetic as the baseline to beat. RL with a PER reward is the direct test of whether the model stops "fixing" children's errors.

  **Evidence since (competitor code, 2026-11-05):**
  - Every phonetic winner used CTC encoders, and DrivenData says LLM-based models "offered no clear advantage" for phonemes.
  - 2nd place in the word track already trains Qwen3-ASR to output IPA (swap the target to `phonetic_text`, IPA system prompt). So the Qwen route is cheap to try.
  - A reasonable plan: SFT both (Qwen-IPA and WavLM-CTC), compare on validation, then do RL on Qwen-IPA. Whether RL closes the gap to CTC is itself a result.
- **Reference recipes (published):**
  - pasketti-word (Qwen3-ASR-1.7B, full fine-tune): 2 epochs, global batch 32 (8 x A100), AdamW, peak LR 3e-5, 2% linear warmup, linear decay sized for 5 epochs.
    - Code: `drivendataorg/on-top-of-pasketti`, word-track/3rd Place.
  - pasketti-phonetic (WavLM Large, CTC): 5 epochs, AdamW, LR 3e-5 backbone / 1e-4 head, 10% warmup then cosine to zero, weight decay 0.01 (not on biases/LayerNorm), 1 x H100, batch 4 x 2 grad accumulation.
    - It used a second head predicting words, so the large word-only data also trained the encoder.
    - Code: phonetic-track/1st Place.
- **SFT techniques that worked for the competitors** (details in `notes/2026-11-05_competitor_pipelines.md`):
  - **Qwen3-ASR:**
    - Full fine-tune at LR 1e-5 to 2e-5, linear decay, 1–2% warmup, 2–3 epochs. LoRA was competitive only with weight averaging.
    - Prompt prefix (chat template + audio + `language English<asr_text>`) masked to -100.
    - Average checkpoints and runs ("soup").
    - An optional final pass at LR 1e-6.
    - KL to the frozen base as a regularizer (Mitchell, α 0.5 cosine to 0).
  - **CTC encoders:**
    - Full fine-tune. Head LR 10–100x the backbone.
    - Token-weighted CTC loss.
    - EMA 0.999.
    - Negative blank bias at init.
    - An aux word head (weight 0.3) so word-only data also trains the encoder.
  - **Noisy labels:** score each training clip with a second model (WER, or PER for phonemes), train at a few cut-off levels, and average. Don't drop SSD clips just for disagreeing.
  - **Data:** every top team added DrivenData's TalkBank set (about 255k extra utterances). Drop the known bad clip `U_b8a4e8220e65219b`.
  - **Long clips:** split by forced alignment, or chunk and pseudo-label. Never truncate audio while keeping the full label.
- **Adult replay [proposed]:** mix adult phonetic data into SFT. Start around 20–30% adult. Try 2–3 ratios on validation. No separate continued-pretraining stage unless forgetting shows up. Continued pretraining on unlabeled audio matters more for self-supervised encoders like WavLM than for Qwen3-ASR.
- **Balance:** consider oversampling SSD children if they are a small share of training hours.
- **Framework [open]:**
  - ms-swift (SFT + GRPO, multimodal; torchrun on Slurm, no Ray).
  - Or start from the 3rd-place code for SFT.
  - Check that ms-swift supports Qwen3-ASR before choosing.

### Stage 5: RL (RLVR) [proposed]

- **Algorithm:** GRPO (group of sampled transcripts per clip; advantage = reward minus group mean, scaled by group std; PPO-style clipping; KL penalty to the SFT model).
- **Reward:**
  - Start with `1 - min(PER, 1)` against the human transcription of what the child actually said.
  - Possible added terms:
    - repetition penalty
    - length-ratio penalty
    - empty-output reward on silence/noise clips (needs no transcript)
- **Data:**
  - Same training children as SFT. The most signal comes from clips where samples disagree, often the SSD clips.
  - Log within-group reward std. GRPO learns nothing from groups that all score the same.
- **Reference:**
  - Chat B (unverified) cited a paper that ran GRPO on Qwen3-ASR-1.7B after SFT.
  - Its rewards were WER/CER plus hallucination, repetition and length penalties. Its LR was a constant 1e-6, no warmup.
  - Find and read it before starting.
- **Sanity first:** run GRPO on about 5 clips and confirm the reward climbs to near its maximum.
- **Prior work in the competition:** no team used RL.
  - 1st place (phonetic) tried MCER, sequence-level minimum-CER training, and it did not help. Read how they did it before designing the reward.
  - The gap between each team's chosen output and the best of its candidates (oracle) was 0.03–0.1 CER. Sampled outputs often contain a better answer. That is the room GRPO tries to use.

### Stage 6: Evaluation [proposed]

- PER (DrivenData normalization) on validation, Test A, Test B and the adult test set.
- Always report separately by group (typical vs SSD vs other), by corpus, and by age band. Never only the overall mean.
- Also report substitutions / insertions / deletions.
- Comparisons: pasketti-phonetic, our SFT model, our SFT+RL model.
- Pre-specify the main comparison before running tests. A failed pre-specified test stays a failure. Anything decided afterwards is labeled post-hoc.

## 4. Monitoring

- **Tools:**
  - W&B (reachable from the login node; test from a GPU node).
  - Or TensorBoard on the login node via VS Code port forwarding.
  - Interactive `srun --pty` GPU sessions for debugging.
- **Every run:** loss, LR schedule, grad norm (before clipping), clip fraction, throughput, GPU memory.
- **Layer-level:**
  - Per-layer weight drift from the starting checkpoint, ‖W − W₀‖ / ‖W₀‖, logged separately for the audio encoder, projector and LLM blocks.
  - Per-layer gradient norms.
  - Activation stats (mean / std / max) from forward hooks on a fixed probe batch of about 32 validation clips every N steps.
  - Note: `wandb.watch` logs weights and gradients only, not activations, and is misleading under ZeRO-3.
- **RL-specific:**
  - Reward mean and std, within-group reward std, KL to the SFT model, output length, token entropy.
  - A table of sampled outputs with their rewards every N steps. Read the highest-reward ones.
- **Reward-hacking guards:**
  - Greedy-decoding PER on validation that the reward never touches. If training reward rises while validation PER stalls or worsens, stop and look.
  - Output count on silence/noise clips.
  - Hypothesis/reference length ratio.
  - Deletion and insertion rates.
  - n-gram repetition rate.
  - Normalizer exploits: compare raw vs normalized output.
- **Checkpoints:**
  - Weights-only every N steps (about 3.4 GB each for 1.7B).
  - Full resumable checkpoint (about 24 GB) only for the latest 1–2.
  - Keep the best by validation PER permanently.
  - Quest `gengpu` jobs are capped at 2 days, so resuming must work.

## 5. Compute (Quest)

- Accounts: p32576 (this work). General-access GPUs:
  - `gengpu`: 15 nodes of 4x H100, 11 of 4x A100, 13 of 2x A100. 2-day limit.
  - `gengpu-long`: 2 nodes of 2x A100. 10-day limit.
- A 1.7B full fine-tune fits on one 4-GPU node. Multi-node is not needed at this size.
- Environments: mamba at `/projects/p32576/simron/envs/`. Check `which python` before any pip install.
- pasketti-word needs `qwen_asr==0.0.6`, `vllm==0.14.0`, `transformers==4.57.6`, `torch==2.9.1`. On Quest, vLLM needs the re-tagged wheel; see `pasketti-inference/docs/04_known_issues.md`. Check RL framework compatibility with these pins.

## 6. Planned Claude skills [proposed]

Each skill is written once its step has been done by hand at least once. A skill should encode a procedure that is known to work.

| Skill | What it does |
|---|---|
| `corpus-audit` | Run Stage 0 checks on a downloaded corpus and write its audit report |
| `phone-map` | Build or extend the phone mapping for a corpus, with tests; fail on unmapped symbols |
| `build-splits` | Make child-level splits from metadata, check no child leaks across splits, record checksums |
| `launch-train` | Write and submit a Slurm job for SFT or RL with the standard env, logging and checkpoint settings |
| `run-health-check` | Read a run's logs/W&B and report curves, layer drift and reward-hacking signs |
| `eval-report` | Score a checkpoint on all eval sets, broken down by group/corpus/age, against baselines |
| `log-entry` | Append a correctly formatted entry to `RESEARCH_LOG.md` |

## 7. Open decisions

1. Model for phoneme ASR: Qwen3-ASR to IPA vs. CTC (WavLM).
2. Whether DrivenData shares its child-level split (and the new transcripts).
3. Framework for SFT and RL (ms-swift support for Qwen3-ASR still unchecked).
4. Adult phonetic data source and the replay ratio.
5. Rules for transcription marks and adult-overlap clips.
6. Which held-out-task corpora form Test B.

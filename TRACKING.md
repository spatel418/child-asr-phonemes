# What we track in SFT and RLVR

Status: **proposed** (2026-11-13). Everything goes to Weights & Biases (W&B). W&B only shows what we send it: the framework logs its built-in numbers, and we compute the rest in callbacks or offline jobs.

Built in tiers. Get tier 1 clean on a real run before adding tier 2, then tier 3 (one thing at a time).

| Tier | When | Cost |
|---|---|---|
| **1. Every step** | Inside the training loop | Cheap |
| **2. Every N steps** | At each val pass | Medium |
| **3. Per checkpoint, offline** | Separate job on saved checkpoints (start, middle, end, best) | Expensive (probes, MI, CKA) |

## 0. Rules

**Privacy (decided).** The audio and transcripts are children's data under access agreements, and W&B is a cloud service. So W&B gets **numbers and hashed clip IDs only**. No audio and no transcripts go to W&B. Audio, transcripts and model outputs are viewed in a local viewer on Quest.

**W&B layout.**
- One project. Each training run plus its offline analysis runs share a `group`; analysis runs use `job_type="analysis"` and log against the checkpoint's training step (`wandb.define_metric`), so their curves sit next to the training curves.
- Key prefixes: `train/`, `val/`, `layers/`, `rl/`, `analysis/`, `clinical/`, `sys/`. W&B groups panels by prefix.
- Checkpoints, split files and manifests are W&B Artifacts (metadata and checksums only for data), so every analysis links to the exact checkpoint and split it used.
- Per-layer keys (24 layers × several stats) are logged every N steps, not every step.

**Run process** (from the Hugging Face "contracts"): plan → verify model ID and versions → smoke test (50 steps on a tiny subset; overfit one batch) → log config, git commit, seed, split checksums, Slurm job ID → integrity check (no child in two splits) → select by val → test once at the end → lessons in the research log.

## 1. Every step (tier 1)

**SFT**
- `train/loss` (CTC). Loss at step 0 is recorded and checked for sanity.
- `train/lr`, `train/grad_norm`, `train/clip_frac` (share of steps where gradient clipping triggered).
- Loss spikes and NaN/Inf counts (alert on any NaN).
- `sys/throughput` (audio seconds per second), step time, GPU memory.
- `train/batch_*`: share of SSD, adult and each corpus in the batch; mean clip length.

**RLVR adds**
- `rl/reward_mean`, `rl/reward_std`, and **each reward part separately** (e.g. PER part, length part).
- `rl/kl` (and which estimator, k1 or k3, since they behave differently), `rl/entropy`, `rl/len_ratio` (output ÷ reference length).
- `rl/clip_frac` (share of updates clipped), importance-ratio mean/max, advantage mean/std.

## 2. Every N steps (tier 2)

**Validation (the numbers that pick checkpoints)**
- `val/per`: average within each tag group first, then across groups. Also each group: in-domain / Preston / SPROUT × typical / SSD; per corpus; per age band; per sex.
- PER split into substitutions, deletions, insertions.
- **`val/target_bias`**: on phones where the human IPA differs from the target, how often the model writes the *target* instead. Overall and per phoneme. **The most important number for the SSD goal.** Also error-detection recall and precision.
- `val/blank_rate` (share of frames predicted blank; early CTC training can collapse to all blanks), `val/len_ratio`, empty-output rate.
- Step-0 baseline for every val number, so every curve has a starting point.

**Layers (cheap ones)**
- Per layer: update-to-weight ratio (‖ΔW‖ ÷ ‖W‖; healthy roughly 1e-3), gradient norm, weight norm, distance from the start.
- With LoRA: norm of each adapter's update per layer and module.
- Weight and activation histograms (`wandb.Histogram`) for a few chosen layers.
- **Gradient signal-to-noise ratio** per layer (mean gradient ÷ spread across batches), plus the **gradient noise scale**, which says whether the batch size is sensible.

**Sample table** (same ~50 fixed val clips every time, across groups): clip ID, target, human IPA, model IPA, PER, target-bias hits. Only the clip ID (hashed) and the scores go to W&B; the IPA strings are shown in the local viewer (§0).

**RLVR adds**
- **Zero-variance share**: groups where every attempt scores the same (all wrong or all right). These teach nothing; plateaus usually come from them. Per tag group.
- **Reward by group**: SSD vs typical, separately. RL can raise the average by improving easy children while getting worse on SSD children.
- Output variety: number of distinct outputs among the K attempts per clip.
- Insertion/deletion balance against the human labels. Writing fewer phones can game PER.
- Val PER on clinician-labeled clips **never used in the reward**. Reward rising while this stays flat = reward hacking.
- If rollouts use a separate inference engine: the gap between rollout and trainer log-probabilities. A growing gap means training is quietly off-policy. (Likely not needed: our CTC rollouts can come from the trainer itself.)

## 3. Per checkpoint, offline (tier 3)

Run on a fixed probe set: the same frames at every layer and checkpoint, balanced across child/adult, typical/SSD, corpora, clean/noisy. Include the **untrained (random-init) model** and the **starting model** as baselines.

**What each layer knows**
- **Linear probes** per layer for phoneme identity, **with control tasks** (Hewitt & Liang 2019). A good probe scores high on the real task and low on the control task. Report selectivity, not just accuracy. Linear probes only.
- **Mutual information** (HuBERT method: k-means on each layer's frames, same k everywhere, normalized MI):
  - layer ↔ phoneme: should rise in the top layers;
  - layer ↔ corpus and layer ↔ speaker: should *not* rise (otherwise the model is learning recording quirks);
  - layer ↔ group (typical vs SSD): analysis only.
- **CKA** (layer similarity) vs the starting model: shows which layers SFT and RL actually changed. Cheaper than MI.
- Attention entropy per head; activation stats per layer.
- All of the above **split by input type** (child vs adult, typical vs SSD, clean vs noisy, each corpus).

**Output analysis**
- Phoneme confusion matrix (`wandb.plot.confusion_matrix` or a heatmap image).
- **Calibration**: reliability curve and ECE per phoneme. When the model is 90% sure, is it right 90% of the time? For screening, a confident wrong answer is worse than an unsure one.
- CTC alignment plots for a few clips.
- PER by audio SNR band; denoised vs plain-copy audio.

**Clinical endpoint (per child, not per clip)**
- Automated PCC (percent consonants correct, from model IPA vs target) vs clinician PCC (human IPA vs target): correlation across children.
- SSD vs typical separation per child: AUC.
- Broken down by age band, sex, and dialect or bilingual status where the metadata exists. A model can improve overall and get worse for one subgroup.
- The metric definitions are fixed before looking at test results. Pre-specified tests stay the headline; anything added after is labeled post-hoc.

**Evaluation rigor**
- **Uncertainty on every number:** bootstrap over children for 95% intervals; 2–3 seeds for key comparisons (LoRA vs full fine-tune, denoised vs plain). With few children, many "improvements" are noise.
- **Human ceiling:** PER between two clinicians on the same clips. Without it we can't tell when the model has hit the limit of the labels. Source to find: Providence has 10% retranscribed; check which other corpora have double transcriptions.
- **Forgetting check:** PER on held-out speech never trained on (adult speech once we have hand-labeled adult phones; until then, track it on held-out child corpora).
- **Input-independent baseline:** can silence, noise, or metadata alone (corpus, site, clip length) predict SSD? If yes, there is a recording-site confound and the AUC above is suspect.

## Reading list

- Karpathy, "A Recipe for Training Neural Networks".
- Google, "Deep Learning Tuning Playbook".
- Andy Jones, "Debugging RL, Without the Agonizing Pain".
- Huang et al., "The 37 Implementation Details of PPO" (ICLR blog track); its W&B project is a model of a well-instrumented RL run.
- Lilian Weng, "Reward Hacking in Reinforcement Learning".
- Hewitt & Liang (2019), "Designing and Interpreting Probes with Control Tasks".

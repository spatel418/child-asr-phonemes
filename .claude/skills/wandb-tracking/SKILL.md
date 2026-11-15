---
name: wandb-tracking
description: Use when adding, changing or checking Weights & Biases logging for SFT, RL or offline analysis runs - run names, groups, metric keys, artifacts, sample tables, alerts - or when reading W&B to judge a run. Enforces the numbers-only privacy rule.
---

# W&B tracking

What to log is listed in `TRACKING.md` (tiers 1–3). This skill covers **how**: naming, layout, privacy and checks.

## Privacy (check every time)

- W&B gets **numbers and hashed clip IDs only**.
- Never log: audio, spectrograms of child audio, IPA strings, transcripts, target words, child IDs in plain form, names, file paths that contain child IDs.
- Hash clip IDs with a salted hash. The salt stays on Quest (in `.env`), so the hash can't be reversed from W&B.
- Sample tables: hashed clip ID + scores only. The strings go to the local viewer.
- Before a new logging call ships, read what it sends. When in doubt, leave it out.

## Run identity

- `project`: from `WANDB_PROJECT`. `entity`: from `WANDB_ENTITY`. `dir`: `WANDB_DIR` (on /projects, not /home or /tmp).
- `name`: `<stage>-<short change>-<seed>`, e.g. `sft-lora16-lr3e-4-s0`. Stage is `smoke`, `overfit`, `sft`, `sweep`, `rl`, `analysis`.
- `group`: one per training run; its offline analysis runs join the same group with `job_type="analysis"`.
- Chained Slurm jobs **resume the same run** (`id=<run_id>`, `resume="must"`), so one run = one curve.
- `config` holds everything needed to rerun: full hyperparameters, git commit, dirty flag, seed, base model revision, split checksums, manifest checksums, Slurm job IDs (appended on each resume), package versions.
- `tags`: stage, and `baseline` / `candidate` / `final` where it applies.

## Keys

- Prefixes: `train/`, `val/`, `layers/`, `rl/`, `analysis/`, `clinical/`, `sys/`.
- One x-axis: optimizer step (`trainer/step`). Analysis runs log against the checkpoint's step via `wandb.define_metric`.
- Group breakdowns are part of the key: `val/per/preston_ssd`, `val/per/sprout_typical`, ...
- The selection score has one fixed key: `val/select_score`. Its definition is in the config.
- Every val number also gets a step-0 value.

## Artifacts

- Checkpoints: type `model` (or just metadata + path if too big to upload). Split and manifest files: type `dataset`, **metadata and checksums only, never the file contents**.
- Analysis runs `use_artifact` the checkpoint they read, so the link is recorded.

## Alerts

`wandb.alert` on: any NaN/Inf, loss spike above a set threshold, `val/blank_rate` > 0.95 after warmup, and (RL) zero-variance share above a set threshold.

## Logs on Quest

Slurm stdout should also print key numbers in one greppable line, e.g. `STEP 1200 train/loss=0.412 val/select_score=0.231`, so a run can be checked without W&B.

## Reading a run (report shape)

When asked to check a run, report:
1. Status (running / done / failed) and the W&B link.
2. Selection score now vs step 0 vs the baseline run.
3. Anything wrong: NaN, spikes, blank collapse, clip fraction high, throughput drop, group where PER got worse.
4. What it means and the one suggested next change.

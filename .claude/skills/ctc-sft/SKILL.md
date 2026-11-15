---
name: ctc-sft
description: Use when designing, writing, launching, debugging or reviewing supervised fine-tuning of the CTC phoneme model (WavLM-Large + CTC head, LoRA) - training code, smoke tests, overfit tests, baselines, hyperparameter sweeps, checkpoint selection, or reading why a run went wrong.
---

# CTC SFT: WavLM-Large → IPA with LoRA

Goal: a phoneme model that writes what the child said, not the target pronunciation. Training follows Karpathy's recipe: look at the data, then a working skeleton with dumb baselines, then overfit, regularize, tune. RL comes after this.

Data comes from the `phone-data` skill. Logging follows `wandb-tracking`. Plan details: `TRAINING_RECIPE.md`.

## Stages (do not skip)

1. **Skeleton + baselines** (no learning yet)
   - Fixed seed. Data loader, model, loss, val loop, logging all run end to end.
   - Check loss at step 0 is sane.
   - Dumb baselines on val, scored with the same code:
     - "write the target pronunciation" (where a target exists)
     - empty output / most common phone
     - input-independent model (silence or zeros in, same labels) → should do no better than chance
   - **Padding check:** the gradient for clip A must not depend on clip B in the batch (attention mask / padding leak).
2. **Smoke test:** about 50 steps on a tiny subset. Nothing NaN; loss falls; checkpoint saves and **resumes** to the same loss.
3. **Overfit:** 5 clips, then one batch, to near-zero loss and PER. If it can't, there's a bug. Fix it before going on.
4. **First real run** with the defaults below. One change at a time after that.
5. **Regularize** only once train PER is well below val PER (SpecAugment, dropout, layerdrop).
6. **Tune** with a sweep (below), not by hand.

## Defaults

- Base: `microsoft/wavlm-large`. Record the exact revision hash.
- **LoRA** on the transformer layers (start: rank 16, alpha 32, dropout 0, q/k/v/out projections). CNN front end frozen. CTC head trained in full.
- Two learning rates: adapters and head. Start with a short **head-only** stage (a few hundred steps), then adapters too.
- Warmup, then **constant LR** first. Add decay later as a separate change.
- Grad clip 1.0, bf16, EMA of weights 0.999 (evaluate the EMA copy too).
- Blank bias −2 at init (helps avoid the all-blank phase).
- Weight decay on the head only.
- In the skeleton and overfit stages: dropout, SpecAugment and layerdrop **off**.
- Batches by total audio seconds, not clip count. SSD children ~30–40% of examples, sampled by child with a cap on clips per child.
- Schedule in **steps**. Up to ~10 epochs, val about every quarter epoch, early stop on the selection score.
- 1 H100 per job. Code works with `torchrun` (DDP), but use one GPU unless a test shows more helps.
- Long runs = chained resumable Slurm jobs with the **same run ID**. The LR schedule continues; it doesn't restart.

## CTC traps

- A label longer than the clip's frame count (50 frames/s) cannot be learned: CTC gives inf loss. Use `zero_infinity=True` **and** count how often it happens; a non-zero count means a data problem.
- Never truncate audio and keep the full label. Skip or flag over-long clips (max ~20 s) instead.
- All-blank output early is normal for a short time. If `val/blank_rate` stays near 1, check the LR, the blank bias and the labels.
- Vocabulary = the shared phone set + blank (+ word boundary if used). Changing it means a new head, so it's a new run, not a resume.

## Choosing checkpoints

- Score = **val PER averaged within each tag group, then across groups** (in-domain / Preston / SPROUT × typical / SSD).
- Guard: SSD val **target-bias** must not get worse than the previous best.
- Held-out loss is only a proxy. Pick on PER and the guard, not CTC loss.
- Test sets: once, at the end, on the final pick only.

## Sweeps

1. LR range test first.
2. Random search in W&B (~30–40 short runs, Hyperband early stop). Ranges: adapter LR 1e-5–1e-3, head LR ratio, LoRA rank 8/16/32/64, alpha, LoRA dropout, which layers/modules, warmup, batch seconds; SpecAugment later.
3. Top 3 configs × 2–3 seeds, full length.
- Data-mix questions (Providence age 2, JIBO, TIMIT replay share) are separate experiments, not sweep parameters.
- Full fine-tuning: one run later, only to check whether LoRA is the limit.

## Every run

- Before: write the plan (what changes, why, what result would mean what) and the stop rule.
- Log: config, git commit, seed, split checksums, base model revision, Slurm job IDs.
- After: one row in `results.tsv`, lessons in the research log, then the `integrity-reviewer` agent.
- A failed setup isn't rerun unless something real changed.

## When a run looks wrong

Check in this order: data (listen to a few clips and read their labels in the viewer) → labels/vocab mapping → loss at step 0 → LR and grad norm → blank rate → padding/mask → resume logic. Change one thing, rerun the smoke test.

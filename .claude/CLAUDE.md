# child-asr-phonemes: rules for Claude

Read this first in every session. Private details (paths, accounts, ID lists) are in `CLAUDE.local.md`, which is git-ignored.

## What this project is

- A phoneme ASR model for children's speech: audio in, IPA of what the child **actually said** out.
- Use: help flag speech sound disorders (SSD) by comparing the target pronunciation with what the model hears.
- Plan: WavLM-Large + CTC head → SFT with LoRA → our own GRPO-style RL loop (reward = 1 − PER).
- Background: `DATA.md` (data, splits, preprocessing), `TRACKING.md` (what we log), `TRAINING_RECIPE.md` (training plan).

## How we work

- **Talk before building.** The user sets direction. Propose, then wait for a yes before writing large pieces or spending GPU time.
- **One change at a time.** Every run changes one thing from a known-good run, and the reason is written down first.
- **Every run follows the loop:** plan → check model ID and versions → smoke test (about 50 steps on a tiny subset, overfit a few clips) → full run → log → integrity check → lessons.
- **Look at the data before training on it.** Use the local viewer, not W&B.
- **Pre-set rules stay fixed.** Decide stop rules and success metrics before a run. A failed pre-set test stays a failure. Anything decided afterwards is labeled post-hoc.
- **Don't rerun a failed setup** unless something real changed. Write down why it failed before moving on.
- **Write it down.** Decisions and reasons go in the research log (append-only). One row per run goes in `results.tsv`.
- **Plain writing.** Short sentences, everyday words, in docs and replies.

## Data rules (never break these)

- Children's audio and transcripts **never leave the cluster**. No uploads to the Hugging Face Hub, HF Jobs or any other cloud service.
- **W&B gets numbers and hashed clip IDs only.** No audio, no IPA strings, no transcripts, no child names or IDs in plain form.
- **Never commit data:** no audio, manifests, transcripts or split files with child IDs. No credentials.
- **Raw data is read-only.** Copy, never move. Write outputs to `processed/`, `manifests/` and `splits/`.
- **Splits are by child, never by clip.** A child in two corpora gets one ID and goes in one split.
- **Sealed children are never trained on.** One SPROUT group is sealed for another study (list in `CLAUDE.local.md`). It may only be used in the final test.
- **Test sets are opened once, at the end.** Checkpoints are picked on val only.
- **Don't use DrivenData competition data.** Their released model is a benchmark only.
- Some files must never be shown to certain people. See `CLAUDE.local.md`.

## Code and environment

- Use the project Python environment (path in `CLAUDE.local.md`). Check `which python` before any `pip install`.
- Run Python that reads downloaded data with `-I`.
- `external/` holds other people's repos for reading only. Don't run code from it.
- GPU work goes through Slurm, not the login node.
- Unknown phone symbols **stop the script**. Never drop them silently.

## Git

- Local commits are fine. **Ask before every `git push`.**
- No AI co-author or attribution lines in commits or PRs.
- The repo is public. Before committing, check that nothing private is in the diff (see Data rules).

## Skills and agents

- Skills live in `.claude/skills/`, agents in `.claude/agents/`.
- Run the `integrity-reviewer` agent before trusting any result: leakage, sealed children, early test use, private data headed to W&B.

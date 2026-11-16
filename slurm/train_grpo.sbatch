#!/bin/bash
# GRPO on one H100. No chaining: runs are short; on the time limit it saves last.pt and a resubmit resumes.
#
#   sbatch slurm/train_grpo.sbatch RUNS/grpo-s0 --init RUNS/sft-r16-plain-word-s0 [extra train_grpo.py args]
#   sbatch --time=01:00:00 slurm/train_grpo.sbatch RUNS/probe --init RUNS/sft-r16-plain-word-s0 --probe 3000 --wandb off
#SBATCH --job-name=grpo
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /projects/p32576/simron/repos/pasketti-rlvr
RUN_DIR=$1; shift
PY=/projects/p32576/simron/envs/pasketti-phonetic/bin/python
export HF_HOME=/projects/p32576/simron/hf_cache HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
[ -f .env ] && eval "$(grep -E "^export WANDB_|^WANDB_[A-Z_]+=" .env | sed "s/^\(WANDB_\)/export \1/")"   # only WANDB_* settings from .env
# Queued behind an SFT run: refuse to start unless that run finished (SFT chains can end a job without DONE).
if [ -n "${REQUIRE_DONE:-}" ] && [ ! -f "$REQUIRE_DONE/DONE" ]; then echo "$REQUIRE_DONE has no DONE: not starting"; exit 1; fi
# Twin submissions (same run queued twice with different time limits): keep whichever starts first.
if [ -n "${TWIN_DIR:-}" ] && [ -f "$TWIN_DIR/config.json" ]; then echo "twin $TWIN_DIR already started; exiting"; exit 0; fi
[ -n "${TWIN_JOB:-}" ] && scancel "$TWIN_JOB" 2>/dev/null && echo "cancelled twin job $TWIN_JOB" || true
mkdir -p "$RUN_DIR"
echo "job $SLURM_JOB_ID host $(hostname) run $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set +e
$PY -I scripts/train_grpo.py --run-dir "$RUN_DIR" --time-limit-min "${TLIM:-230}" "$@"
RC=$?
set -e
for d in "$RUN_DIR"/wandb/offline-run-*; do   # upload anything logged offline (numbers only)
  [ -d "$d" ] && "$(dirname $PY)/wandb" sync "$d" >/dev/null 2>&1 && echo "synced $d to W&B" || true
done
exit "$RC"

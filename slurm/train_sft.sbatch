#!/bin/bash
# SFT on one H100, chained: each job runs until ~45 min before the 48 h limit, saves,
# and resubmits itself with the same run dir (same W&B run, schedule continues),
# until train_sft.py writes <run-dir>/DONE. A crash (non-zero exit) stops the chain.
#
#   sbatch slurm/train_sft.sbatch RUNS/sft-r16-s0 [extra train_sft.py args]
#   TLIM=25 NO_CHAIN=1 sbatch --time=00:40:00 slurm/train_sft.sbatch RUNS/smoke --max-steps 50 ...
#SBATCH --job-name=sft
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
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
mkdir -p "$RUN_DIR"
echo "job $SLURM_JOB_ID host $(hostname) git $(git rev-parse --short HEAD) run $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -I scripts/train_sft.py --run-dir "$RUN_DIR" --time-limit-min "${TLIM:-2835}" "$@"
if [ ! -f "$RUN_DIR/DONE" ] && [ -z "${NO_CHAIN:-}" ]; then
  N=$(ls "$RUN_DIR"/chain_* 2>/dev/null | wc -l)
  if [ "$N" -ge 10 ]; then echo "chain limit (10 jobs) reached; not resubmitting"; exit 0; fi
  touch "$RUN_DIR/chain_$SLURM_JOB_ID"
  sbatch "$0" "$RUN_DIR" "$@"
fi

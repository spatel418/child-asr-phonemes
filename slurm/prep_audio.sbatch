#!/bin/bash
# Audio prep for all usable clips (scripts/prep_audio.py): 64 CPU tasks.
#   sbatch slurm/prep_audio.sbatch
# Safe to resubmit: finished clips are skipped.
#SBATCH --job-name=prep_audio
#SBATCH --account=p32576
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --array=0-63
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd /projects/p32576/simron/repos/pasketti-rlvr
PY=/projects/p32576/simron/envs/pasketti-phonetic/bin/python
export OMP_NUM_THREADS=2
echo "git $(git rev-parse --short HEAD) host $(hostname) task $SLURM_ARRAY_TASK_ID"
$PY -I scripts/prep_audio.py --shard "$SLURM_ARRAY_TASK_ID" --nshards 64

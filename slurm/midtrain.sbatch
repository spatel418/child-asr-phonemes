#!/bin/bash
# Mid-training on one H100 (scripts/midtrain.py). On the time limit it saves last.pt; resubmit to resume.
#   sbatch slurm/midtrain.sbatch --kmeans --out RUNS/midtrain-v1
#   sbatch --time=08:00:00 --export=ALL,TLIM=470 slurm/midtrain.sbatch --train --out RUNS/midtrain-v1
#SBATCH --job-name=midtrain
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /projects/p32576/simron/repos/pasketti-rlvr
PY=/projects/p32576/simron/envs/pasketti-phonetic/bin/python
export HF_HOME=/projects/p32576/simron/hf_cache HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
echo "job $SLURM_JOB_ID host $(hostname) git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -I scripts/midtrain.py --time-limit-min "${TLIM:-230}" "$@"

#!/bin/bash
# Val re-score of a saved checkpoint.   sbatch slurm/eval_ckpt.sbatch RUNS/sft-r16-s0 [--ckpt last.pt]
#SBATCH --job-name=eval
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /projects/p32576/simron/repos/pasketti-rlvr
export HF_HOME=/projects/p32576/simron/hf_cache HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
/projects/p32576/simron/envs/pasketti-phonetic/bin/python -I scripts/eval_ckpt.py "$@"

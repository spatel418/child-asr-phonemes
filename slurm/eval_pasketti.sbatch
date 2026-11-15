#!/bin/bash
# pasketti-phonetic on our val split.   sbatch slurm/eval_pasketti.sbatch [--audio denoised]
#SBATCH --job-name=evalpk
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /projects/p32576/simron/repos/pasketti-rlvr
export HF_HOME=/projects/p32576/simron/hf_cache HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
/projects/p32576/simron/envs/pasketti-phonetic/bin/python -I scripts/eval_pasketti.py "$@"
# upload the eval numbers to W&B (numbers only; failure here does not fail the job)
HF_HUB_OFFLINE=1 WANDB_SILENT=true /projects/p32576/simron/envs/pasketti-phonetic/bin/python -I scripts/log_eval_wandb.py || echo "W&B upload failed; rerun scripts/log_eval_wandb.py"

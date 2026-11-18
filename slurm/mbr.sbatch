#!/bin/bash
# MWER step 1: N-best ceiling check (no training).   sbatch slurm/nbest.sbatch --init RUNS/x --out RUNS/nbest-x
#SBATCH --job-name=mbr
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
/projects/p32576/simron/envs/pasketti-phonetic/bin/python -I scripts/mbr_decode.py "$@"

#!/bin/bash
# Audio-judge step 2 (no training).  sbatch slurm/audio_judge.sbatch --clips <private clips.jsonl> --out-dir <private dir> [--limit N]
#SBATCH --job-name=audjudge
#SBATCH --account=p32576
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /projects/p32576/simron/repos/pasketti-rlvr
export HF_HOME=/projects/p32576/simron/hf_cache HF_HUB_OFFLINE=1
export CACHED_PATH_CACHE_ROOT=/projects/p32576/simron/hf_cache/cached_path
unset TRANSFORMERS_CACHE
/projects/p32576/simron/envs/tts-judge/bin/python -I scripts/audio_judge_check.py "$@"

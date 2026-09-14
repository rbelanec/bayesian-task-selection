#!/bin/bash
# One job per SOURCE task combination from get_task_combinations(SOURCE_TASKS) in
# src/eval/eval_target.py; each job sweeps the merge coefficient and evaluates the
# merged model on the TARGET tasks.
# eval_target.py picks its combination via $SLURM_ARRAY_TASK_ID (same indexing as
# eval.py / eval_array.sh).
#
# 12 source tasks → 4083 combinations of size >=2 (indices 0..4082). This exceeds
# MaxArraySize (1001) and the ~48 queued-job cap, so submit in batches via the
# wrapper (it sets COMBO_OFFSET per batch and waits between batches):
#   scripts/slurm/run_eval_batched.sh eval_target_array.sh
# The global combination index is COMBO_OFFSET + SLURM_ARRAY_TASK_ID.

#SBATCH --partition=gpu_short
#SBATCH --account=perun2601404
#SBATCH --qos=perun2601404
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --job-name=eval_target_array
#SBATCH -o logs_bts_merged/eval_target_array_%A_%a.out
#SBATCH -e logs_bts_merged/eval_target_array_%A_%a.err

# NOT `conda activate pf`: the miniconda install still has its pre-remount prefix
# baked in (etc/profile.d/conda.sh and bin/conda's shebang both point at the dead
# /mnt/data/home/$USER/miniconda3), so the hook fails before it can activate
# anything. The env's interpreter is a plain ELF binary and needs no activation,
# so call it directly and skip conda entirely.
PYTHON="/mnt/home/$USER/miniconda3/envs/pf/bin/python"
[[ -x "$PYTHON" ]] || { echo "No interpreter at $PYTHON" >&2; exit 1; }

# Same Lustre filesystem the cache has always lived on; it was remounted from
# /lustre/scratch to /mnt/scratch in early Aug 2026.
export HF_HOME="/mnt/scratch/$USER/huggingface"
# Every source/target dataset is already in $HF_HOME/datasets. Without this, all
# 48 concurrent array tasks re-resolve 12 datasets against the Hub on startup and
# some get a 429, which kills the job several GPU-minutes in. Offline = cache only.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs_bts_merged

echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

PYTHONPATH=src "$PYTHON" src/eval/eval_target.py config_templates/eval/eval.yaml

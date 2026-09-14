#!/bin/bash
# One job per task combination from get_task_combinations(TASKS) in src/eval/eval.py.
# eval.py picks its combination via $SLURM_ARRAY_TASK_ID.
#
# 12 base tasks → 4083 combinations of size >=2 (indices 0..4082). This exceeds
# MaxArraySize (1001) and the ~48 queued-job cap, so submit in batches via the
# wrapper (it sets COMBO_OFFSET per batch and waits between batches):
#   scripts/slurm/run_eval_batched.sh eval_array.sh
# The global combination index is COMBO_OFFSET + SLURM_ARRAY_TASK_ID.

#SBATCH --partition=gpu_short
#SBATCH --account=perun2601404
#SBATCH --qos=perun2601404
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --job-name=eval_array
#SBATCH -o logs_bts_merged/eval_array_%A_%a.out
#SBATCH -e logs_bts_merged/eval_array_%A_%a.err

# See eval_target_array.sh: conda's own scripts still point at the pre-remount
# prefix, so activation fails; call the env's interpreter directly instead.
PYTHON="/mnt/home/$USER/miniconda3/envs/pf/bin/python"
[[ -x "$PYTHON" ]] || { echo "No interpreter at $PYTHON" >&2; exit 1; }

export HF_HOME="/mnt/scratch/$USER/huggingface"
# See eval_target_array.sh: concurrent array tasks hitting the Hub get 429'd.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs_bts_merged

echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

PYTHONPATH=src "$PYTHON" src/eval/eval.py config_templates/eval/eval.yaml

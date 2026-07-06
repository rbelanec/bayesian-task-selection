#!/bin/bash
# One job per SOURCE task combination from get_task_combinations(SOURCE_TASKS) in
# src/eval/eval_target.py; each job sweeps the merge coefficient and evaluates the
# merged model on the TARGET tasks (mrpc, boolq, rte, cola).
# eval_target.py picks its combination via $SLURM_ARRAY_TASK_ID (same indexing as
# eval.py / eval_array.sh).
#
# Usage:
#   sbatch --array=0-25 scripts/slurm/eval_target_array.sh
# (5 source tasks → 26 combinations of size >=2 → indices 0..25)

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

eval "$(conda shell.bash hook)"
conda activate pf

export HF_HOME="/lustre/scratch/$USER/huggingface"

mkdir -p logs_bts_merged

echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

PYTHONPATH=src python src/eval/eval_target.py config_templates/eval/eval.yaml

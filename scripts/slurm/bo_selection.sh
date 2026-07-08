#!/bin/bash
# Online BO task selection (scripts/bo_task_selection.py): optimize the
# source-task subset per target with target SAR as the live objective,
# validated against exhaustive/greedy/random. CPU-only (SVDs over lora
# factors + GP fitting); loads task-vector checkpoints from
# saves_bts_preliminary + saves_pretrained_weights.
#   raw SAR -> figures/bo, iso-SAR -> figures/bo_iso
#
# Usage:
#   mkdir -p logs_analysis && sbatch scripts/slurm/bo_selection.sh

#SBATCH --partition=cpu_short
#SBATCH --account=perun2601404
#SBATCH --qos=perun2601404
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=bo_selection
#SBATCH -o logs_analysis/bo_selection_%j.out
#SBATCH -e logs_analysis/bo_selection_%j.err

eval "$(conda shell.bash hook)"
conda activate pf

set -e
export PYTHONPATH=src
python scripts/bo_task_selection.py --out figures/bo
python scripts/bo_task_selection.py --iso --out figures/bo_iso

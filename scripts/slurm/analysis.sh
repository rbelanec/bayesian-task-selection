#!/bin/bash
# CPU-only retrospective analyses on the precomputed target-transfer grid
# (figures/target/target_summary.csv):
#   - scripts/analysis_correlations.py  -> figures/analysis
#   - scripts/analysis_bo_variants.py   -> figures/bo_variants
#
# Usage:
#   mkdir -p logs_analysis && sbatch scripts/slurm/analysis.sh

#SBATCH --partition=cpu_short
#SBATCH --account=perun2601404
#SBATCH --qos=perun2601404
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=bts_analysis
#SBATCH -o logs_analysis/analysis_%j.out
#SBATCH -e logs_analysis/analysis_%j.err

eval "$(conda shell.bash hook)"
conda activate pf

set -e
export PYTHONPATH=src
python scripts/analysis_correlations.py
python scripts/analysis_bo_variants.py

#!/bin/bash
# Cosine-similarity task-selection baseline: one pass over the 25 checkpoints
# (1 pretrained + 12 source + 12 target fine-tunes) to build the 24x24 task-vector
# Gram matrix, then all 4083 mixtures x 12 targets in closed form.
#
# CPU-only and I/O bound (~118 GB of safetensors reads); no GPU is requested.
# The Gram matrix is cached to figures/target/target_cos_gram.npz, so re-emitting
# the CSV later (or with different signals) costs seconds:
#   PYTHONPATH=src python scripts/compute_task_cos.py
#
# Usage:
#   mkdir -p logs_analysis && sbatch scripts/slurm/task_cos.sh

#SBATCH --partition=cpu_short
#SBATCH --account=ANON_ACCOUNT
#SBATCH --qos=ANON_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --time=0-04:00:00
#SBATCH --mem=16G
#SBATCH --job-name=bts_task_cos
#SBATCH -o logs_analysis/task_cos_%j.out
#SBATCH -e logs_analysis/task_cos_%j.err

set -e

# See analysis.sh: conda's own scripts still point at the pre-remount prefix, so
# activation fails; the env's interpreter is a plain ELF binary and needs none.
PYTHON="/mnt/home/$USER/miniconda3/envs/pf/bin/python"
[[ -x "$PYTHON" ]] || { echo "No interpreter at $PYTHON" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

echo "node=$(hostname)  threads=$OMP_NUM_THREADS"
"$PYTHON" scripts/compute_task_cos.py \
    --model "${MODEL:-llama-3.2-1b-instruct}" \
    --seed "${SEED:-42}" \
    --method "${METHOD:-base}" \
    --out "${OUT:-figures/target/target_cos.csv}"

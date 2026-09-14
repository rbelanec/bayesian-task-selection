#!/bin/bash
# CPU-only retrospective analyses on the precomputed target-transfer grid
# (figures/target/target_summary.csv):
#   - scripts/analysis_correlations.py  -> figures/analysis      (~10 min)
#   - scripts/analysis_bo_variants.py   -> figures/bo_variants   (1-4 h of GP fits)
#
# Usage:
#   mkdir -p logs_analysis && sbatch scripts/slurm/analysis.sh
#   ONLY=bo_variants N_ITERATIONS=60 SEEDS=10 sbatch scripts/slurm/analysis.sh
#
# ONLY=bo_variants / ONLY=correlations runs just one half. bo_variants checkpoints
# per target to figures/bo_variants/bo_variants_partial.npz and resumes from it, so a
# wall-clock timeout costs one target rather than the whole run. Once it holds every
# target, `analysis_bo_variants.py --from-curves` redraws all figures in seconds.

#SBATCH --partition=cpu_short
#SBATCH --account=perun2601404
#SBATCH --qos=perun2601404
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --job-name=bts_analysis
#SBATCH -o logs_analysis/analysis_%j.out
#SBATCH -e logs_analysis/analysis_%j.err

set -e

# NOT `conda activate pf`: the miniconda install still has its pre-remount prefix
# baked in (etc/profile.d/conda.sh and bin/conda's shebang both point at the dead
# /mnt/data/home/$USER), so the hook fails before it can activate anything. The
# env's interpreter is a plain ELF binary and needs no activation.
PYTHON="/mnt/home/$USER/miniconda3/envs/pf/bin/python"
[[ -x "$PYTHON" ]] || { echo "No interpreter at $PYTHON" >&2; exit 1; }

# BO budget. The default of 10 was set when the grid held 26 combinations; at
# 4083 candidates 13 evaluations is 0.3% of the space and every method degenerates
# to noise, so override it deliberately:
#   N_ITERATIONS=60 sbatch scripts/slurm/analysis.sh
METHOD="${METHOD:-base}"
N_INITIAL="${N_INITIAL:-3}"
N_ITERATIONS="${N_ITERATIONS:-60}"
SEEDS="${SEEDS:-5}"

# Pin the linear-algebra thread count. 8 is not a guess — it is measured
# (logs_analysis/bench_threads_79420.out, one target, 15 BO iterations):
#
#   threads    1      2      4      8     16     64
#   s/fit   0.53   0.55   0.51   0.46   0.92   >16
#
# The GP fits here are tiny (<=63 training points, <=12 dims) and EI is one batched
# matmul, so past ~8 threads LAPACK's synchronisation costs more than the arithmetic
# it parallelises. Asking for 64 cores made the job ~14x SLOWER than 8, not faster.
# Do not raise this without re-running that benchmark.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# Unbuffered: bo_variants' progress was invisible until exit, so a job killed at the
# wall clock left no record of how far it had got.
export PYTHONUNBUFFERED=1

export PYTHONPATH=src
echo "method=$METHOD  budget=$(( N_INITIAL + N_ITERATIONS )) evals  seeds=$SEEDS"
echo "threads=$OMP_NUM_THREADS  node=$(hostname)"

ONLY="${ONLY:-all}"

if [[ "$ONLY" == "all" || "$ONLY" == "correlations" ]]; then
    "$PYTHON" scripts/analysis_correlations.py \
        --method "$METHOD" --n-initial "$N_INITIAL" \
        --n-iterations "$N_ITERATIONS" --seeds "$SEEDS"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "bo_variants" ]]; then
    "$PYTHON" scripts/analysis_bo_variants.py \
        --method "$METHOD" --n-initial "$N_INITIAL" \
        --n-iterations "$N_ITERATIONS" --seeds "$SEEDS"
fi

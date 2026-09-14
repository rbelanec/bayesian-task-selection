#!/bin/bash
# Target-SAR over the 4083 source combinations, sharded across an array.
#
# One GPU task per shard: shard i takes combos [i*PER, (i+1)*PER) in
# compute_target_sar.py's deterministic discovery order and writes its own CSV
# under figures/target/shards/. Shards never touch each other's file, so a
# failed shard is re-runnable on its own and --resume skips what it finished.
#
#   sbatch --array=0-47 scripts/slurm/target_sar_array.sh          # PER from N_SHARDS
#   sbatch --array=7    scripts/slurm/target_sar_array.sh          # redo one shard
#   bash scripts/slurm/target_sar_array.sh --merge                 # shards -> target_sar.csv
#
# N_SHARDS must match the --array size, and the array must fit the ~48 queued-job
# per-user cap (a larger --array is rejected outright with AssocMaxSubmitJobLimit).
# 48 shards x ~86 combos x ~24s is roughly 35 min per shard; arrays here run
# ~15-wide, so the wall time is ~3-4 waves.
#
# The single-process alternative (no array) is ~27 h:
#   PYTHONPATH=src $PYTHON scripts/compute_target_sar.py --device cuda

#SBATCH --partition=gpu_short
#SBATCH --account=ANON_ACCOUNT
#SBATCH --qos=ANON_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --job-name=target_sar
#SBATCH -o logs_analysis/target_sar_%A_%a.out
#SBATCH -e logs_analysis/target_sar_%A_%a.err

set -e

REPO=/path/to/bayesian-task-selection
SHARD_DIR="$REPO/figures/target/shards"
OUT="$REPO/figures/target/target_sar.csv"
N_SHARDS="${N_SHARDS:-48}"
N_COMBOS="${N_COMBOS:-4083}"

# ---------------------------------------------------------------- merge mode
# Concatenates the shards into the single CSV the rest of the pipeline reads
# (summarize_target.py joins it on method+combo+target).
if [[ "$1" == "--merge" ]]; then
    cd "$REPO"
    shopt -s nullglob
    shards=("$SHARD_DIR"/target_sar_shard_*.csv)
    (( ${#shards[@]} )) || { echo "no shards in $SHARD_DIR" >&2; exit 1; }
    head -n 1 "${shards[0]}" > "$OUT"
    for s in "${shards[@]}"; do tail -n +2 "$s"; done >> "$OUT"
    rows=$(( $(wc -l < "$OUT") - 1 ))
    combos=$(tail -n +2 "$OUT" | cut -d, -f2 | sort -u | wc -l)
    echo "merged ${#shards[@]} shards -> $OUT ($rows rows, $combos distinct combos)"
    expected=$(( N_COMBOS * 12 ))
    if (( rows != expected )); then
        echo "WARNING: expected $expected rows ($N_COMBOS combos x 12 targets)." >&2
        echo "         Re-run the missing shards before using this file." >&2
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------- shard mode
cd "$REPO"

# conda's hook still has its pre-remount prefix baked in and fails before it can
# activate; the env interpreter is a plain ELF binary that needs no activation.
PYTHON="/mnt/home/$USER/miniconda3/envs/pf/bin/python"
[[ -x "$PYTHON" ]] || { echo "No interpreter at $PYTHON" >&2; exit 1; }

export HF_HOME="/mnt/scratch/$USER/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs_analysis "$SHARD_DIR"

PER=$(( (N_COMBOS + N_SHARDS - 1) / N_SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))

if (( START >= N_COMBOS )); then
    echo "shard $SLURM_ARRAY_TASK_ID starts at $START, past the last combo — nothing to do"
    exit 0
fi

echo "shard $SLURM_ARRAY_TASK_ID/$N_SHARDS: combos $START..$(( START + PER - 1 )) (PER=$PER)"

PYTHONPATH=src "$PYTHON" scripts/compute_target_sar.py \
    --device cuda \
    --start "$START" \
    --limit "$PER" \
    --out "$SHARD_DIR/target_sar_shard_${SLURM_ARRAY_TASK_ID}.csv"

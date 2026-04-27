#!/bin/bash
# Usage: merged_eval.sh [START [COUNT]]
#   START  index of first job to submit (default: 0)
#   COUNT  number of jobs to submit     (default: all remaining)

START=${1:-0}
COUNT=${2:-}

# evaluation of merged task-vector models
source_datasets=(mnli qnli qqp sst2 record)
peft_methods=(lora)
models=(llama-3.2-1b-instruct)

saves_output_dir="saves_bts_merged"
logging_dir="logs_bts_merged"
seeds=(42)

# generate all combinations of 2+ source datasets
generate_combinations() {
    local -a arr=("$@")
    local n=${#arr[@]}
    for ((size=2; size<=n; size++)); do
        # iterate bitmask over all subsets of given size
        for ((mask=0; mask<(1<<n); mask++)); do
            local count=0
            for ((i=0; i<n; i++)); do
                (( (mask >> i) & 1 )) && ((count++))
            done
            if [[ $count -eq $size ]]; then
                local combo=()
                for ((i=0; i<n; i++)); do
                    (( (mask >> i) & 1 )) && combo+=("${arr[$i]}")
                done
                echo "${combo[*]}"
            fi
        done
    done
}

# count runs before submitting
total_runs=0
skipped_combos=0
echo "=== Merged eval run summary ==="
for s in "${seeds[@]}"; do
    for m in "${models[@]}"; do
        for pm in "${peft_methods[@]}"; do
            while IFS=' ' read -ra combo; do
                merged_name=$(IFS=_; echo "${combo[*]}")
                merged_checkpoint="${saves_output_dir}/${pm}/${m}/${merged_name}_${s}"
                if [[ ! -d "$merged_checkpoint" ]]; then
                    ((skipped_combos++))
                else
                    ((total_runs += ${#combo[@]}))
                fi
            done < <(generate_combinations "${source_datasets[@]}")
        done
    done
done
echo "  Seeds:           ${seeds[*]}"
echo "  Models:          ${models[*]}"
echo "  Methods:         ${peft_methods[*]}"
echo "  Combinations:    $(generate_combinations "${source_datasets[@]}" | wc -l) total, ${skipped_combos} missing checkpoints"
echo "  Jobs to submit:  ${total_runs}"
echo "================================"
echo ""

# collect all (merged_checkpoint, dataset, model, method, seed) tuples
declare -a JOB_MERGED JOB_DATASET JOB_MODEL JOB_METHOD JOB_SEED
for s in "${seeds[@]}"; do
    for m in "${models[@]}"; do
        for pm in "${peft_methods[@]}"; do
            while IFS=' ' read -ra combo; do
                merged_name=$(IFS=_; echo "${combo[*]}")
                merged_checkpoint="${saves_output_dir}/${pm}/${m}/${merged_name}_${s}"
                if [[ ! -d "$merged_checkpoint" ]]; then
                    continue
                fi
                for d in "${combo[@]}"; do
                    JOB_MERGED+=("$merged_checkpoint")
                    JOB_DATASET+=("$d")
                    JOB_MODEL+=("$m")
                    JOB_METHOD+=("$pm")
                    JOB_SEED+=("$s")
                done
            done < <(generate_combinations "${source_datasets[@]}")
        done
    done
done

end=$(( COUNT ? START + COUNT : ${#JOB_MERGED[@]} ))
end=$(( end > ${#JOB_MERGED[@]} ? ${#JOB_MERGED[@]} : end ))

echo "Submitting jobs ${START} to $((end - 1)) of $((${#JOB_MERGED[@]} - 1))"
echo ""

for (( i=START; i<end; i++ )); do
    merged_checkpoint="${JOB_MERGED[$i]}"
    d="${JOB_DATASET[$i]}"
    m="${JOB_MODEL[$i]}"
    pm="${JOB_METHOD[$i]}"
    s="${JOB_SEED[$i]}"
    merged_name=$(basename "$merged_checkpoint" | sed "s/_${s}$//")

    echo "  [$i] merged=${merged_name}  eval_dataset=${d}  seed=${s}"
    TIMESTAMP=$(date +%s)
    OUTPUT_DIR="${saves_output_dir}/${pm}/${m}/eval_${merged_name}_on_${d}_${s}_${TIMESTAMP}"
    WANDB_NAME="${pm}_${m}_eval_merged_${merged_name}_on_${d}_${s}_${TIMESTAMP}"
    ADAPTER="${merged_checkpoint}"
    DATASET="${d}_eval"
    SEED="${s}"
    WANDB_PROJECT="bts_merged_eval"

    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${logging_dir}"

    export OUTPUT_DIR WANDB_NAME ADAPTER DATASET SEED WANDB_PROJECT
    envsubst < config_templates/${pm}/${m}/eval.yaml > "${OUTPUT_DIR}/eval.yaml"

    sbatch \
        --job-name "${merged_name}_${d}_${s}" \
        -o "${logging_dir}/${merged_name}_${d}_${s}_${TIMESTAMP}.out" \
        -e "${logging_dir}/${merged_name}_${d}_${s}_${TIMESTAMP}.err" \
        scripts/slurm/run_eval.sh "${OUTPUT_DIR}/eval.yaml"

    sleep 1
done

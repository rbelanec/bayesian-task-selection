#!/bin/bash

datasets=(mnli qnli qqp sst2 record mrpc boolq rte cola)
peft_methods=(zero-shot)
models=(llama-3.2-1b-instruct)
saves_output_dir="saves_bts_preliminary"
logging_dir="logs_bts_preliminary"
seeds=(42)

for s in ${seeds[@]};
do
    for d in ${datasets[@]};
    do
        for m in ${models[@]};
        do
            for pm in ${peft_methods[@]};
            do
                TIMESTAMP=`date +%s`
                SEED="${s}"
                WANDB_PROJECT="bts_preliminary"
                OUTPUT_DIR="${saves_output_dir}/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}"
                WANDB_NAME="${pm}_${m}_eval_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}_eval"

                mkdir -p ${OUTPUT_DIR}
                mkdir -p ${logging_dir}

                export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME

                envsubst < config_templates/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

                sbatch --job-name ${pm}_${m}_${d}_${s}_${TIMESTAMP} -o ${logging_dir}/${pm}_${m}_${d}_${s}_${TIMESTAMP}.out -e ${logging_dir}/${pm}_${m}_${d}_${s}_${TIMESTAMP}.err scripts/slurm/run_eval.sh ${saves_output_dir}/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}/eval.yaml

                sleep 1
            done
        done
    done
done
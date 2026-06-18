#!/bin/bash

datasets=(sst2)
# peft_methods=(lora base)
peft_methods=(freeze)
models=(llama-3.2-1b-instruct)
saves_output_dir="saves_bts_preliminary"
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
                OUTPUT_DIR="${saves_output_dir}/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}"
                SEED="${s}"
                WANDB_PROJECT="bts_preliminary"
                WANDB_NAME="${pm}_${m}_train_${d}_${s}_${TIMESTAMP}"


                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME
                envsubst < config_templates/${pm}/${m}/train.yaml > ${OUTPUT_DIR}/train.yaml

                OUTPUT_DIR="${saves_output_dir}/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}"
                WANDB_NAME="${pm}_${m}_eval_${d}_${s}_${TIMESTAMP}"
                ADAPTER="${saves_output_dir}/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}_eval"

                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR WANDB_NAME ADAPTER DATASET
                envsubst < config_templates/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

                llamafactory-cli train ${saves_output_dir}/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}/train.yaml
                llamafactory-cli train ${saves_output_dir}/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}/eval.yaml
            done
        done
    done
done
#!/bin/bash

# peft_methods=(lora base)  # base/lora snapshots already generated (or symlink them)
peft_methods=(freeze)
models=(llama-3.2-1b-instruct)
seeds=(42)

saves_output_dir="saves_pretrained_weights"

EPOCHS=0

for s in ${seeds[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            TIMESTAMP=`date +%s`
            OUTPUT_DIR="${saves_output_dir}/${pm}/${m}/pretrained_weights_${s}"
            DATASET="sst2" # dumy dataset, not used for training since EPOCHS=0
            SEED="${s}"

            mkdir -p ${OUTPUT_DIR}

            export OUTPUT_DIR DATASET SEED EPOCHS
            envsubst < config_templates/${pm}/${m}/pre.yaml > ${OUTPUT_DIR}/pre.yaml

            llamafactory-cli train ${OUTPUT_DIR}/pre.yaml
        done
    done
done
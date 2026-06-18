#!/bin/bash

#SBATCH --partition=gpu_short
#SBATCH --account=perun2601404
#SBATCH --qos=perun2601404
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=48G

eval "$(conda shell.bash hook)"
conda activate pf

export HF_HOME="/lustre/scratch/$USER/huggingface"

llamafactory-cli train $1
llamafactory-cli train $2
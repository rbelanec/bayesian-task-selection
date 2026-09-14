#!/bin/bash

#SBATCH --partition=gpu_short
#SBATCH --account=ANON_ACCOUNT
#SBATCH --qos=ANON_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=48G

eval "$(conda shell.bash hook)"
conda activate pf

export HF_HOME="/mnt/scratch/$USER/huggingface"

llamafactory-cli train $1
#!/bin/bash
#SBATCH --job-name=YOUR_JOB_NAME
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/%j.out   # %j = job ID
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/%j.err
#SBATCH --time=HH:MM:SS           # max runtime, e.g. 12:00:00
#SBATCH --mem=XXG                 # RAM, e.g. 32G or 64G
#SBATCH --cpus-per-task=X         # CPU cores, e.g. 4
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:X              # remove this line if no GPU needed

mkdir -p /mnt/home/cdyer/colorcorrection/logs

conda activate YOUR_ENV_NAME

python /mnt/home/cdyer/colorcorrection/code/YOUR_SCRIPT.py \
    --arg1 value1 \
    --arg2 value2

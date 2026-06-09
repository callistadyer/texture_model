#!/bin/bash
#SBATCH --job-name=prep_imagenet
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/prep_imagenet_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/prep_imagenet_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=YOUR_PARTITION

mkdir -p /mnt/home/cdyer/colorcorrection/logs

conda activate YOUR_ENV_NAME

python /mnt/home/cdyer/colorcorrection/code/prep_imagenet_from_zip.py \
    --src_dir /mnt/home/gkrawezik/ceph/AI_DATASETS/imagenet/ \
    --dst_dir /mnt/home/cdyer/colorcorrection/

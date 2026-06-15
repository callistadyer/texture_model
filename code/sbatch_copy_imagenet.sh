#!/bin/bash
#SBATCH --job-name=copy_imagenet
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/copy_imagenet_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/copy_imagenet_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=64

mkdir -p /mnt/home/cdyer/colorcorrection/logs

source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

cd /mnt/home/cdyer/colorcorrection/images/imagenet/train/
unzip -qq ../train.zip 

cd ../val
unzip -qq ../val.zip

python /mnt/home/cdyer/colorcorrection/code/prep_imagenet.py

# cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet /mnt/home/cdyer/colorcorrection/images

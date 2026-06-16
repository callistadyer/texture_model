#!/bin/bash
#SBATCH --job-name=imagenet_pipeline
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/imagenet_pipeline_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/imagenet_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=64

mkdir -p /mnt/home/cdyer/colorcorrection/logs

source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

# skip unzip if train/ already has contents
if [ -z "$(ls -A /mnt/home/cdyer/colorcorrection/images/imagenet/train/)" ]; then
    echo "[$(date)] Step 1: unzipping train.zip"
    cd /mnt/home/cdyer/colorcorrection/images/imagenet/train/
    unzip -qq ../train.zip
    echo "[$(date)] train.zip done"
else
    echo "[$(date)] Step 1: train/ already unzipped, skipping"
fi

# skip unzip if val/ already has contents
if [ -z "$(ls -A /mnt/home/cdyer/colorcorrection/images/imagenet/val/)" ]; then
    echo "[$(date)] Step 2: unzipping val.zip"
    cd /mnt/home/cdyer/colorcorrection/images/imagenet/val/
    unzip -qq ../val.zip
    echo "[$(date)] val.zip done"
else
    echo "[$(date)] Step 2: val/ already unzipped, skipping"
fi

# skip prep if .pt files already exist
if [ -f "/mnt/home/cdyer/colorcorrection/images/imagenet/train_80x80_color_list.pt" ]; then
    echo "[$(date)] Step 3: .pt files already exist, skipping prep"
else
    echo "[$(date)] Step 3: running prep_imagenet.py"
    cd /mnt/home/cdyer/colorcorrection/texture_model/code
    python prep_imagenet.py
    echo "[$(date)] prep done"
fi

echo "[$(date)] Step 4: running main.py (training)"
cd /mnt/home/cdyer/colorcorrection/texture_model/code
python main.py
echo "[$(date)] training done"

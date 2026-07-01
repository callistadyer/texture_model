#!/bin/bash
#SBATCH --job-name=imagenet_dichromacy_pipeline
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/imagenet_dichromacy_pipeline_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/imagenet_dichromacy_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=64

mkdir -p /mnt/home/cdyer/colorcorrection/logs

module load cuda/12.5.1
source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

# Copy preprocessed dichromacy .pt files to fast node-local storage before training
if [ -f "/mnt/home/cdyer/ceph/images/imagenet/train_80x80_color_dichromacy_deuteranopia_list.pt" ]; then
    echo "[$(date)] Step 1: copying dichromacy .pt files to /tmp..."
    mkdir -p /tmp/imagenet
    cp /mnt/home/cdyer/ceph/images/imagenet/train_80x80_color_dichromacy_deuteranopia_list.pt /tmp/imagenet/
    cp /mnt/home/cdyer/ceph/images/imagenet/test_80x80_color_dichromacy_deuteranopia_list.pt /tmp/imagenet/
    echo "[$(date)] copy done"
else
    echo "[$(date)] ERROR: dichromacy .pt files not found on ceph. Run prep_imagenet_dichromacy.py first to generate them."
    exit 1
fi

# Step 2: train on dichromacy images
# --data_name imagenet_dichromacy loads the dichromacy .pt files
# --optional_dir_label tags the save folder so it is distinct from the trichromat run
# model weights are saved as model_dichromacy.pt inside the results folder
echo "[$(date)] Step 2: running main.py (dichromacy training)"
cd /mnt/home/cdyer/colorcorrection/texture_model/code
python -u main.py \
    --data_name imagenet_dichromacy \
    --num_epochs 45 \
    --data_root_path /tmp/ \
    --optional_dir_label color_no_skip_deep_enc_dichromacy
echo "[$(date)] dichromacy training done"

#!/bin/bash
#SBATCH --job-name=imagenet_pipeline
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/imagenet_pipeline_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/imagenet_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

mkdir -p /mnt/home/cdyer/colorcorrection/logs

# Callista edit: load CUDA 12.5.1 to match the GPU driver on the worker nodes
module load cuda/12.5.1
source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

# Copy preprocessed .pt files to fast node-local storage before training.
if [ -f "/mnt/home/cdyer/ceph/images/imagenet/train_80x80_color_list_linearized.pt" ]; then
    echo "[$(date)] Step 1: copying linearized .pt files to /tmp..."
    mkdir -p /tmp/imagenet
    cp /mnt/home/cdyer/ceph/images/imagenet/train_80x80_color_list_linearized.pt /tmp/imagenet/
    cp /mnt/home/cdyer/ceph/images/imagenet/test_80x80_color_list_linearized.pt /tmp/imagenet/
    echo "[$(date)] copy done"
else
    echo "[$(date)] ERROR: linearized .pt files not found on ceph. Run prep_imagenet.py first to generate them."
    exit 1
fi

echo "[$(date)] Step 2: running main.py (training)"
cd /mnt/home/cdyer/colorcorrection/texture_model/code
python -u main.py --num_epochs 500 --data_root_path /tmp/ \
    --optional_dir_label color_no_skip_deep_dec_inv_sqrt
echo "[$(date)] training done"

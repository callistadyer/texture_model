#!/bin/bash
#SBATCH --job-name=imagenet_splithalves
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/imagenet_splithalves_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/imagenet_splithalves_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=64

mkdir -p /mnt/home/cdyer/colorcorrection/logs

module load cuda/12.5.1
source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

cd /mnt/home/cdyer/colorcorrection/texture_model/code

# Copy preprocessed .pt files to fast node-local storage before training
if [ -f "/mnt/home/cdyer/ceph/images/imagenet/train_80x80_color_list.pt" ]; then
    echo "[$(date)] Copying .pt files to /tmp..."
    mkdir -p /tmp/imagenet
    cp /mnt/home/cdyer/ceph/images/imagenet/train_80x80_color_list.pt /tmp/imagenet/
    cp /mnt/home/cdyer/ceph/images/imagenet/test_80x80_color_list.pt /tmp/imagenet/
    echo "[$(date)] copy done"
else
    echo "[$(date)] ERROR: .pt files not found on ceph. Run prep_imagenet.py first to generate them."
    exit 1
fi

# Sanity check: verify the split is correct before wasting compute on training
echo "[$(date)] Checking split sizes..."
python -c "
import torch
full = torch.load('/tmp/imagenet/train_80x80_color_list.pt', weights_only=True)

# 1. total image counts per split
countA = sum(len(d) // 2 for d in full)
countB = sum(len(d) - len(d) // 2 for d in full)
print(f'  splitA images: {countA}')
print(f'  splitB images: {countB}')
print(f'  difference:    {abs(countA - countB)}')

# 2. A + B = full for every class (no images dropped or duplicated)
errors = 0
for i, d in enumerate(full):
    nA = len(d) // 2
    nB = len(d) - len(d) // 2
    if nA + nB != len(d):
        print(f'  ERROR class {i}: {nA} + {nB} != {len(d)}')
        errors += 1
if errors == 0:
    print(f'  A+B=full check passed for all {len(full)} classes')

# 3. A and B contain different images (spot-check first class)
d0 = full[0]
A0 = d0[:len(d0)//2]
B0 = d0[len(d0)//2:]
pixel_diff = (A0[0] - B0[0]).abs().mean().item()
print(f'  pixel diff between first image of A and B in class 0: {pixel_diff:.4f} (should be > 0)')
"

# Step 1: train on first half of images from each class
echo "[$(date)] Step 1: training splitA (first half of each class)"
python -u main.py \
    --data_name imagenet_splitA \
    --num_epochs 45 \
    --data_root_path /tmp/ \
    --optional_dir_label color_no_skip_deep_dec_inv_sqrt_splitA
echo "[$(date)] splitA done"

# Step 2: train on second half of images from each class
echo "[$(date)] Step 2: training splitB (second half of each class)"
python -u main.py \
    --data_name imagenet_splitB \
    --num_epochs 45 \
    --data_root_path /tmp/ \
    --optional_dir_label color_no_skip_deep_dec_inv_sqrt_splitB
echo "[$(date)] splitB done"

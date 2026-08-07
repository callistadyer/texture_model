#!/bin/bash
#SBATCH --job-name=prep_imagenet
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/prep_imagenet_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/prep_imagenet_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

mkdir -p /mnt/home/cdyer/colorcorrection/logs

source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

# prep_imagenet.py reads raw JPEGs from zkadkhodaie's imagenet folder (images_dir_path)
# and writes train_80x80_color_list_linearized.pt / test_80x80_color_list_linearized.pt
# to our own ceph folder (output_dir_path) - see prep_imagenet.py for both paths.
echo "[$(date)] Step 1: running prep_imagenet.py"
cd /mnt/home/cdyer/colorcorrection/texture_model/code
python -u prep_imagenet.py
echo "[$(date)] prep_imagenet.py done"

# Check number of images
echo "[$(date)] verifying saved .pt files"
python -u <<'EOF'
import torch

output_dir_path = '/mnt/home/cdyer/ceph/images/imagenet/'

# Each .pt file is a list of tensors, one tensor per ImageNet class, each
# shaped (N_class, 3, 80, 80). Total images = sum of N_class across the list.
train_sets = torch.load(output_dir_path + 'train_80x80_color_list_linearized.pt', weights_only=True)
train_num_classes = len(train_sets)
train_num_images = sum(class_tensor.shape[0] for class_tensor in train_sets)

test_sets = torch.load(output_dir_path + 'test_80x80_color_list_linearized.pt', weights_only=True)
test_num_classes = len(test_sets)
test_num_images = sum(class_tensor.shape[0] for class_tensor in test_sets)

print(f'[verify] train_80x80_color_list_linearized.pt: {train_num_classes} classes, {train_num_images} images')
print(f'[verify] test_80x80_color_list_linearized.pt:  {test_num_classes} classes, {test_num_images} images')
print(f'[verify] total images across both files: {train_num_images + test_num_images}')
EOF
echo "[$(date)] Step 2 done"

#!/bin/bash
#SBATCH --job-name=move_images
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/move_images_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/move_images_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=64

mkdir -p /mnt/home/cdyer/colorcorrection/logs

echo "[$(date)] Starting rsync..."
rsync -av --progress /mnt/home/cdyer/colorcorrection/images/ /mnt/home/cdyer/ceph/images/

echo "[$(date)] rsync done. Verifying file counts..."
src_count=$(find /mnt/home/cdyer/colorcorrection/images/ -type f | wc -l)
dst_count=$(find /mnt/home/cdyer/ceph/images/ -type f | wc -l)
echo "  source:      $src_count files"
echo "  destination: $dst_count files"

if [ "$src_count" -eq "$dst_count" ]; then
    echo "[$(date)] File counts match. Safe to delete source."
    rm -rf /mnt/home/cdyer/colorcorrection/images/
    echo "[$(date)] Source deleted."
else
    echo "[$(date)] WARNING: file counts differ -- NOT deleting source. Check the log."
fi

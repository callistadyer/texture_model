#!/bin/bash
#SBATCH --job-name=plot_psnr
#SBATCH --output=/mnt/home/cdyer/colorcorrection/logs/plot_psnr_%j.out
#SBATCH --error=/mnt/home/cdyer/colorcorrection/logs/plot_psnr_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=8

mkdir -p /mnt/home/cdyer/colorcorrection/logs

module load cuda/12.5.1
source /mnt/home/cdyer/colorcorrection/colorcorrection_env/bin/activate

cd /mnt/home/cdyer/colorcorrection/texture_model/code

echo "[$(date)] Starting plot_psnr.py"
python -u plot_psnr.py \
    --models splitA splitB UNet_full conv3 \
    --output /mnt/home/cdyer/colorcorrection/texture_model/psnr_curves_full.png
echo "[$(date)] Done"

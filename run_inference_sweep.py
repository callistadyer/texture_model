"""
run_inference_sweep.py

What this script does, in plain terms:
  1. Load one image.
  2. Load a trained denoising model (either "UNet" or "conv3").
  3. For several different amounts of noise, add noise to the image and then
     ask the model to remove it ("denoise" it).
  4. Measure how good the result is before and after denoising, using PSNR 
  5. Save all the numbers to a CSV file 
  6. Build one big image showing every noise level
"""

import os       
import sys      
import argparse
import csv     

import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# This project's code lives in a couple of different folders
# ----------------------------------------------------------------------------
sys.path.insert(0, '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/code')
sys.path.insert(0, '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction')

import torch
from model_loader_func import load_learned_model
from quality_metrics_func import batch_ave_psnr_torch  # computes PSNR (image quality score, see below)
from models.denoiser import Denoiser                 # the "conv3" model's network architecture
from utils.helper import parse_args as load_conv3_args  # single source of truth for conv3's architecture
from PIL import Image
import torchvision.transforms as T


# ----------------------------------------------------------------------------
# conv3 LQ's already-trained model (not trained by this project).
# Unlike the UNet models, it doesn't come with a saved "exp_arguments.pkl" file
# describing its architecture. Its architecture is loaded the same way
# Denoiser_Reconstruction/recon_visualize_dichromat.ipynb and code/plot_psnr.py
# load it: via Denoiser_Reconstruction/utils/helper.py::parse_args(), which contains 
# default architecture (padding, num_kernels,
# kernel_size, num_layers, im_channels) - it also returns a bunch of unrelated
# dataset/training defaults that Denoiser simply ignores.
# ----------------------------------------------------------------------------
conv3_args = load_conv3_args()
conv3_weights_path = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction/assets/conv3_ln.pt'

# ----------------------------------------------------------------------------
# Where results get saved
# ----------------------------------------------------------------------------
results_dir = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/runInferenceResults'
grids_dir = os.path.join(results_dir, 'grids')

# Set this to False if you just want the PSNR numbers and don't want to
# spend time building/saving/showing the image grid.
PLOT = True


def main():
    # ------------------------------------------------------------------------
    # STEP 1 - read command-line options.
    # Example of running this script:
    #   python run_inference_sweep.py --image flower.png --model_type conv3
    # argparse automatically reads whatever comes after "python run_inference_sweep.py"
    # on the command line and turns it into the "args" object used below.
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, required=True, help='path to the input image file, e.g. Denoiser_Reconstruction/flower1.png')
    parser.add_argument('--model_type', type=str, choices=['unet', 'conv3'], default='unet', help="which model to use: 'unet' (trained by this project) or 'conv3' (outside, pre-trained model used for comparison)")
    parser.add_argument('--model_dir', type=str, default='/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/models_trained/UNet_full_240541imgs_1000epochs/', help='folder containing model.pt and exp_arguments.pkl. Only used when --model_type is unet.')
    # nargs='+' means "accept one or more numbers", e.g. --noise_levels 10 50 100
    parser.add_argument('--noise_levels', type=int, nargs='+', default=[10, 25, 50, 75, 100, 150, 200], help='list of noise levels (0-255) to test, from least to most noisy. Defaults to 7 levels.')

    args = parser.parse_args()

    # Make sure the output folders exist before we try to save anything into them.
    # exist_ok=True means "don't raise an error if the folder is already there".
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 2 - load whichever model was asked for.
    # ------------------------------------------------------------------------
    if args.model_type == 'unet':
        print(f'Loading UNet model from {args.model_dir}')
        model = load_learned_model(args.model_dir)
        model.eval()  # "eval mode" turns off training-only behavior (like dropout); always do this before using a model
        # Turn ".../models_trained/UNet_full_240541imgs_1000epochs/" into just "UNet_full_240541imgs_1000epochs"
        model_dir_without_trailing_slash = args.model_dir.rstrip('/')
        model_label = os.path.basename(model_dir_without_trailing_slash)

    else:  # args.model_type == 'conv3'
        print(f'Loading conv3 model from {conv3_weights_path}')
        model = Denoiser(conv3_args)
        # torch.load reads the saved weights (numbers) from disk into memory.
        state_dict = torch.load(conv3_weights_path, map_location='cpu', weights_only=False)
        # load_state_dict copies those saved numbers into our freshly-created model.
        model.load_state_dict(state_dict)
        model.eval()
        model_label = 'conv3ln'

    # ------------------------------------------------------------------------
    # STEP 3 - load the image and get it ready for the model.
    # ------------------------------------------------------------------------
    print(f'Loading image from {args.image}')
    img = Image.open(args.image).convert('RGB')  # convert('RGB') makes sure it has exactly 3 color channels

    # The UNet has 3 pooling layers, and each one halves the image's width and
    # height (2 x 2 x 2 = 8). So width and height must both be a multiple of 8,
    # otherwise the shapes won't line up when the image is put back together
    # on the way out, and the model will crash. We crop off a few leftover
    # pixels here so that both dimensions divide evenly by 8.
    original_width, original_height = img.size
    new_width = (original_width // 8) * 8
    new_height = (original_height // 8) * 8
    img = img.resize((new_width, new_height), Image.BICUBIC)

    # Convert the image into the format PyTorch models expect:
    #   T.ToTensor() turns the image into a tensor of shape (3, height, width),
    #     with pixel values scaled from 0-255 down to 0.0-1.0.
    #   .unsqueeze(0) adds one more dimension at the front, making the shape
    #     (1, 3, height, width). The "1" represents a batch of 1 image, because
    #     PyTorch models always expect a batch dimension, even when there's
    #     only a single image.
    img_tensor = T.ToTensor()(img).unsqueeze(0)

    # ------------------------------------------------------------------------
    # STEP 4 - the noise sweep.
    # For every noise level in args.noise_levels, we:
    #   (a) make a noisy copy of the image
    #   (b) run it through the model to get a denoised copy
    #   (c) score both copies against the original with PSNR
    # We keep everything in plain Python lists, in the same order as
    # args.noise_levels, so that item i in every list all describe the same run.
    # ------------------------------------------------------------------------

    # PSNR ("Peak Signal-to-Noise Ratio") is a single number, measured in
    # decibels (dB), that says how close two images are to each other.
    # Higher PSNR = more similar = better. A denoised image should have a
    # HIGHER PSNR than the noisy image it started from, since denoising is
    # supposed to move the image back closer to the clean original.

    noisy_images = []       # will hold one noisy image tensor per noise level
    denoised_images = []    # will hold one denoised image tensor per noise level
    psnr_before_list = []   # will hold one "PSNR of the noisy image" number per noise level
    psnr_after_list = []    # will hold one "PSNR of the denoised image" number per noise level

    for noise in args.noise_levels:
        print(f'Running inference with noise level {noise}...')

        # args.noise_levels are given in the 0-255 range (a common convention
        # for image noise), but our image tensor's pixel values are 0.0-1.0,
        # so we convert the noise level to that same 0.0-1.0 scale.
        noise_level = noise / 255.0

        # torch.randn_like(img_tensor) creates random numbers (Gaussian/"bell
        # curve" noise) in the exact same shape as img_tensor. Multiplying by
        # noise_level controls how strong the noise is, and adding it to
        # img_tensor is what actually makes the image noisy.
        # .clamp(0, 1) then forces every pixel value back into the valid
        # 0.0-1.0 range, in case adding noise pushed some pixels outside it.
        random_noise = torch.randn_like(img_tensor) * noise_level
        noisy = (img_tensor + random_noise).clamp(0, 1)

        # torch.no_grad() tells PyTorch not to bother tracking gradients here.
        # Gradients are only needed during training, and skipping them makes
        # inference (just using the model) faster and use less memory.
        with torch.no_grad():
            if args.model_type == 'conv3':
                # conv3 was trained to predict the NOISE ITSELF (the difference
                # between the noisy and clean image), not the clean image
                # directly. So to get the denoised image, we have to subtract
                # that predicted noise back off of the noisy input ourselves.
                predicted_noise = model(noisy)
                denoised = noisy - predicted_noise
            else:
                # This particular trained UNet was set up to predict the
                # CLEAN image directly, so its output already IS the
                # denoised image - no extra subtraction needed.
                denoised = model(noisy)

        # Compute PSNR by comparing the noisy image to the true/original
        # image (psnr_before), and separately comparing the denoised image
        # to the true/original image (psnr_after). max_I=1. tells the
        # function that our pixel values go up to 1.0 (not 255).
        psnr_before = batch_ave_psnr_torch(img_tensor, noisy, max_I=1.).item()
        psnr_after = batch_ave_psnr_torch(img_tensor, denoised, max_I=1.).item()
        print(f'  PSNR before denoising: {psnr_before:.2f} dB   PSNR after denoising: {psnr_after:.2f} dB')

        # Save everything from this noise level so we can use it later,
        # once the loop has finished running all the noise levels.
        noisy_images.append(noisy)
        denoised_images.append(denoised)
        psnr_before_list.append(psnr_before)
        psnr_after_list.append(psnr_after)

    # ------------------------------------------------------------------------
    # STEP 5 - print a summary table and save it to a CSV file.
    # A CSV file is just a plain text file where each line is one row, and
    # commas separate the columns - it opens directly in Excel/Google Sheets.
    # ------------------------------------------------------------------------
    print('\n=== PSNR summary ===')
    print(f"{'noise':>6} | {'PSNR before':>12} | {'PSNR after':>11} | {'improvement':>11}")
    summary_path = os.path.join(results_dir, 'psnr_sweep_summary.csv')
    with open(summary_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # the first row of a CSV is usually a header naming each column
        writer.writerow(['noise_level', 'psnr_before', 'psnr_after', 'improvement_db'])

        for i, noise in enumerate(args.noise_levels):
            before = psnr_before_list[i]
            after = psnr_after_list[i]
            improvement = after - before  # how many dB better the denoised image is vs. the noisy one

            print(f'{noise:>6} | {before:>12.2f} | {after:>11.2f} | {improvement:>11.2f}')
            writer.writerow([noise, before, after, improvement])

    print(f'Saved summary to {summary_path}')

    # ------------------------------------------------------------------------
    # STEP 6 - build one big combined image (the "grid").
    # It has one row per noise level, and 3 columns:
    #   column 0: the original clean image (same in every row)
    #   column 1: the noisy image at that row's noise level
    #   column 2: the denoised result at that row's noise level
    # ------------------------------------------------------------------------
    if PLOT:
        number_of_rows = len(args.noise_levels)

        # plt.subplots(number_of_rows, 3, ...) creates a grid of empty plots:
        # `axes` ends up being a 2D array of plot "slots", indexed as
        # axes[row_index, column_index], which we fill in below.
        fig, axes = plt.subplots(number_of_rows, 3, figsize=(9, 3 * number_of_rows))

        # One title over the whole figure, saying which model produced these results.
        fig.suptitle(f'Model: {model_label}', fontsize=14)

        for i, noise in enumerate(args.noise_levels):
            # --- column 0: original image ---
            # img_tensor has shape (1, 3, height, width). imshow() wants
            # (height, width, 3) instead, so:
            #   [0]              drops the batch dimension -> (3, height, width)
            #   .permute(1, 2, 0) reorders the dimensions   -> (height, width, 3)
            axes[i, 0].imshow(img_tensor[0].permute(1, 2, 0))
            axes[i, 0].axis('off')  # hides the x/y axis ticks and numbers, since this is a photo, not a chart
            axes[i, 0].set_title('Original' if i == 0 else '')  # only label the very first row, to avoid repeating it

            # --- column 1: noisy image ---
            axes[i, 1].imshow(noisy_images[i][0].permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f'Noisy (level={noise})\nPSNR={psnr_before_list[i]:.2f} dB')

            # --- column 2: denoised image ---
            # .detach() removes this tensor from PyTorch's gradient-tracking
            # system before converting it for display; harmless here since we
            # already wrapped the model call in torch.no_grad() above, but
            # it's a common safety habit when displaying model outputs.
            axes[i, 2].imshow(denoised_images[i][0].detach().permute(1, 2, 0).clamp(0, 1))
            axes[i, 2].axis('off')
            axes[i, 2].set_title(f'Denoised\nPSNR={psnr_after_list[i]:.2f} dB')

        # tight_layout() automatically adds spacing so titles/images don't
        # overlap. The `rect` argument reserves a little extra room at the
        # very top (from 0% to 97% of the figure height) for fig.suptitle().
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Build a file name that includes both the image name and the model
        # name, e.g. "flower1_conv3ln_psnr_sweep_grid.png", so results from
        # different images/models don't overwrite each other.
        image_label = os.path.splitext(os.path.basename(args.image))[0]
        grid_path = os.path.join(grids_dir, f'{image_label}_{model_label}_psnr_sweep_grid.png')

        plt.savefig(grid_path)
        print(f'Saved grid plot to {grid_path}')

        plt.show()  # pops up an interactive window with the figure (only works if a display/screen is available)


# This is a standard Python pattern: the code inside this "if" only runs when
# this file is executed directly (e.g. "python run_inference_sweep.py"), and
# NOT if this file is ever imported from somewhere else.
if __name__ == '__main__':
    main()

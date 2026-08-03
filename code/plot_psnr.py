'''
Plots output-PSNR vs input-PSNR curves for one or more trained denoisers,
in the style of Kadkhodaie et al. figure (same style as plot_split_psnr.py,
which only ever supported splitA/splitB).

The core piece is plot_psnr(): it takes ANY already-loaded model plus a
train/test set, computes its input/output PSNR curve, and draws it onto a
shared pair of train/test panels. main() below calls it once per model in
--models, so any combination of this project's models can be compared on the
same figure:

  'splitA'    -> UNet_flex trained on splitA only (first half of each class)
  'splitB'    -> UNet_flex trained on splitB only (second half of each class)
  'UNet_full' -> UNet_flex trained on all of ImageNet
  'conv3'     -> a pretrained third-party Denoiser (conv3_ln.pt), not trained
                 in this project - useful as an outside comparison point

    python plot_psnr.py
    python plot_psnr.py --models UNet_full conv3
'''

import argparse
import datetime
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

from network import UNet_flex
from noise import add_noise_torch
from quality_metrics_func import batch_ave_psnr_torch


# ==================================================================================================
# SETUP - paths and constants used throughout this file
# ==================================================================================================

# This file lives at texture_model/code/plot_psnr.py, so going up two folders
# gets us to texture_model/, the root of this whole project.
_this_file_path = os.path.abspath(__file__)
_code_folder = os.path.dirname(_this_file_path)
_repo_root = os.path.dirname(_code_folder)

# The conv3 model's class lives in Denoiser_Reconstruction/models/denoiser.py,
# not anywhere under code/, so that folder needs to be added to sys.path
# before "from models.denoiser import Denoiser" below will work.
_denoiser_reconstruction_folder = os.path.join(_repo_root, 'Denoiser_Reconstruction')
sys.path.insert(0, _denoiser_reconstruction_folder)
from models.denoiser import Denoiser

# --------------------------------------------------------------------------------------------------
# Exact folder name (inside models_trained/) for each UNet_flex model this
# script knows about by name. Update these if a model gets retrained further
# and the image/epoch counts in its folder name change.
# --------------------------------------------------------------------------------------------------
UNET_MODEL_DIRS = {
    'splitA':    'UNet_splitA_120229imgs_1000epochs',
    'splitB':    'UNet_splitB_120312imgs_1000epochs',
    'UNet_full': 'UNet_full_240541imgs_1000epochs',
}

# --------------------------------------------------------------------------------------------------
# conv3 is not a UNet_flex model and doesn't live under models_trained/, so it
# needs its own weights path and hand-specified architecture. These numbers
# were found by inspecting the shapes of the saved weights in conv3_ln.pt -
# conv3 has no exp_arguments.pkl to read them from automatically.
# --------------------------------------------------------------------------------------------------
CONV3_WEIGHTS_PATH = os.path.join(_denoiser_reconstruction_folder, 'assets', 'conv3_ln.pt')
CONV3_ARCH_ARGS = SimpleNamespace(
    padding=1,
    num_kernels=64,
    kernel_size=3,
    num_layers=20,
    im_channels=3,
)

# --------------------------------------------------------------------------------------------------
# One line color for each model's curve, so every plot uses the same color
# for the same model and curves are easy to tell apart.
# --------------------------------------------------------------------------------------------------
MODEL_COLORS = {
    'splitA':    'tab:orange',
    'splitB':    'tab:blue',
    'UNet_full': 'tab:green',
    'conv3':     'tab:red',
}


# ==================================================================================================
# MODEL LOADERS
#
# Each of these returns a pair: (model, skip)
#   model -> the loaded network, already in eval mode, already moved to `device`
#   skip  -> a boolean flag saying what the model's output means:
#              skip=True  -> the model predicts the NOISE RESIDUAL (noisy - clean)
#              skip=False -> the model predicts the CLEAN IMAGE directly
#            psnr_curve() (further down this file) uses this flag to know how
#            to turn the model's raw output into a denoised image.
# ==================================================================================================

def load_unet_model(model_dir, device):
    '''
    Loads a UNet_flex denoiser from a models_trained/ subfolder containing
    model.pt and exp_arguments.pkl. Works for any UNet_flex model trained via
    code/main.py - splitA, splitB, and the fully trained UNet all load the
    same way, only model_dir differs.
    '''

    # ----------------------------------------------------------------------------
    # STEP 1 - load the architecture settings that were saved at training time.
    # exp_arguments.pkl is a plain Python dictionary, saved to disk with pickle.
    # ----------------------------------------------------------------------------
    exp_arguments_path = os.path.join(model_dir, 'exp_arguments.pkl')
    with open(exp_arguments_path, 'rb') as pkl_handle:
        saved_arguments_dict = pickle.load(pkl_handle)
    unet_args = argparse.Namespace(**saved_arguments_dict)

    # ----------------------------------------------------------------------------
    # STEP 2 - build an empty (randomly-initialized) network in the right shape.
    # ----------------------------------------------------------------------------
    model = UNet_flex(unet_args)

    # ----------------------------------------------------------------------------
    # STEP 3 - load the trained weights from disk.
    # weights_only=False because these checkpoints were saved with an older
    # PyTorch pickle format.
    # ----------------------------------------------------------------------------
    weights_path = os.path.join(model_dir, 'model.pt')
    saved_state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

    # ----------------------------------------------------------------------------
    # STEP 4 - strip the "module." prefix that training added to every weight
    # name, because this model was trained with nn.DataParallel across
    # multiple GPUs. The freshly-built network above doesn't have that
    # prefix in its layer names, so the names would otherwise fail to match.
    # ----------------------------------------------------------------------------
    fixed_state_dict = {}
    for saved_key, saved_value in saved_state_dict.items():
        fixed_key = saved_key.removeprefix('module.')
        fixed_state_dict[fixed_key] = saved_value

    # ----------------------------------------------------------------------------
    # STEP 5 - copy the trained weights into the empty network, then switch to
    # evaluation mode (turns off training-only behavior like dropout) and move
    # the model onto the requested device (CPU, MPS, or CUDA).
    # ----------------------------------------------------------------------------
    model.load_state_dict(fixed_state_dict)
    model.eval()
    model.to(device)

    # ----------------------------------------------------------------------------
    # STEP 6 - read the skip/residual flag straight from the saved training
    # arguments, so we don't have to guess or hard-code it per model.
    # ----------------------------------------------------------------------------
    skip = unet_args.skip

    return model, skip


def load_conv3_model(device):
    '''
    Loads Ling-Qi's pretrained conv3 denoiser (not trained in this project).
    conv3 was trained to predict the noise itself, not the clean image, so
    skip is always True for this model.
    '''

    # STEP 1 - build an empty network using the hand-specified architecture above.
    model = Denoiser(CONV3_ARCH_ARGS)

    # STEP 2 - load conv3's saved weights from disk.
    saved_state_dict = torch.load(CONV3_WEIGHTS_PATH, map_location='cpu', weights_only=False)

    # STEP 3 - copy the trained weights in, switch to eval mode, move to device.
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.to(device)

    # STEP 4 - conv3 always predicts the noise residual, not the clean image.
    skip = True

    return model, skip


def load_model_by_name(name, results_dir, device):
    '''
    name:        one of 'splitA', 'splitB', 'UNet_full', or 'conv3'
    results_dir: folder containing models_trained/ (only used for the UNet models)
    returns (model, skip) - see the "MODEL LOADERS" section comment above
    '''
    if name == 'conv3':
        return load_conv3_model(device)

    # Every other name in this script refers to a UNet_flex model, so look up
    # its folder name and load it the same way.
    model_folder_name = UNET_MODEL_DIRS[name]
    model_dir = os.path.join(results_dir, 'models_trained', model_folder_name)
    return load_unet_model(model_dir, device)


# ==================================================================================================
# DATASET LOADERS
#
# Each of these returns a pair: (train_set, test_set)
# Both are single tensors of CLEAN (not noisy) images, ready to be handed to
# psnr_curve() below, which will add noise to them itself.
# ==================================================================================================

def load_split_data(split, data_root_path, max_images=None):
    '''
    split:          which half to load - 'splitA' or 'splitB'
    data_root_path: folder containing the imagenet/ subfolder with the preprocessed .pt files
    max_images:     optional cap on the number of images returned per set, for quick smoke tests;
                    default None uses every image
    returns (train_set, test_set): flat tensors of clean images for this split
    '''
    data_path = os.path.join(data_root_path, 'imagenet')

    # ----------------------------------------------------------------------------
    # STEP 1 - load the full training set. It comes back as a Python list of
    # tensors, one tensor per ImageNet class (e.g. one tensor of "cat" images,
    # one tensor of "dog" images, and so on).
    # ----------------------------------------------------------------------------
    train_list_path = os.path.join(data_path, 'train_80x80_color_list.pt')
    full_train_list = torch.load(train_list_path, weights_only=True)

    # ----------------------------------------------------------------------------
    # STEP 2 - for every class, keep only this split's half of the images.
    # splitA gets the first half of each class, splitB gets the second half.
    # ----------------------------------------------------------------------------
    train_halves = []
    for class_images in full_train_list:
        n_images_in_class = len(class_images)
        halfway_point = n_images_in_class // 2

        if split == 'splitA':
            this_class_half = class_images[:halfway_point]
        else:
            this_class_half = class_images[halfway_point:]

        train_halves.append(this_class_half)

    # STEP 3 - glue all the per-class halves together into one big tensor.
    train_set = torch.cat(train_halves)

    # ----------------------------------------------------------------------------
    # STEP 4 - load the test set. Unlike the training set, the test set is NOT
    # split - both splitA and splitB models are evaluated against the exact
    # same held-out test images, so their scores are directly comparable.
    # ----------------------------------------------------------------------------
    test_list_path = os.path.join(data_path, 'test_80x80_color_list.pt')
    full_test_list = torch.load(test_list_path, weights_only=True)
    test_set = torch.cat(full_test_list)

    # STEP 5 - optionally shrink both sets, for quick smoke-test runs.
    if max_images is not None:
        train_set = train_set[:max_images]
        test_set = test_set[:max_images]

    return train_set, test_set


def load_full_data(data_root_path, max_images=None):
    '''
    data_root_path: folder containing the imagenet/ subfolder with the preprocessed .pt files
    max_images:     optional cap on the number of images returned per set, for quick smoke tests
    returns (train_set, test_set): the full (non-split) ImageNet train/test sets - what
        UNet_full was trained on, and what conv3 gets evaluated against for comparison
        even though conv3 was never trained on it.
    '''
    data_path = os.path.join(data_root_path, 'imagenet')

    # STEP 1 - load and combine every class's training images (no splitting this time).
    train_list_path = os.path.join(data_path, 'train_80x80_color_list.pt')
    full_train_list = torch.load(train_list_path, weights_only=True)
    train_set = torch.cat(full_train_list)

    # STEP 2 - load and combine the held-out test images.
    test_list_path = os.path.join(data_path, 'test_80x80_color_list.pt')
    full_test_list = torch.load(test_list_path, weights_only=True)
    test_set = torch.cat(full_test_list)

    # STEP 3 - optionally shrink both sets, for quick smoke-test runs.
    if max_images is not None:
        train_set = train_set[:max_images]
        test_set = test_set[:max_images]

    return train_set, test_set


def load_data_for_model(name, data_root_path, max_images=None):
    '''
    name: one of 'splitA', 'splitB', 'UNet_full', or 'conv3'
    returns (train_set, test_set) appropriate for that model: the split half
        it was actually trained on for splitA/splitB, or the full dataset for
        UNet_full and conv3 (conv3 wasn't trained on this dataset at all, but
        evaluating it on the full set still gives a fair comparison point).
    '''
    if name in ('splitA', 'splitB'):
        return load_split_data(name, data_root_path, max_images)
    else:
        return load_full_data(data_root_path, max_images)


# ==================================================================================================
# CORE MEASUREMENT - turns a model + a dataset into a PSNR curve
# ==================================================================================================

def psnr_curve(model, dataset, sigmas, device, batch_size, skip):
    '''
    model:      the trained denoiser
    dataset:    a single tensor of clean images to test against (e.g. one split's train or test set)
    sigmas:     1D tensor of noise levels in 0-255 units (same convention as args.noise_level_range)
    device:     torch device to run inference on (e.g. 'cuda')
    batch_size: number of images per forward pass through the model
    skip:       whether the model was trained with the skip/residual convention -
                        if True, model output is the predicted noise residual (noisy - clean);
                        if False, model output is the clean image directly
    returns (input_psnr, output_psnr): arrays of len(sigmas), averaged over the whole dataset
    '''
    n_images = dataset.shape[0]

    # One output value per noise level (sigma) we're going to test.
    # These start out as all-zero arrays and get filled in as we go.
    input_psnr = np.zeros(len(sigmas))
    output_psnr = np.zeros(len(sigmas))

    # ----------------------------------------------------------------------------
    # Outer loop: one pass through this whole loop body per noise level.
    # ----------------------------------------------------------------------------
    for sigma_index in range(len(sigmas)):
        sigma = sigmas[sigma_index]

        # Per-batch PSNR values at this noise level get collected here, then
        # averaged together after the inner loop finishes.
        input_psnr_values_this_sigma = []
        output_psnr_values_this_sigma = []

        # ------------------------------------------------------------------------
        # Inner loop: one pass through this loop body per batch of images.
        # We can't run the whole dataset through the model at once (it might
        # not fit in memory), so we process it in smaller chunks of size
        # batch_size instead.
        # ------------------------------------------------------------------------
        for batch_start in range(0, n_images, batch_size):
            batch_end = min(batch_start + batch_size, n_images)
            clean_images = dataset[batch_start:batch_end]
            clean_images = clean_images.to(device)

            # Add Gaussian noise of the current sigma to this batch of clean
            # images. add_noise_torch is defined in noise.py.
            noisy_images, _, _ = add_noise_torch(all_patches=clean_images, noise_level=sigma.item())

            # Run the denoiser. torch.no_grad() turns off gradient tracking,
            # since we're only using the model here, not training it.
            with torch.no_grad():
                model_output = model(noisy_images)

            # Turn the model's raw output into an actual denoised image.
            #   skip=True  -> model_output IS the noise, so subtract it off
            #   skip=False -> model_output already IS the denoised image
            if skip:
                denoised_images = noisy_images - model_output
            else:
                denoised_images = model_output

            # PSNR of the noisy image vs. the clean original - this is the
            # "input" (x-axis) quality: how bad was the image before denoising?
            batch_input_psnr = batch_ave_psnr_torch(clean_images, noisy_images, 1.)
            batch_input_psnr = batch_input_psnr.item()
            input_psnr_values_this_sigma.append(batch_input_psnr)

            # PSNR of the denoised image vs. the clean original - this is the
            # "output" (y-axis) quality: how good was the image after denoising?
            batch_output_psnr = batch_ave_psnr_torch(clean_images, denoised_images, 1.)
            batch_output_psnr = batch_output_psnr.item()
            output_psnr_values_this_sigma.append(batch_output_psnr)

        # Average across all batches at this noise level, and store the result
        # in this sigma's slot in the output arrays.
        input_psnr[sigma_index] = np.mean(input_psnr_values_this_sigma)
        output_psnr[sigma_index] = np.mean(output_psnr_values_this_sigma)

        print(
            f'  sigma={sigma.item():.2f} -> '
            f'input PSNR={input_psnr[sigma_index]:.2f}, '
            f'output PSNR={output_psnr[sigma_index]:.2f}'
        )

    return input_psnr, output_psnr


# ==================================================================================================
# PLOTTING - one model's curve at a time
# ==================================================================================================

def plot_psnr(model_label, model, skip, train_set, test_set, sigmas, device, batch_size, color, train_ax, test_ax):
    '''
    Computes input/output PSNR curves for ONE model (any model - it doesn't
    need to be a split model, or even a UNet) on its train and test sets, and
    draws its curve onto the given train/test axes, the same way
    plot_split_psnr.py used to do by hand for just splitA and splitB. Call
    this once per model to build up a multi-model comparison figure.

    model_label: name used for this curve's legend entry and print statements
    model:       the loaded model, in eval mode, on `device`
    skip:        whether `model` outputs a noise residual (True) or the clean image directly (False)
    train_set:   tensor of clean images this model was trained on (used for the "Train" panel)
    test_set:    tensor of held-out clean images (used for the "Test" panel)
    sigmas:      1D tensor of noise levels to evaluate at
    device:      torch device to run inference on
    batch_size:  number of images per forward pass
    color:       matplotlib color for this model's curve
    train_ax:    axes to draw the train-panel curve on
    test_ax:     axes to draw the test-panel curve on

    returns (train_input_psnr, train_output_psnr, test_input_psnr, test_output_psnr)
    '''

    # ----------------------------------------------------------------------------
    # STEP 1 - compute this model's PSNR curve on its training set.
    # ----------------------------------------------------------------------------
    n_train_images = train_set.shape[0]
    print(f'[{model_label}] computing train curve ({n_train_images} images)...')
    train_input_psnr, train_output_psnr = psnr_curve(model, train_set, sigmas, device, batch_size, skip)

    # ----------------------------------------------------------------------------
    # STEP 2 - compute this model's PSNR curve on its held-out test set.
    # ----------------------------------------------------------------------------
    n_test_images = test_set.shape[0]
    print(f'[{model_label}] computing test curve ({n_test_images} images)...')
    test_input_psnr, test_output_psnr = psnr_curve(model, test_set, sigmas, device, batch_size, skip)

    # ----------------------------------------------------------------------------
    # STEP 3 - draw this model's curve onto both panels, using the same color
    # and label in each, so it's clear both curves came from the same model.
    # ----------------------------------------------------------------------------
    train_ax.plot(train_input_psnr, train_output_psnr, '-o', color=color, label=model_label)
    test_ax.plot(test_input_psnr, test_output_psnr, '-o', color=color, label=model_label)

    return train_input_psnr, train_output_psnr, test_input_psnr, test_output_psnr


def report_diff(name, curve_a, curve_b):
    '''
    Prints how far apart two output-PSNR curves are. Curves from different
    models/splits can look nearly identical when plotted, since the
    differences between them are small relative to the plot's axis range, so
    this prints the actual numbers instead of relying on the picture alone.
    '''
    diffs = curve_a - curve_b
    rounded_diffs = np.round(diffs, 4).tolist()
    abs_diffs = np.abs(diffs)
    max_abs_diff = np.max(abs_diffs)
    mean_abs_diff = np.mean(abs_diffs)
    identically_zero = np.all(diffs == 0)

    print(f'{name}: per-sigma diff = {rounded_diffs}')
    print(f'{name}: max abs diff = {max_abs_diff:.4f} dB')
    print(f'{name}: mean abs diff = {mean_abs_diff:.4f} dB')
    print(f'{name}: identically zero? {identically_zero}')


# ==================================================================================================
# MAIN - ties everything above together into one runnable script
# ==================================================================================================

def main():
    '''
    Loads every model in --models, computes each one's train/test PSNR curves
    across noise levels via plot_psnr(), and saves a comparison figure to
    --output.
    '''

    # ----------------------------------------------------------------------------
    # STEP 1 - read command-line options.
    # Example of running this script:
    #   python plot_psnr.py --models UNet_full conv3
    # ----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models', nargs='+', default=['splitA', 'splitB', 'UNet_full', 'conv3'],
        choices=['splitA', 'splitB', 'UNet_full', 'conv3'],
        help='which models to plot curves for',
    )
    parser.add_argument(
        '--results_dir', default=_repo_root,
        help='folder containing models_trained/, which holds one subfolder per model, each with model.pt and exp_arguments.pkl',
    )
    parser.add_argument('--data_root_path', default='/mnt/home/cdyer/ceph/images/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_images', type=int, default=None)
    # how many noise levels to evaluate at - this is how many points end up on each curve
    parser.add_argument('--n_sigma', type=int, default=10)
    # sigma range in 0-255 units, matching args.noise_level_range's default [1, 765] in main.py
    parser.add_argument('--sigma_min', type=float, default=1)
    parser.add_argument('--sigma_max', type=float, default=765)
    parser.add_argument(
        '--output', default='psnr_curves.png',
        help='output filename (just the name - it is always saved into --output_dir)',
    )
    parser.add_argument(
        '--output_dir', default=os.path.join(_repo_root, 'split_psnr_plots'),
        help='folder where every run of this script saves its output plot',
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------------
    # STEP 2 - work out where to save the output plot.
    # A timestamp gets appended to the filename so each run saves to its own
    # new file, instead of overwriting the previous run's result.
    # ----------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    output_filename = os.path.basename(args.output)
    output_root, output_ext = os.path.splitext(output_filename)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_filename = f'{output_root}_{timestamp}{output_ext}'
    args.output = os.path.join(args.output_dir, timestamped_filename)

    # ----------------------------------------------------------------------------
    # STEP 3 - pick which device to run on: GPU if available, otherwise CPU.
    # ----------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # ----------------------------------------------------------------------------
    # STEP 4 - build the list of noise levels every model gets evaluated at.
    # logspace spaces the values evenly on a LOG scale rather than a linear
    # one, so we get good coverage at both low and high noise levels.
    # ----------------------------------------------------------------------------
    sigma_min_log = np.log10(args.sigma_min)
    sigma_max_log = np.log10(args.sigma_max)
    sigmas = torch.logspace(sigma_min_log, sigma_max_log, args.n_sigma)

    # ----------------------------------------------------------------------------
    # STEP 5 - set up the figure: one row, two panels side by side.
    # train_ax is the left panel (Train), test_ax is the right panel (Test).
    # sharey=True means both panels use the same y-axis scale, so curves are
    # directly comparable just by looking at height.
    # ----------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    train_ax = fig.add_subplot(1, 2, 1)
    test_ax = fig.add_subplot(1, 2, 2, sharey=train_ax)

    # This dictionary collects every model's output-PSNR curves as we go, so
    # that STEP 7 below can compare train vs test for each model afterward.
    output_curves_by_model = {}

    # ----------------------------------------------------------------------------
    # STEP 6 - the main loop: for every requested model, load it, load its
    # data, compute its PSNR curve, and add it to the plot.
    # ----------------------------------------------------------------------------
    for model_label in args.models:
        print(f'--- {model_label} ---')

        model, skip = load_model_by_name(model_label, args.results_dir, device)
        train_set, test_set = load_data_for_model(model_label, args.data_root_path, max_images=args.n_images)
        color = MODEL_COLORS[model_label]

        _, train_output_psnr, _, test_output_psnr = plot_psnr(
            model_label, model, skip,
            train_set, test_set,
            sigmas, device, args.batch_size,
            color, train_ax, test_ax,
        )

        output_curves_by_model[model_label] = (train_output_psnr, test_output_psnr)

    # ----------------------------------------------------------------------------
    # STEP 7 - print the train-vs-test gap for each model explicitly. The
    # curves can look nearly identical when just looking at the plot, since
    # the differences are small relative to the plot's axis range.
    # ----------------------------------------------------------------------------
    for model_label, (train_output_psnr, test_output_psnr) in output_curves_by_model.items():
        report_diff(f'{model_label} train vs test', train_output_psnr, test_output_psnr)

    # ----------------------------------------------------------------------------
    # STEP 8 - finish each panel: add a dashed "identity line" (output PSNR ==
    # input PSNR) as a reference, then label the axes and add a legend.
    # ----------------------------------------------------------------------------
    train_lims = train_ax.get_xlim()
    train_ax.plot(train_lims, train_lims, '--', color='gray', label='identity')
    train_ax.set_xlabel('Input PSNR')
    train_ax.set_title('Train')
    train_ax.legend()

    test_lims = test_ax.get_xlim()
    test_ax.plot(test_lims, test_lims, '--', color='gray', label='identity')
    test_ax.set_xlabel('Input PSNR')
    test_ax.set_title('Test')
    test_ax.legend()

    train_ax.set_ylabel('Output PSNR')

    # ----------------------------------------------------------------------------
    # STEP 9 - save the finished figure to disk.
    # ----------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'saved to {args.output}')


# This is a standard Python pattern: the code inside this "if" only runs when
# this file is executed directly (e.g. "python plot_psnr.py"), and NOT if
# this file is ever imported from somewhere else.
if __name__ == '__main__':
    main()

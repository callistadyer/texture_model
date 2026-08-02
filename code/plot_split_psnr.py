'''
Plots output-PSNR vs input-PSNR curves for the splitA and splitB denoisers,
in the style of Kadkhodaie et al. figure 

Unlike that figure, this only has one training-set size per curve (splitA and
splitB each saw ~half of ImageNet, split within each class - see main.py's
imagenet_splitA/imagenet_splitB branches) - so this shows two curves, not a
memorization-to-generalization sweep across N.

    python plot_split_psnr.py
'''

import argparse
import datetime
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from network import UNet_flex
from noise import add_noise_torch
from quality_metrics_func import batch_ave_psnr_torch

# Function for loading the split halves models
def load_split_model(split, results_dir, device):
    '''
    split:       which half to load - 'splitA' or 'splitB'
    results_dir: folder containing models_trained/, which holds one subfolder per model,
                 each with a model.pt and exp_arguments.pkl
    device:      torch device to load the model onto (e.g. 'cuda')
    returns (model, unet_args): the loaded model in eval mode, and its architecture config
    '''
    # Exact folder name (inside models_trained/) for each split - update these two lines
    # if the models get retrained further and the image/epoch counts in the name change
    split_dir_names = {
        'splitA': 'UNet_splitA_120229imgs_1000epochs',
        'splitB': 'UNet_splitB_120312imgs_1000epochs',
    }
    split_dir = os.path.join(results_dir, 'models_trained', split_dir_names[split])
    with open(os.path.join(split_dir, 'exp_arguments.pkl'), 'rb') as f:
        unet_args = argparse.Namespace(**pickle.load(f))

    # Construct an empty network with the saved architecture
    model = UNet_flex(unet_args)
    # Load the trained weights
    state_dict = torch.load(os.path.join(split_dir, 'model.pt'), map_location='cpu', weights_only=False)
    # Checkpoints trained with nn.DataParallel (multi-GPU) prefix every key with "module." Get rid of that 
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    # Put the trained weights into the empty model
    model.load_state_dict(state_dict)
    # Switch off training only behavior 
    model.eval().to(device)
    return model, unet_args

# Function for loading the data that was used to train the split halves models
def load_split_data(split, data_root_path, max_images=None):
    '''
    split:          which half to reconstruct - 'splitA' or 'splitB'
    data_root_path: folder containing the imagenet/ subfolder with the preprocessed .pt files
    max_images:     optional cap on the number of images returned per set, for quick smoke tests;
                    default None uses every image
    returns (train_set, test_set): flat tensors of clean images for this split
    '''
    # Where the preprocessed .pt files live
    data_path = os.path.join(data_root_path, 'imagenet')
    # Load the full training set: a list of tensors, one tensor per ImageNet class
    full_train = torch.load(os.path.join(data_path, 'train_80x80_color_list.pt'), weights_only=True)

    train_set = []
    for d in full_train:
        n_images = len(d)
        # Divide to get the split point for this class
        half = n_images // 2
        # splitA gets the first half of each class, splitB gets the second half
        if split == 'splitA':
            class_half = d[:half]
        else:
            class_half = d[half:]
        train_set.append(class_half)
    train_set = torch.cat(train_set)

    # Load the held-out test set
    test_set = torch.load(os.path.join(data_path, 'test_80x80_color_list.pt'), weights_only=True)
    # Test set is NOT split. Both models are evaluated against the same full test set
    test_set = torch.cat(test_set)

    # Can set max_images to a small number to test this on a small num of images
    if max_images is not None:
        train_set = train_set[:max_images]
        test_set = test_set[:max_images]

    # Hand back both datasets
    return train_set, test_set


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
    # shuffle=False since order doesn't matter for computing an average
    # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    n_images = dataset.shape[0]

    # One output value per noise level (sigma) we're going to test
    input_psnr  = np.zeros(len(sigmas))
    output_psnr = np.zeros(len(sigmas))

    # Loop over the dataset for every noise level, so we get one (input, output) point per sigma
    for s_idx in range(len(sigmas)):
        sigma = sigmas[s_idx]
        # Per-batch PSNR values at this noise level, collected here and averaged after the inner loop
        input_psnr_values  = []
        output_psnr_values = []
        # for batch in loader: 
        for batch_start in range(0, n_images, batch_size):
            batch_end = min(batch_start + batch_size, n_images)
            batch = dataset[batch_start:batch_end]

            # move this batch of clean images onto the GPU or correct device
            cleanImgs = batch.to(device)

            # Add Gaussian noise of the current sigma to the clean images (add_noise_torch is defined in noise.py)
            noisyImgs, _, _ = add_noise_torch(all_patches=cleanImgs, noise_level=sigma.item())

            # Run the denoiser. no_grad because this is evaluation onl
            with torch.no_grad():
                output = model(noisyImgs)
                
            # UNet_flex trained with skip=True predicts the noise residual (noisy - clean), so recover
            # the denoised image by subtracting that from the noisy input; skip=False models output
            # the clean image directly, so no subtraction needed
            if skip:
                denoised = noisyImgs - output
            else:
                denoised = output

            # PSNR of the noisy image vs. the clean original - this is the "input" (x-axis) quality
            # (batch_ave_psnr_torch is defined in quality_metrics_func.py)
            batch_input_psnr = batch_ave_psnr_torch(cleanImgs, noisyImgs, 1.)
            batch_input_psnr = batch_input_psnr.item()
            input_psnr_values.append(batch_input_psnr)

            # PSNR of the denoised image vs. the clean original - this is the "output" (y-axis) quality
            batch_output_psnr = batch_ave_psnr_torch(cleanImgs, denoised, 1.)
            batch_output_psnr = batch_output_psnr.item()
            output_psnr_values.append(batch_output_psnr)

        # Average across all batches
        input_psnr[s_idx]  = np.mean(input_psnr_values)
        output_psnr[s_idx] = np.mean(output_psnr_values)
        print(f'  sigma={sigma.item():.2f} -> input PSNR={input_psnr[s_idx]:.2f}, output PSNR={output_psnr[s_idx]:.2f}')

    return input_psnr, output_psnr


def main():
    '''
    Loads both split models, computes their train/test PSNR curves across noise levels,
    and saves a comparison figure to --output.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         help='folder containing models_trained/, which holds one subfolder per model, each with model.pt and exp_arguments.pkl')
    parser.add_argument('--data_root_path', default='/mnt/home/cdyer/ceph/images/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_images', type=int, default=None)
    parser.add_argument('--n_sigma', type=int, default=10)     # how many noise levels to evaluate at - this is how many points end up on each curve
    parser.add_argument('--sigma_min', type=float, default=1)  # sigma range in 0-255 units, matching args.noise_level_range's default [1, 765] in main.py
    parser.add_argument('--sigma_max', type=float, default=765)
    parser.add_argument('--skip', action='store_true', help='set if the model was trained with --skip (residual output)')
    parser.add_argument('--output', default='split_psnr_curves.png',
                         help='output filename (just the name - it is always saved into --output_dir)')
    parser.add_argument('--output_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'split_psnr_plots'),
                         help='folder where every run of this script saves its output plot')
    args = parser.parse_args()

    # Make sure the designated output folder exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Append a timestamp to the output filename so each run saves to its own new file
    # instead of overwriting the previous run's result, then place it in --output_dir
    # (only the filename part of --output is used, so a full path passed in still works)
    output_filename      = os.path.basename(args.output)
    output_root, output_ext = os.path.splitext(output_filename)
    timestamp            = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_filename = f'{output_root}_{timestamp}{output_ext}'
    args.output          = os.path.join(args.output_dir, timestamped_filename)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # The noise levels both models get evaluated at
    sigmas = torch.logspace(np.log10(args.sigma_min), np.log10(args.sigma_max), args.n_sigma)

    # Colors for the two curves
    colors = {'splitA': 'tab:orange', 'splitB': 'tab:blue'}

    # Load splitA model
    splitA_model, _ = load_split_model('splitA', args.results_dir, device)
    # Load splitA data
    splitA_train_set, splitA_test_set = load_split_data('splitA', args.data_root_path, max_images=args.n_images)
    # Load splitB model
    splitB_model, _ = load_split_model('splitB', args.results_dir, device)
    # Load splitB data
    splitB_train_set, splitB_test_set = load_split_data('splitB', args.data_root_path, max_images=args.n_images)

    # Each call here is a full forward pass over the whole dataset, once per sigma
    # this is the expensive part of the script (see psnr_curve)
    print(f'[splitA] computing train curve ({splitA_train_set.shape[0]} images)...')
    splitA_train_input_psnr, splitA_train_output_psnr = psnr_curve(splitA_model, splitA_train_set, sigmas, device, args.batch_size, args.skip)
    print(f'[splitA] computing test curve ({splitA_test_set.shape[0]} images)...')
    splitA_test_input_psnr, splitA_test_output_psnr = psnr_curve(splitA_model, splitA_test_set, sigmas, device, args.batch_size, args.skip)

    print(f'[splitB] computing train curve ({splitB_train_set.shape[0]} images)...')
    splitB_train_input_psnr, splitB_train_output_psnr = psnr_curve(splitB_model, splitB_train_set, sigmas, device, args.batch_size, args.skip)
    print(f'[splitB] computing test curve ({splitB_test_set.shape[0]} images)...')
    splitB_test_input_psnr, splitB_test_output_psnr = psnr_curve(splitB_model, splitB_test_set, sigmas, device, args.batch_size, args.skip)

    # the 4 output-PSNR curves look nearly identical when plotted, since
    # differences are small relative to the plot's axis range. Check to see if exactly equal
    def report_diff(name, curve_a, curve_b):
        diffs         = curve_a - curve_b
        rounded_diffs = np.round(diffs, 4).tolist()
        abs_diffs     = np.abs(diffs)
        max_abs_diff  = np.max(abs_diffs)
        mean_abs_diff = np.mean(abs_diffs)
        identically_zero = np.all(diffs == 0)

        print(f'{name}: per-sigma diff = {rounded_diffs}')
        print(f'{name}: max abs diff = {max_abs_diff:.4f} dB')
        print(f'{name}: mean abs diff = {mean_abs_diff:.4f} dB')
        print(f'{name}: identically zero? {identically_zero}')

    report_diff('splitA train vs test', splitA_train_output_psnr, splitA_test_output_psnr)
    report_diff('splitB train vs test', splitB_train_output_psnr, splitB_test_output_psnr)
    report_diff('train splitA vs splitB', splitA_train_output_psnr, splitB_train_output_psnr)
    report_diff('test splitA vs splitB', splitA_test_output_psnr, splitB_test_output_psnr)

    # Plot: one figure with two side-by-side panels, one axes object for each
    fig = plt.figure(figsize=(12, 5))
    train_ax = fig.add_subplot(1, 2, 1)
    test_ax  = fig.add_subplot(1, 2, 2, sharey=train_ax)

    # Train panel
    train_ax.plot(splitA_train_input_psnr, splitA_train_output_psnr, '-o', color=colors['splitA'], label='splitA')
    train_ax.plot(splitB_train_input_psnr, splitB_train_output_psnr, '-o', color=colors['splitB'], label='splitB')
    
    # Dashed identity line (output PSNR == input PSNR) as a reference
    train_lims = train_ax.get_xlim()
    train_ax.plot(train_lims, train_lims, '--', color='gray', label='identity')
    train_ax.set_xlabel('Input PSNR')
    train_ax.set_title('Train')
    train_ax.legend()

    # Test panel
    test_ax.plot(splitA_test_input_psnr, splitA_test_output_psnr, '-o', color=colors['splitA'], label='splitA')
    test_ax.plot(splitB_test_input_psnr, splitB_test_output_psnr, '-o', color=colors['splitB'], label='splitB')
    test_lims = test_ax.get_xlim()
    test_ax.plot(test_lims, test_lims, '--', color='gray', label='identity')
    test_ax.set_xlabel('Input PSNR')
    test_ax.set_title('Test')
    test_ax.legend()

    train_ax.set_ylabel('Output PSNR')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'saved to {args.output}')


if __name__ == '__main__':
    main()

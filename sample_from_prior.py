"""
sample_from_prior.py

Starts from a white noise image and iteratively denoises it using the
trained UNet_flex model (model.pt), producing a sample from the model's
learned image prior.

This uses the same stochastic gradient ascent loop as the reconstruction
notebook. The only difference is there is no measurement constraint, so
the image is free to settle wherever the model's prior pulls it.

Usage:
    python sample_from_prior.py --model UNet
    python sample_from_prior.py --model UNet_45500
    python sample_from_prior.py --model conv3
    python sample_from_prior.py --model UNet --size 128 --output my_sample.png
    python sample_from_prior.py --model conv3 --seed 42 --sig_end 0.005

Arguments:
    --model     Which denoiser to use (default: UNet)
                  "UNet"       = UNet_flex trained on 240k ImageNet images (model.pt)
                  "UNet_45500" = UNet_flex trained on 45.5k ImageNet images (model_45500.pt)
                  "conv3"      = pretrained conv3_ln Denoiser (Denoiser_Reconstruction/assets/conv3_ln.pt)
    --size      Side length of the output image in pixels (default: 80,
                which matches the 80x80 patches the model was trained on)
    --output    Path to save the final denoised image (default: sample.png)
    --seed      Random seed for reproducibility (default: no seed)
    --h_init    Initial step size for the gradient updates (default: 0.01)
    --beta      Controls how much noise is re-injected each step (default: 0.01)
    --sig_end   Stopping threshold: loop ends when estimated noise magnitude
                drops below this value (default: 0.005)
    --stride    Print progress every this many iterations (default: 50)
"""

import argparse
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings

# Base results folder. Outputs are routed into subfolders by model:
#   UNet       -> sampleFromPriorResults/UNet/
#   UNet_45500 -> sampleFromPriorResults/UNet_45500/
#   conv3      -> sampleFromPriorResults/conv3/
RESULTS_BASE = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/sampleFromPriorResults'


# Parse arguments
parser = argparse.ArgumentParser(description='Sample from denoiser prior')
parser.add_argument('--model',  type=str,   default='UNet',
                    help='"UNet" = UNet_flex 240k images; '
                         '"UNet_45500" = UNet_flex 45.5k images; '
                         '"conv3" = pretrained Denoiser (conv3_ln.pt)')
parser.add_argument('--size',   type=int,   default=80,
                    help='Image side length in pixels (default: 80)')
parser.add_argument('--output', type=str,   default=None,
                    help='Output file path. Defaults to sampleFromPriorResults/UNet/ or conv3/ based on --model')
parser.add_argument('--seed',   type=int,   default=None,
                    help='Random seed (default: none)')
parser.add_argument('--h_init', type=float, default=0.01,
                    help='Initial gradient step size (default: 0.01)')
parser.add_argument('--beta',   type=float, default=0.01,
                    help='Noise re-injection strength (default: 0.01)')
parser.add_argument('--sig_end',type=float, default=0.005,
                    help='Stop when sigma drops below this (default: 0.005)')
parser.add_argument('--stride', type=int,   default=50,
                    help='Print progress every N iterations (default: 50)')
args = parser.parse_args()

# If no --output was given, build the filename automatically.
# Format: sample_seed{N}_run{M}_{model}.png
#   seed label : "seed42" if --seed 42 was passed, "seedrand" if no seed given
#   model label: "UNet" or "conv3" -- matches the subfolder name
#   run number : scan existing files in the subfolder for this seed+model combo
#                and use the next number (starts at 1 if none found yet)
if args.output is None:
    seed_label  = f'seed{args.seed}' if args.seed is not None else 'seedrand'
    output_dir  = Path(RESULTS_BASE) / args.model

    # Look for existing files like sample_seed42_run3_UNet.png in the subfolder
    run_pattern   = re.compile(rf'^sample_{seed_label}_run(\d+)_{args.model}\.png$')
    existing_runs = []
    if output_dir.exists():
        for f in output_dir.iterdir():
            m = run_pattern.match(f.name)
            if m:
                existing_runs.append(int(m.group(1)))

    # Next run is one past the highest found, or 1 if this seed has never been run
    run_num      = max(existing_runs) + 1 if existing_runs else 1
    filename     = f'sample_{seed_label}_run{run_num}_{args.model}.png'
    args.output  = str(output_dir / filename)

# Make sure the output folder exists (creates UNet/ or conv3/ if needed)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

# Device selection: prefer Apple MPS, then CUDA, then CPU
# (same pattern used in recon_visualize_dichromat.ipynb)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# STEP 1-4: Load the model selected by --model.
#
# "UNet" loads trained UNet_flex (model.pt).
#   - UNet_flex outputs the clean image directly (trained with skip=False),
#     so a ResidualWrapper converts it to a noise residual for the sampler.
#
# "conv3" loads the pretrained Denoiser (conv3_ln.pt) from Denoiser_Reconstruction.
#   - The Denoiser already outputs the noise residual directly, so no wrapper
#     is needed.

if args.model == 'UNet':
    # UNet_flex trained on 240k ImageNet images
    # copied from recon_visualize_dichromat.ipynb

    code_path = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/code'
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    from network import UNet_flex

    args_path    = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/exp_arguments.pkl'
    weights_path = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/model.pt'

    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)
    unet_args = SimpleNamespace(**args_dict)
    unet_base = UNet_flex(unet_args)
    state = torch.load(weights_path, map_location='cpu', weights_only=False)
    unet_base.load_state_dict(state)
    unet_base.eval()

    class ResidualWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, y):
            return y - self.m(y)

    model = ResidualWrapper(unet_base).to(device)
    print(f'Loaded UNet model (UNet_flex) from {weights_path}')

elif args.model == 'UNet_45500':
    # UNet_flex trained on 45.5k ImageNet images
    # copied from recon_visualize_dichromat.ipynb

    code_path = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/code'
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    from network import UNet_flex

    args_path    = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/exp_arguments_45500.pkl'
    weights_path = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/model_45500.pt'

    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)
    unet_args = SimpleNamespace(**args_dict)
    unet_base = UNet_flex(unet_args)
    state = torch.load(weights_path, map_location='cpu', weights_only=False)
    unet_base.load_state_dict(state)
    unet_base.eval()

    class ResidualWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, y):
            return y - self.m(y)

    model = ResidualWrapper(unet_base).to(device)
    print(f'Loaded UNet_45500 model (UNet_flex) from {weights_path}')

elif args.model == 'conv3':
    # conv3: pretrained Denoiser (conv3_ln.pt) from Denoiser_Reconstruction
    # model class defined in Denoiser_Reconstruction/models/denoiser.py
    recon_path = str((Path('/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model') / 'Denoiser_Reconstruction').resolve())
    if recon_path not in sys.path:
        sys.path.insert(0, recon_path)
    from models.denoiser import Denoiser

    # default arg values taken from Denoiser_Reconstruction/utils/helper.py
    # (imported directly here to avoid pulling in h5py via main.py)
    lq_args = SimpleNamespace(
        padding=1,
        kernel_size=3,
        num_kernels=64,
        num_layers=20,
        im_channels=3,
    )

    weights_path = Path('/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model') / 'Denoiser_Reconstruction' / 'assets' / 'conv3_ln.pt'
    model = Denoiser(lq_args)
    # Read the weights file from disk into memory as a dictionary.
    # Each entry in the dictionary is one layer's worth of learned numbers,
    # e.g. state['conv_layers.0.weight'] = tensor of numbers for layer 1.
    # map_location='cpu' loads onto CPU RAM regardless of where it was trained.
    # weights_only=False is needed because this file was saved in an older format.
    # At this point the numbers are just sitting in 'state' -- not yet in the network.
    state = torch.load(weights_path, map_location='cpu', weights_only=False)
    # Slot the numbers from 'state' into the correct layers of the network
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    print(f'Loaded conv3 model (Denoiser) from {weights_path}')

else:
    raise ValueError(f'Unknown model "{args.model}". Use "UNet", "UNet_45500", or "conv3".')

# STEP 5: Select seed so results can be reproduced.
# Skipped when seed is not given. Each run will produce a different image.
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f'Random seed: {args.seed}')

# STEP 6: Import sample_prior from Denoiser_Reconstruction/inverse/sampler.py
_recon_path = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction'
if _recon_path not in sys.path:
    sys.path.insert(0, _recon_path)
from inverse.sampler import sample_prior

# STEP 7: Initialize from white (Gaussian) noise and run the sampling loop.
# init is the starting image: random noise centered at 0.5 (the middle of
# the [0,1] image range). Shape is (3, H, W) -- sample_prior adds the batch dim.
# sample_prior returns all_ys: a list of numpy images (H, W, 3) saved every
# args.stride iterations, showing the image getting cleaner over time.
H = W = args.size
init = 0.5 + torch.randn(3, H, W)
# sample from the implicit prior
# def sample_prior(model, init, h_init=0.01, beta=0.01, 
#                  sig_end=0.005, stride=10, fix_h=False):
all_ys = sample_prior(model, init,
                      h_init=args.h_init,
                      beta=args.beta,
                      sig_end=args.sig_end,
                      stride=args.stride)

# The last entry in all_ys is the final denoised image
final = np.clip(all_ys[-1], 0, 1)

# ---- OLD: hand-written loop (replaced by sample_prior call above) ----
# H = W = args.size
# y = (0.5 + torch.randn(1, 3, H, W)).to(device)
# n = y.numel()
# def log_grad(y):
#     with torch.no_grad():
#         return -model(y)
# sigma = torch.norm(log_grad(y)) / np.sqrt(n)
# snapshots = []
# t = 1
# while sigma > args.sig_end:
#     h = (args.h_init * t) / (1 + args.h_init * (t - 1))
#     d = log_grad(y)
#     sigma = torch.norm(d) / np.sqrt(n)
#     if sigma > 1e2:
#         warnings.warn('Divergence detected -- restarting with larger beta.')
#         args.beta *= 2
#         y = (0.5 + torch.randn(1, 3, H, W)).to(device)
#         snapshots = []
#         t = 1
#         continue
#     gamma = np.sqrt((1 - args.beta * h) ** 2 - (1 - h) ** 2) * sigma
#     noise = torch.randn_like(y)
#     y = y + h * d + gamma * noise
#     if (t - 1) % args.stride == 0:
#         print(f'iter {t:4d}  sigma {sigma.item():.4f}')
#         snap = np.clip(y.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
#         snapshots.append((t, snap))
#     t += 1
# final = np.clip(y.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
# snapshots.append((t, final))

# STEP 8: Plot a 1x5 grid showing the image at 5 evenly-spaced points
# during the run: pure noise -> ... -> final result.
# all_ys is a plain list of numpy arrays (H, W, 3) -- one saved every
# args.stride iterations by sample_prior.

# Pick 5 evenly-spaced snapshots from across the full run
n_panels = 5
if len(all_ys) <= n_panels:
    chosen_indices = list(range(len(all_ys)))
    chosen = all_ys
else:
    # Spread indices evenly from 0 to len-1 inclusive so we always get
    # the very first (noisiest) and very last (cleanest) snapshots
    chosen_indices = [int(round(i * (len(all_ys) - 1) / (n_panels - 1)))
                      for i in range(n_panels)]
    chosen = [all_ys[i] for i in chosen_indices]

fig, axs = plt.subplots(1, n_panels, figsize=(n_panels * 3, 3))
for panel_i, (ax, img) in enumerate(zip(axs, chosen)):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'snapshot {panel_i + 1}/{n_panels}')
    ax.axis('off')

model_label = {'UNet': 'UNet_flex 240k (model.pt)',
               'UNet_45500': 'UNet_flex 45.5k (model_45500.pt)',
               'conv3': 'Denoiser conv3_ln'}.get(args.model, args.model)
fig.suptitle(f'Denoising from white noise  |  model: {model_label}', fontsize=13)
fig.tight_layout()

# Save the plot as a PNG next to the output image
plot_path = args.output.replace('.png', '_progress.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Progress plot saved to {plot_path}')
plt.show()

# Also save the final image on its own
from PIL import Image
Image.fromarray((final * 255).astype(np.uint8)).save(args.output)
print(f'Final image saved to {args.output}')

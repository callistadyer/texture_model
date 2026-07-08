"""
sample_from_prior.py

Starts from a white noise image and iteratively denoises it using a trained
model, producing a sample from the model's learned image prior.

This is the unconstrained version of linear_inverse (Denoiser_Reconstruction/
inverse/solver.py). It runs the same stochastic gradient ascent loop, but
linear_inverse adds a measurement-consistency correction to each step, 
while this has no measurement at all and lets the image settle wherever the prior pulls it.

Usage:
    python sample_from_prior.py --model UNet
    python sample_from_prior.py --model UNet_45500
    python sample_from_prior.py --model UNet_splitA
    python sample_from_prior.py --model UNet_splitB
    python sample_from_prior.py --model conv3
    python sample_from_prior.py --model UNet --size 128 --output my_sample.png
    python sample_from_prior.py --model conv3 --seed 42 --sig_end 0.005

Arguments:
    --model     Which denoiser to use (default: UNet)
                  "UNet"        = UNet_flex trained on 240k ImageNet images (model.pt)
                  "UNet_45500"  = UNet_flex trained on 45.5k ImageNet images (model_45500.pt)
                  "UNet_splitA" = UNet_flex trained on first half of each ImageNet class (model_splitA.pt)
                  "UNet_splitB" = UNet_flex trained on second half of each ImageNet class (model_splitB.pt)
                  "conv3"       = pretrained conv3_ln Denoiser (Denoiser_Reconstruction/assets/conv3_ln.pt)
    --size      Side length of the output image in pixels (default: 80,
                which matches the 80x80 patches the model was trained on;
                use --size 48 for conv3 which was trained on 48x48 patches)
    --output    Path to save the final denoised image.
                If not given, the file is saved to:
                  sampleFromPriorResults/<model>/sample_<seed>_<model>.png
                Outputs are routed into subfolders by model:
                  UNet        -> sampleFromPriorResults/UNet/
                  UNet_45500  -> sampleFromPriorResults/UNet_45500/
                  UNet_splitA -> sampleFromPriorResults/UNet_splitA/
                  UNet_splitB -> sampleFromPriorResults/UNet_splitB/
                  conv3       -> sampleFromPriorResults/conv3/
    --seed      Random seed for reproducibility (default: no seed)
    --h_init    Initial step size for the gradient updates (default: 0.01)
    --beta      Controls how much noise is re-injected each step (default: 0.01)
    --sig_end   Stopping threshold: loop ends when estimated noise magnitude
                drops below this value (default: 0.005)
    --stride    Print progress every this many iterations (default: 50)
"""

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

results_dir = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/sampleFromPriorResults'
texture_dir = Path('/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model')
recon_dir   = texture_dir / 'Denoiser_Reconstruction'

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Sample from denoiser prior')
parser.add_argument('--model',   type=str,   default='UNet')
parser.add_argument('--size',    type=int,   default=80)
parser.add_argument('--output',  type=str,   default=None)
parser.add_argument('--seed',    type=int,   default=None)
parser.add_argument('--h_init',  type=float, default=0.01)
parser.add_argument('--beta',    type=float, default=0.01)
parser.add_argument('--sig_end', type=float, default=0.005)
parser.add_argument('--stride',  type=int,   default=50)
args = parser.parse_args()

# build output filename if not given
if args.output is None:
    seed_label  = f'seed{args.seed}' if args.seed is not None else 'seedrand'
    output_dir  = Path(results_dir) / args.model
    args.output = str(output_dir / f'sample_{seed_label}_{args.model}.png')

Path(args.output).parent.mkdir(parents=True, exist_ok=True)

# ── device selection (same pattern as recon_visualize_dichromat.ipynb) ───────

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# ── load model ────────────────────────────────────────────────────────────────
#
# Two models, same steps:
#   1. Make the model class importable
#   2. Load saved architecture config (exp_arguments.pkl) or use defaults
#   3. Build the empty network, then fill it with trained weights (load_state_dict)
#   4. Set to eval mode
#
# UNet_flex (model.pt / model_45500.pt / model_splitA.pt / model_splitB.pt):
# All UNet_flex variants share the same architecture (exp_arguments.pkl) and
# differ only in training data. UNet_flex was trained with skip=False, so
# model(noisy) = clean image. But sample_prior (and linear_inverse) expect
# model(y) = noise residual, i.e. model(y) = noisy - clean.
# ResidualWrapper converts: forward(y) = y - UNet_flex(y) = noise residual.
# This is identical to what recon_visualize_dichromat.ipynb does.
#
# conv3 Denoiser (conv3_ln.pt):
# The Denoiser already outputs the noise residual directly (trained with
# skip=True convention), so no wrapper is needed.

code_path = str(texture_dir / 'code')
if code_path not in sys.path:
    sys.path.insert(0, code_path)

recon_path = str(recon_dir)
if recon_path not in sys.path:
    sys.path.insert(0, recon_path)

if args.model in ('UNet', 'UNet_45500', 'UNet_splitA', 'UNet_splitB'):
    from network import UNet_flex

    pkl_file = {
        'UNet':       'exp_arguments.pkl',
        'UNet_45500': 'exp_arguments_45500.pkl',
        'UNet_splitA': 'exp_arguments.pkl',
        'UNet_splitB': 'exp_arguments.pkl',
    }[args.model]
    weights_file = {
        'UNet':       'model.pt',
        'UNet_45500': 'model_45500.pt',
        'UNet_splitA': 'model_splitA.pt',
        'UNet_splitB': 'model_splitB.pt',
    }[args.model]

    with open(texture_dir / pkl_file, 'rb') as f:
        unet_args = SimpleNamespace(**pickle.load(f))

    unet_base = UNet_flex(unet_args)
    unet_base.load_state_dict(torch.load(texture_dir / weights_file, map_location='cpu', weights_only=False))
    unet_base.eval()

    # Wrap so sample_prior receives noise residual instead of clean image
    class ResidualWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, y):
            return y - self.m(y)   # noisy - clean = noise residual

    model = ResidualWrapper(unet_base).to(device)
    print(f'Loaded {args.model} (UNet_flex) from {weights_file}')
    print(device)
    
elif args.model == 'conv3':
    from models.denoiser import Denoiser

    # These were grabbed from the defaults in Denoiser_Reconstruction/utils/helper.py
    lq_args = SimpleNamespace(padding=1, kernel_size=3, num_kernels=64,
                               num_layers=20, im_channels=3)
    weights_path = recon_dir / 'assets' / 'conv3_ln.pt'

    model = Denoiser(lq_args)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    model.eval().to(device)
    print(f'Loaded conv3 Denoiser from {weights_path}')
    print(device)

else:
    raise ValueError(f'Unknown model "{args.model}". Use "UNet", "UNet_45500", "UNet_splitA", "UNet_splitB", or "conv3".')

# ── seed ──────────────────────────────────────────────────────────────────────

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f'Random seed: {args.seed}')

# ── run the sampling loop ─────────────────────────────────────────────────────
#
# This now calls sample_prior from Denoiser_Reconstruction/inverse/sampler.py.
# That function runs the following loop:
#
#   log_grad = lambda y: -model(y)      # prior gradient = -noise_residual
#   n = number of pixels
#   y = init (white noise image, unsqueezed to (1,C,H,W))
#   sigma = ||log_grad(y)|| / sqrt(n)   # initial noise estimate
#
#   while sigma > sig_end:
#       h = (h_init * t) / (1 + h_init * (t-1))   # adaptive step size
#       d = log_grad(y)                             # = -model(y)
#       sigma = ||d|| / sqrt(n)                     # update noise estimate
#       gamma = sqrt((1 - beta*h)^2 - (1-h)^2) * sigma  # noise to inject
#       noise = randn(size of y)
#       y = y + h*d + gamma*noise                   # gradient step + noise
#       t += 1
#
# linear_inverse (solver.py) runs the same loop but replaces d with:
#   d = d - R_T(R(d)) + proj - R_T(R(y))
# which projects out the measurable component and adds data consistency.
# Here there is no measurement, so d is used directly.

from inverse.sampler import sample_prior

H = W = args.size
init = 0.5 + torch.randn(3, H, W)   # start from white noise centered at 0.5

all_ys = sample_prior(model, init,
                      h_init=args.h_init,
                      beta=args.beta,
                      sig_end=args.sig_end,
                      stride=args.stride)

# ── plot and save ─────────────────────────────────────────────────────────────
#
# all_ys is a list of numpy images (H, W, 3) saved every args.stride iterations.
# We pick 5 evenly-spaced snapshots to show the image evolving from noise to output.
# Two files are saved:
#   <output>_progress.png  -- 1x5 panel showing snapshots across the run
#   <output>.png           -- the final denoised image on its own

final = np.clip(all_ys[-1], 0, 1)

# pick 5 evenly-spaced snapshots (always includes first and last)
n_panels = 5
if len(all_ys) <= n_panels:
    chosen = all_ys
else:
    chosen_indices = [int(round(i * (len(all_ys) - 1) / (n_panels - 1))) for i in range(n_panels)]
    chosen = [all_ys[i] for i in chosen_indices]

fig, axs = plt.subplots(1, n_panels, figsize=(n_panels * 3, 3))
for ax, img in zip(axs, chosen):
    ax.imshow(np.clip(img, 0, 1))
    ax.axis('off')

model_label = {'UNet':        'UNet_flex 240k (model.pt)',
               'UNet_45500':  'UNet_flex 45.5k (model_45500.pt)',
               'UNet_splitA': 'UNet_flex splitA (model_splitA.pt)',
               'UNet_splitB': 'UNet_flex splitB (model_splitB.pt)',
               'conv3':       'Denoiser conv3_ln'}.get(args.model, args.model)
seed_label = f'seed {args.seed}' if args.seed is not None else 'no seed'
fig.suptitle(f'Denoising from white noise  |  model: {model_label}  |  {seed_label}', fontsize=13)
fig.tight_layout()

# save progress plot and final image
plot_path = args.output.replace('.png', '_progress.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Progress plot saved to {plot_path}')
plt.show()

from PIL import Image
Image.fromarray((final * 255).astype(np.uint8)).save(args.output)
print(f'Final image saved to {args.output}')

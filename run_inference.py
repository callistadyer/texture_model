import os
import sys
import argparse
sys.path.insert(0, '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/code')

import torch
from model_loader_func import load_learned_model
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# All run_inference.py outputs are saved here, regardless of which model was used
results_dir = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/runInferenceResults'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',    type=str, required=True,  help='path to input image')
    parser.add_argument('--noise',    type=int, default=50,     help='noise level (0-255)')
    parser.add_argument('--output',   type=str, default=None,
                         help='output filename (just the name - it is always saved into runInferenceResults/); '
                              'default includes the image name, model name, and noise level')
    parser.add_argument('--model_dir',type=str, default='/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/models_trained/UNet_full_240541imgs_1000epochs/', help='folder containing model.pt and exp_arguments.pkl')
    args = parser.parse_args()

    # Make sure the results folder exists
    os.makedirs(results_dir, exist_ok=True)

    # Name of the model used, taken from the last folder in --model_dir
    # (e.g. '.../models_trained/UNet_full_240541imgs_483epochs/' -> 'UNet_full_240541imgs_483epochs')
    model_dir_no_trailing_slash = args.model_dir.rstrip('/')
    model_label = os.path.basename(model_dir_no_trailing_slash)

    # Name of the input image, without its file extension
    image_filename = os.path.basename(args.image)
    image_label = os.path.splitext(image_filename)[0]

    if args.output is None:
        # No filename given - build one from the image name, the model name, and the noise level
        output_filename = f'{image_label}_{model_label}_noise{args.noise}.png'
    else:
        # A filename was given - still save it into results_dir, using just the name part
        output_filename = os.path.basename(args.output)

    args.output = os.path.join(results_dir, output_filename)

    print(f'Loading model from {args.model_dir}')
    model = load_learned_model(args.model_dir)
    model.eval()

    print(f'Loading image from {args.image}')
    img = Image.open(args.image).convert('RGB')
    # Callista edit: the UNet has 3 pooling layers (2^3 = 8), so H and W must be divisible by 8.
    # If not, pooling halves a dimension (e.g. 125 -> 62) and upsampling doubles it back (62 -> 124),
    # which no longer matches the 125-pixel skip connection saved before pooling, causing a crash.
    w, h = img.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    img = img.resize((new_w, new_h), Image.BICUBIC)
    img_tensor = T.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)

    noise_level = args.noise / 255.0
    noisy = (img_tensor + torch.randn_like(img_tensor) * noise_level).clamp(0, 1)

    print(f'Running inference with noise level {args.noise}...')
    with torch.no_grad():
        denoised = model(noisy)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_tensor[0].permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(noisy[0].permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title(f'Noisy (level={args.noise})')
    axes[1].axis('off')
    axes[2].imshow(denoised[0].permute(1, 2, 0).detach().clamp(0, 1))
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Saved result to {args.output}')
    plt.show()

if __name__ == '__main__':
    main()

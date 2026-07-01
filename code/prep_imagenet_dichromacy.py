import numpy as np
import os
import sys
import time
import torch
import scipy.io
from dataloader_func import load_dataset, load_nested_dataset, prep_dataset
import argparse

# Callista change 6/23/26 - add paths for dichromacy simulation
sys.path.insert(0, '/mnt/home/cdyer/colorcorrection/display')
sys.path.insert(0, '/mnt/home/cdyer/colorcorrection/texture_model/Denoiser_Reconstruction/helpers')
from loadDisplay import loadDisplay
from DichromRenderLinear import DichromRenderLinear

# Callista edit: load gamma table for sRGB -> linear light conversion
_GAMMA_TABLE = scipy.io.loadmat(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'gamma.mat')
)['gammaTable']

def gamma_linear(data):
    # data: torch tensor (B, C, H, W) in [0,1], sRGB encoded
    # returns: torch tensor (B, C, H, W) in [0,1], linear light
    data_np = data.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
    out = np.zeros_like(data_np)
    for c in range(data_np.shape[-1]):
        out[:, :, :, c] = np.interp(data_np[:, :, :, c],
            np.linspace(0.0, 1.0, _GAMMA_TABLE.shape[0]), _GAMMA_TABLE[:, c])
    return torch.from_numpy(out).permute(0, 3, 1, 2).float()  # (B, C, H, W)

# Callista change 6/23/26 - apply dichromacy simulation to a full class tensor at once
def apply_dichromacy_to_tensor(batch, Disp, dichromat_type):
    # batch is a tensor of shape (B, C, H, W): B images, C=3 color channels, H height, W width
    B, C, H, W = batch.shape
    # Convert to float64 numpy array so matrix math stays precise
    arr = batch.numpy().astype(np.float64)
    # Rearrange to "cal format": (3, B*H*W) where each column is one pixel's RGB values.
    # transpose moves channels to front: (B,C,H,W) -> (C,B,H,W), then flatten everything after C
    rgb_cal = arr.transpose(1, 0, 2, 3).reshape(C, -1)   # (3, B*H*W)
    # Convert linear RGB to LMS cone excitations using the monitor's color matrix (3x3 @ 3xN = 3xN)
    lms_cal = Disp.M_rgb2cones @ rgb_cal                  # (3, B*H*W)
    # Run the dichromacy simulation. Returns linear RGB cal format already clipped to [0,1]
    # the _ variables are the simulated LMS and the 3x3 transformation matrix (not needed here)
    _, rgb_lin_di_cal, _ = DichromRenderLinear(lms_cal, dichromat_type, Disp)
    # Reshape back from cal format (3, B*H*W) to image tensor (B, C, H, W)
    di_arr = rgb_lin_di_cal.reshape(C, B, H, W).transpose(1, 0, 2, 3)
    # Convert back to float32 torch tensor to match the format of the rest of the dataset
    return torch.from_numpy(di_arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    # Callista change 6/23/26 - dichromat type argument
    parser.add_argument('--dichromat_type', type=str, default='Deuteranopia',
                        help='Deuteranopia | Protanopia | Tritanopia')
    args = parser.parse_args()

    # Callista change 6/23/26 - load display parameters once, shared across all images
    Disp = loadDisplay()
    dtype = args.dichromat_type
    dtype_lower = dtype.lower()
    print(f'Dichromat type: {dtype}')

    start_time_total = time.time()

    # dir_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/nano_imagenet/'
    # dir_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/imagenet/'
    # dir_path ='/mnt/home/zkadkhodaie/ceph/datasets/imagenet/'
    # Callista edit: read from local copy and save .pt files there too
    dir_path = '/mnt/home/cdyer/ceph/images/imagenet/'
    folder_names = os.listdir(dir_path + 'train/')

    train_sets = []
    # Callista edit: track names in the same order as tensors so we can search by label later
    train_names = []
    gamma_check_done = False
    for i, name in enumerate(folder_names):
        try:
            # Callista edit: removed prep_dataset call - load_dataset already returns float (B,C,H,W).
            # calling prep_dataset after would apply permute(0,3,1,2) a second time, corrupting shape to (B,W,C,H)
            data = load_dataset(dir_path + 'train/'+name+'/',s=(80,80), crop=True)
            # data = prep_dataset(data, grayscale=False)
            # Callista edit: apply inverse gamma correction (RGB -> linear) before saving
            # on the first successful class, print pixel values before and after to confirm gamma is working
            if not gamma_check_done:
                print(f'[gamma check] mean pixel value BEFORE gamma correction: {data.mean():.4f}')
            data = gamma_linear(data)
            if not gamma_check_done:
                print(f'[gamma check] mean pixel value AFTER gamma correction: {data.mean():.4f}')
                print('[gamma check] values should be lower after correction (linear light is darker than sRGB)')
                gamma_check_done = True
            # Callista change 6/23/26 - apply dichromacy simulation
            data = apply_dichromacy_to_tensor(data, Disp, dtype)
            train_sets.append(data)
            train_names.append(name)
            # Callista edit: progress report - print class name and running image count
            print(f'[train {i+1}/{len(folder_names)}] loaded {name}: {data.shape[0]} images | total so far: {sum(d.shape[0] for d in train_sets)}')
        except Exception as e:
            print(f'[train {i+1}/{len(folder_names)}] SKIPPED {name}: {e}')

    # Callista change 6/23/26 - save with dichromacy suffix
    torch.save(train_sets, dir_path + f'train_80x80_color_dichromacy_{dtype_lower}_list.pt')
    # Callista edit: save class names alongside tensors (same order) so check_classes.py can search by label
    torch.save(train_names, dir_path + f'train_class_names_dichromacy_{dtype_lower}.pt')
    print(f'train set saved: {len(train_sets)} classes, {sum(d.shape[0] for d in train_sets)} images total')

    test_sets = []
    for i, name in enumerate(folder_names):
        try:
            # Callista edit: removed prep_dataset call - same reason as above
            data = load_dataset(dir_path + 'val/'+name+'/',s=(80,80), crop=True)
            # data = prep_dataset(data, grayscale=False)
            # Apply inverse gamma correction (RGB -> linear) before saving
            data = gamma_linear(data)
            # Callista change 6/23/26 - apply dichromacy simulation
            data = apply_dichromacy_to_tensor(data, Disp, dtype)
            test_sets.append(data)
            print(f'[val {i+1}/{len(folder_names)}] loaded {name}: {data.shape[0]} images | total so far: {sum(d.shape[0] for d in test_sets)}')
        except Exception as e:
            print(f'[val {i+1}/{len(folder_names)}] SKIPPED {name}: {e}')

    # Callista change 6/23/26 - save with dichromacy suffix
    torch.save(test_sets, dir_path + f'test_80x80_color_dichromacy_{dtype_lower}_list.pt')
    print(f'val set saved: {len(test_sets)} classes, {sum(d.shape[0] for d in test_sets)} images total')


    print("--- %s seconds ---" % (time.time() - start_time_total))




if __name__ == "__main__" :
    main()

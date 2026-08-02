import numpy as np
import os
import time
import torch
import scipy.io
from dataloader_func import load_dataset, load_nested_dataset, prep_dataset
import argparse

# gamma table for sRGB -> linear light conversion
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


def main():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.parse_args()

    start_time_total = time.time()

    # TODO: the imagenet dataset is no longer at this path in Callista's ceph folder -
    # find its new location and update dir_path before running this script again.
    dir_path = '/mnt/home/cdyer/ceph/images/imagenet/'
    # dir_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/nano_imagenet/'
    # dir_path = '/mnt/home/gkrawezik/ceph/AI_DATASETS/imagenet/'
    # dir_path ='/mnt/home/zkadkhodaie/ceph/datasets/imagenet/'
    folder_names = os.listdir(dir_path + 'train/')

    # Builds train_80x80_color_list.pt: a list of ~1000 tensors, one per ImageNet class,
    # each shaped (N_class, 3, 80, 80) - not a single dataset tensor. `data` below holds
    # just one class at a time; train_sets accumulates all classes into that list, which
    # is what gets saved to disk and later loaded back as `full_train` in main.py.
    train_sets = []
    train_names = []  # same order as train_sets, so classes can be looked up by name later
    gamma_check_done = False
    for i, name in enumerate(folder_names):
        try:
            # load_dataset already returns float (B,C,H,W); calling prep_dataset after
            # would permute a second time and corrupt the shape to (B,W,C,H)
            data = load_dataset(dir_path + 'train/'+name+'/', s=(80,80), crop=True)

            # print pixel values before/after on the first class only, to confirm the
            # gamma correction is actually doing something (linear light should be darker)
            if not gamma_check_done:
                print(f'[gamma check] mean pixel value BEFORE gamma correction: {data.mean():.4f}')
            data = gamma_linear(data)
            if not gamma_check_done:
                print(f'[gamma check] mean pixel value AFTER gamma correction: {data.mean():.4f}')
                gamma_check_done = True

            train_sets.append(data)
            train_names.append(name)
            print(f'[train {i+1}/{len(folder_names)}] loaded {name}: {data.shape[0]} images | total so far: {sum(d.shape[0] for d in train_sets)}')
        except Exception as e:
            # a class folder can fail to load (e.g. all-grayscale images); skip it and keep going
            print(f'[train {i+1}/{len(folder_names)}] SKIPPED {name}: {e}')

    torch.save(train_sets, dir_path + 'train_80x80_color_list.pt')
    torch.save(train_names, dir_path + 'train_class_names.pt')
    print(f'train set saved: {len(train_sets)} classes, {sum(d.shape[0] for d in train_sets)} images total')

    # same process for the validation split
    test_sets = []
    for i, name in enumerate(folder_names):
        try:
            data = load_dataset(dir_path + 'val/'+name+'/', s=(80,80), crop=True)
            data = gamma_linear(data)
            test_sets.append(data)
            print(f'[val {i+1}/{len(folder_names)}] loaded {name}: {data.shape[0]} images | total so far: {sum(d.shape[0] for d in test_sets)}')
        except Exception as e:
            print(f'[val {i+1}/{len(folder_names)}] SKIPPED {name}: {e}')

    torch.save(test_sets, dir_path + 'test_80x80_color_list.pt')
    print(f'val set saved: {len(test_sets)} classes, {sum(d.shape[0] for d in test_sets)} images total')

    print("--- %s seconds ---" % (time.time() - start_time_total))


if __name__ == "__main__":
    main()

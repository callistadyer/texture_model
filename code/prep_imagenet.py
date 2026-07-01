import numpy as np
import os
import time
import torch
import scipy.io
from dataloader_func import load_dataset, load_nested_dataset, prep_dataset
import argparse

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



def main():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.parse_args()

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
            # Callista edit: apply inverse gamma correction (sRGB -> linear light) before saving
            # on the first successful class, print pixel values before and after to confirm gamma is working
            if not gamma_check_done:
                print(f'[gamma check] mean pixel value BEFORE gamma correction: {data.mean():.4f}')
            data = gamma_linear(data)
            if not gamma_check_done:
                print(f'[gamma check] mean pixel value AFTER gamma correction: {data.mean():.4f}')
                print('[gamma check] values should be lower after correction (linear light is darker than sRGB)')
                gamma_check_done = True
            train_sets.append(data)
            train_names.append(name)
            # Callista edit: progress report - print class name and running image count
            print(f'[train {i+1}/{len(folder_names)}] loaded {name}: {data.shape[0]} images | total so far: {sum(d.shape[0] for d in train_sets)}')
        except Exception as e:
            print(f'[train {i+1}/{len(folder_names)}] SKIPPED {name}: {e}')

    torch.save(train_sets, dir_path + 'train_80x80_color_list.pt')
    # Callista edit: save class names alongside tensors (same order) so check_classes.py can search by label
    torch.save(train_names, dir_path + 'train_class_names.pt')
    print(f'train set saved: {len(train_sets)} classes, {sum(d.shape[0] for d in train_sets)} images total')

    test_sets = []
    for i, name in enumerate(folder_names):
        try:
            # Callista edit: removed prep_dataset call - same reason as above
            data = load_dataset(dir_path + 'val/'+name+'/',s=(80,80), crop=True)
            # data = prep_dataset(data, grayscale=False)
            # Callista edit: apply inverse gamma correction (sRGB -> linear light) before saving
            data = gamma_linear(data)
            test_sets.append(data)
            print(f'[val {i+1}/{len(folder_names)}] loaded {name}: {data.shape[0]} images | total so far: {sum(d.shape[0] for d in test_sets)}')
        except Exception as e:
            print(f'[val {i+1}/{len(folder_names)}] SKIPPED {name}: {e}')

    torch.save(test_sets, dir_path + 'test_80x80_color_list.pt')
    print(f'val set saved: {len(test_sets)} classes, {sum(d.shape[0] for d in test_sets)} images total')


    print("--- %s seconds ---" % (time.time() - start_time_total))




if __name__ == "__main__" :
    main()




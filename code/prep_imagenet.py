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
    dir_path = '/mnt/home/cdyer/colorcorrection/images/imagenet/'
    folder_names = os.listdir(dir_path + 'train/')

    train_sets = []
    for name in folder_names:
        try:
            # Callista edit: removed prep_dataset call - load_dataset already returns float (B,C,H,W).
            # calling prep_dataset after would apply permute(0,3,1,2) a second time, corrupting shape to (B,W,C,H)
            data = load_dataset(dir_path + 'train/'+name+'/',s=(80,80), crop=True)
            # data = prep_dataset(data, grayscale=False)
            # Callista edit: apply inverse gamma correction (sRGB -> linear light) before saving
            data = gamma_linear(data)
            train_sets.append(data)
        except:
            pass

    torch.save(train_sets, dir_path + 'train_80x80_color_list.pt')

    test_sets = []
    for name in folder_names:
        try:
            # Callista edit: removed prep_dataset call - same reason as above
            data = load_dataset(dir_path + 'val/'+name+'/',s=(80,80), crop=True)
            # data = prep_dataset(data, grayscale=False)
            # Callista edit: apply inverse gamma correction (sRGB -> linear light) before saving
            data = gamma_linear(data)
            test_sets.append(data)
        except:
            pass


    torch.save(test_sets, dir_path + 'test_80x80_color_list.pt')


    print("--- %s seconds ---" % (time.time() - start_time_total))




if __name__ == "__main__" :
    main()




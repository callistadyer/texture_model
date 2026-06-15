import numpy as np
import os
import time
import torch
from dataloader_func import load_dataset, load_nested_dataset, prep_dataset
import argparse



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
            data = load_dataset(dir_path + 'train/'+name+'/',s=(80,80), crop=True)
            data = prep_dataset(data, grayscale=False)
            train_sets.append(data)
        except:
            pass

    torch.save(train_sets, dir_path + 'train_80x80_color_list.pt')

    test_sets = []
    for name in folder_names:
        try:
            data = load_dataset(dir_path + 'val/'+name+'/',s=(80,80), crop=True)
            data = prep_dataset(data, grayscale=False)
            test_sets.append(data)
        except:
            pass


    torch.save(test_sets, dir_path + 'test_80x80_color_list.pt')


    print("--- %s seconds ---" % (time.time() - start_time_total))




if __name__ == "__main__" :
    main()




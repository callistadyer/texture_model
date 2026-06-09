"""
Reads images directly from ImageNet zip files without extracting them.
Saves preprocessed .pt tensor files to a destination directory.

Nothing is written to the source imagenet directory.

Usage:
    python prep_imagenet_from_zip.py
    python prep_imagenet_from_zip.py --src_dir /other/path/ --dst_dir /other/out/
"""

import os
import io
import re
import time
import zipfile
import numpy as np
import torch
from PIL import Image
import argparse


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.PNG'}


def get_class_names(zf):
    """
    Return sorted list of synset class folder names inside the zip.
    Handles both flat structure (n01440764/img.JPEG) and
    nested structure (train/n01440764/img.JPEG).
    """
    classes = set()
    for name in zf.namelist():
        parts = [p for p in name.split('/') if p]
        for part in parts:
            if re.match(r'^n\d+$', part):
                classes.add(part)
                break
    return sorted(classes)


def load_class_from_zip(zf, class_name, s=(80, 80)):
    """
    Load all images for one class from an open ZipFile.
    Returns a float tensor of shape (N, C, H, W) in [0, 1], or None if no images found.
    """
    names = [
        n for n in zf.namelist()
        if class_name + '/' in n
        and not n.endswith('/')
        and os.path.splitext(n)[1] in IMAGE_EXTENSIONS
    ]
    if not names:
        return None

    images = []
    for name in names:
        try:
            with zf.open(name) as f:
                img = Image.open(io.BytesIO(f.read()))
                img = img.convert('RGB')
                # crop to square
                w, h = img.size
                side = min(w, h)
                img = img.crop(((w - side) // 2, (h - side) // 2,
                                (w + side) // 2, (h + side) // 2))
                img = img.resize(s, Image.BICUBIC)
                images.append(np.array(img))
        except Exception:
            continue

    if not images:
        return None

    arr = torch.tensor(np.array(images)).permute(0, 3, 1, 2).float() / 255.0
    return arr


def build_split(zip_path, size=(80, 80)):
    """Open zip and build a list of per-class tensors."""
    sets = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        class_names = get_class_names(zf)
        print(f'  Found {len(class_names)} classes in {os.path.basename(zip_path)}')
        for i, cls in enumerate(class_names):
            data = load_class_from_zip(zf, cls, size)
            if data is not None and data.shape[0] > 0:
                sets.append(data)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(class_names)} classes processed...')
    return sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        default='/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/',
                        help='Directory containing train.zip and val.zip')
    parser.add_argument('--dst_dir',
                        default='/mnt/home/cdyer/colorcorrection/',
                        help='Directory where .pt files will be saved')
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    src = args.src_dir.rstrip('/') + '/'
    dst = args.dst_dir.rstrip('/') + '/'

    start = time.time()
    print(f'Source: {src}')
    print(f'Output: {dst}\n')

    print('Building train set...')
    train_sets = build_split(src + 'train.zip')
    out = dst + 'train_80x80_color_list.pt'
    torch.save(train_sets, out)
    print(f'  Saved {len(train_sets)} classes -> {out}\n')

    print('Building val set...')
    val_sets = build_split(src + 'val.zip')
    out = dst + 'test_80x80_color_list.pt'
    torch.save(val_sets, out)
    print(f'  Saved {len(val_sets)} classes -> {out}\n')

    print(f'Total time: {time.time() - start:.1f}s')


if __name__ == '__main__':
    main()

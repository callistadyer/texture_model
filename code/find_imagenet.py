"""
Searches for an ImageNet directory starting at a root path.
Looks for a directory containing both train/ and val/ subdirectories,
each of which holds synset class folders with image files inside.

Usage:
    python find_imagenet.py
    python find_imagenet.py --root /some/other/path
"""

import os
import argparse


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}


def has_images(folder):
    try:
        for name in os.listdir(folder):
            if os.path.splitext(name)[1] in IMAGE_EXTENSIONS:
                return True
    except PermissionError:
        pass
    return False


def is_imagenet_root(path):
    """
    Returns True if path looks like an ImageNet root:
    - has a train/ subdir containing at least one folder with images
    - has a val/ subdir containing at least one folder with images
    """
    for split in ('train', 'val'):
        split_path = os.path.join(path, split)
        if not os.path.isdir(split_path):
            return False
        # at least one class subfolder must contain images
        try:
            class_folders = [
                f for f in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, f))
            ]
        except PermissionError:
            return False
        if not class_folders:
            return False
        found_images = False
        for cls in class_folders[:10]:  # sample first 10 to avoid slow full scan
            if has_images(os.path.join(split_path, cls)):
                found_images = True
                break
        if not found_images:
            return False
    return True


def search(root, max_depth=4):
    """BFS over directories up to max_depth looking for an ImageNet root."""
    root = os.path.abspath(root)
    queue = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        print(f"  checking: {current}")
        if is_imagenet_root(current):
            return current
        if depth < max_depth:
            try:
                subdirs = [
                    os.path.join(current, d)
                    for d in sorted(os.listdir(current))
                    if os.path.isdir(os.path.join(current, d))
                ]
                queue.extend((d, depth + 1) for d in subdirs)
            except PermissionError:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Find ImageNet directory')
    parser.add_argument('--root', default='/mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/',
                        help='Directory to start searching from')
    parser.add_argument('--max-depth', type=int, default=4,
                        help='How many levels deep to search (default: 4)')
    args = parser.parse_args()

    print(f"Searching from: {args.root}\n")

    result = search(args.root, max_depth=args.max_depth)

    if result:
        # ensure trailing slash to match how the code uses it
        result = result.rstrip('/') + '/'
        print(f"\nFound ImageNet root:\n  {result}")
        print(f"\nUse this in prep_imagenet.py:")
        print(f"  dir_path = '{result}'")
        print(f"\nUse this in main.py (--data_root_path or hardcoded):")
        print(f"  data_root_path = '{os.path.dirname(result.rstrip('/')) + '/'}'")
        print(f"  # (with --data_name imagenet, it appends 'imagenet/' to data_root_path)")
    else:
        print(f"\nNo ImageNet root found under {args.root} within depth {args.max_depth}.")
        print("Try increasing --max-depth or check that the path is mounted/accessible.")


if __name__ == '__main__':
    main()

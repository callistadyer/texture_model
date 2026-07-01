"""
check_classes.py

Search for specific objects in your ImageNet training set by keyword.
Parses the prep output log to get the exact list of classes that were loaded,
then cross-references with classes_indexed.txt to find matches.

Usage (run on server):
    python check_classes.py --keywords fruit banana watermelon
    python check_classes.py --list_all
"""

import os
import re
import argparse


def load_class_map(classes_folders_path):
    """Parse classes_folders.txt format: 'n02119789 1 kit_fox'
    Returns dict: {synset_id: label}"""
    class_map = {}
    with open(classes_folders_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                synset_id = parts[0]
                label = parts[2].replace('_', ' ')
                class_map[synset_id] = label
    return class_map


def parse_log(log_path):
    """Extract synset IDs and image counts from lines like:
       [train 55/194] loaded n01943899: 1279 images"""
    loaded = {}
    pattern = re.compile(r'\[train \d+/\d+\] loaded (n\d+): (\d+) images')
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                loaded[m.group(1)] = int(m.group(2))
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', nargs='+', default=[])
    parser.add_argument('--log_file', type=str,
                        default='/mnt/home/cdyer/colorcorrection/logs/imagenet_pipeline_6521289.out')
    parser.add_argument('--classes_file', type=str,
                        default='/mnt/home/cdyer/colorcorrection/images/classes_folders.txt')
    parser.add_argument('--list_all', action='store_true',
                        help='Print every class that was loaded')
    args = parser.parse_args()

    class_map = load_class_map(args.classes_file)
    print(f'Loaded {len(class_map)} class labels')

    loaded = parse_log(args.log_file)
    print(f'{len(loaded)} classes in training set, {sum(loaded.values())} total images\n')

    if args.list_all:
        for synset, n in sorted(loaded.items(), key=lambda x: class_map.get(x[0], x[0])):
            print(f'  {synset}  {class_map.get(synset, "(unknown)")}  ({n} images)')
        return

    if not args.keywords:
        print('Use --keywords to search, or --list_all to see everything')
        return

    total_found = 0
    for keyword in args.keywords:
        kw_lower = keyword.lower()
        matches = [(sid, class_map.get(sid, '(unknown)'), loaded[sid])
                   for sid in loaded if kw_lower in class_map.get(sid, '').lower()]
        not_loaded = [(sid, lbl) for sid, lbl in class_map.items()
                      if kw_lower in lbl.lower() and sid not in loaded]

        if not matches and not not_loaded:
            print(f'[{keyword}]  no ImageNet class found with this label')
        elif not matches:
            print(f'[{keyword}]  exists in ImageNet but was NOT in your training set:')
            for sid, lbl in not_loaded:
                print(f'    {sid}  "{lbl}"')
        else:
            print(f'[{keyword}]  PRESENT — {len(matches)} class(es):')
            for sid, lbl, n in sorted(matches, key=lambda x: x[1]):
                print(f'    {sid}  "{lbl}"  —  {n} images')
                total_found += n
            if not_loaded:
                print(f'  (in ImageNet but not loaded: {[lbl for _, lbl in not_loaded]})')
        print()

    print(f'Total matching images: {total_found}')


if __name__ == '__main__':
    main()

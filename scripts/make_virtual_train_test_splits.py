import random
import os
import json
from tqdm.auto import tqdm
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

def make_train_test_split(img_names, split=0.2, seed=420420):
    n = len(img_names)
    k = int(n * split)
    random.seed(seed)
    test = set(random.sample(img_names, k))
    train = set([img_name for img_name in img_names if not img_name in test])

    return list(train), list(test)


def contains_non_zero_labels(mask_dir):
    for mask in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, mask))
        if np.any(mask != 0):
            return True
        
    return False


def main(dirpath, verbose=False, logfile='/workspaces/xmem_training/dataset_progress.txt'):
    img_dir = os.path.join(dirpath, '480pVolumetricallyExtractedData', 'JPEGImages')
    msk_dir = os.path.join(dirpath, '480pVolumetricallyExtractedData', 'Annotations')
    dataset_file = os.path.join(dirpath, 'dataset.json')
    dataset_info = {}

    videos = sorted(os.listdir(img_dir))

    with open(logfile, 'a') as f:
        f.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}: starting {dirpath}\n")

    # first we have to find the non-zero examples
    non_zero_videos = []
    iterator = tqdm(videos, desc=f'finding non zero videos in {dirpath}') if verbose else videos
    for video in iterator:
        mask_dir = os.path.join(msk_dir, video)
        if contains_non_zero_labels(mask_dir):
            non_zero_videos.append(video)

    print(f"After filtering {len(videos)} videos, we are left with {len(non_zero_videos)} videos in {dirpath}")

    train, test = make_train_test_split(non_zero_videos)

    dataset_info['train'] = []
    for video in train:
        dataset_info['train'].append({
            'image': os.path.join('./480pVolumetricallyExtractedData/JPEGImages', video),
            'label': os.path.join('./480pVolumetricallyExtractedData/Annotations', video)
        })
    dataset_info['test'] = []
    for video in test:
        dataset_info['test'].append({
            'image': os.path.join('./480pVolumetricallyExtractedData/JPEGImages', video),
            'label': os.path.join('./480pVolumetricallyExtractedData/Annotations', video)
        })

    with open(dataset_file, 'w') as f:
        f.write(json.dumps(dataset_info))

    with open(logfile, 'a') as f:
        f.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}: Finished {dirpath}\n")

if __name__ == '__main__':
    from multiprocessing import Pool

    basedir = '/workspaces/data/tom/Totalsegmentator_MSD_format'
    tasks = sorted([os.path.join(basedir, p) for p in os.listdir(basedir)  if os.path.isdir(os.path.join(basedir, p))])
    APPROX_FILES_PER_TASK = 1000
    print(f'Making train test splits over {len(tasks)} tasks and thus {APPROX_FILES_PER_TASK * len(tasks)} datapoints')

    for task in tqdm(tasks, desc='iterating over tasks'):
        main(task, verbose=True)

    # with Pool(70) as pool:
    #     results = pool.map_async(main, tasks)
    #     pool.close()
    #     pool.join()

    # output = results.get()
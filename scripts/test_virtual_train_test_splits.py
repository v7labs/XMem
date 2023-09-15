import random
import os
import json
import cv2
import numpy as np

def check_task(task, n_trials):
    # ensure that there is a dataset.json
    dataset_info = os.path.join(task, 'dataset.json')
    assert os.path.exists(dataset_info)

    # sample n_trials training examples
    dataset_info = json.loads(open(dataset_info, 'r').read())
    datapoints = random.sample(dataset_info['train'], n_trials)
    for datapoint in datapoints:
        full_vid_path = os.path.join(task, datapoint['image'])
        full_msk_path = os.path.join(task, datapoint['label'])
        check_example(full_vid_path, full_msk_path)
    


def check_example(vid_path, msk_path):
    # assert that there is atleast one empty frame
    assert os.path.basename(vid_path) == os.path.basename(msk_path)
    masks = os.listdir(msk_path)

    n_non_empty_frames = 0
    for mask in masks:
        mask = cv2.imread(os.path.join(msk_path, mask))
        if np.any(mask != 0):
            n_non_empty_frames += 1
    
    print(f"{vid_path} has {n_non_empty_frames} non empty frames")

    assert n_non_empty_frames > 0


if __name__ == '__main__':
    # other fun things to look at
    # plot how many non empty frames there are
    # plot hwo many videos there are
    # plot the resolution of the videos
    
    # sample a task at random from the basedir
    # basedir = '/workspaces/data/tom/Totalsegmentator_MSD_format'
    # tasks = sorted(os.listdir(basedir))

    # while True:
    #     chosen_task = random.choice(tasks)
    #     check_task(os.path.join(basedir, chosen_task), 3)

    video = '/workspaces/data/tom/Totalsegmentator_MSD_format/Task_vertebrae_T4/480pVolumetricallyExtractedData/JPEGImages/0864'
    mask = '/workspaces/data/tom/Totalsegmentator_MSD_format/Task_vertebrae_T4/480pVolumetricallyExtractedData/Annotations/0864'
    
    check_example(video, mask)
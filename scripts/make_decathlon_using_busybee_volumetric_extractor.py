import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import nibabel as nib
import cv2
import numpy as np
import sys
import json
from pathlib import Path

import torch 
import numpy as np
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF
import subprocess

# torchvision resize wants tensors
# we handle cv2 (numpy) arrays
# there isn't really a numpy/scipy equivalent
# so here's where we end up
def numpy_resize(size, *args, **kwargs):
    def do_resize(x):
        handle_channels = x.ndim > 2
        if handle_channels:
            x = x.transpose([2,0,1])
        x = torch.tensor(x)
        x = x.unsqueeze(0) # torchvision fns are all batch_dim, h, w
        x = Resize(size, antialias=False, interpolation=TF.InterpolationMode.NEAREST)(x, *args, **kwargs)
        x = x.squeeze(0)
        x = x.numpy()
        if handle_channels:
            x = x.transpose([1,2,0])
        return x
    return do_resize

LABEL_TO_COLOUR = {
    0: [0, 0, 0],
    1: [0, 0, 128],
    2: [0, 128, 0],
    3: [128, 0, 0],
    4: [0, 128, 128]
}

def annot_frame_to_colour(frame):
    h, w = frame.shape

    colour_frame = np.zeros((h, w, 3))
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            colour = LABEL_TO_COLOUR[frame[i, j]]
            for k in range(3):
                colour_frame[i, j, k] = colour[k]

    return colour_frame

@ray.remote
class Processor:

    def __init__(self, idx):
        self.idx = idx
        self.to_480 = numpy_resize(480)
        self.absolute_path_to_volumetric_extractor = '/workspaces/xmem_training/busybee/volumetric-extractor'

    def call_volumetric_extractor(self, nifti_path, output_folder):
        subprocess.run(
            ['python', 'extractor.py', '-i', nifti_path, '-o', output_folder],
            cwd=self.absolute_path_to_volumetric_extractor,
            stdout=subprocess.DEVNULL
        )
        subprocess.run(
            ['rm slice_rg_*'],
            shell=True,
            cwd=output_folder,
            stdout=subprocess.DEVNULL
        )
        subprocess.run(
            ['rm manifest.json'],
            shell=True,
            cwd=output_folder,
            stdout=subprocess.DEVNULL
        )


    def nifti_to_frames(self, nifti_path, output_folder):
        self.call_volumetric_extractor(nifti_path, output_folder)
        # TODO: Maybe convert to JPEGs


    def nifti_to_annotation_frames(self, nifti_path, output_folder):
        self.call_volumetric_extractor(nifti_path, output_folder)


    def process_datapoint(self, task, datapoint):

        print(f"\tWorker #{self.idx} is working on task {Path(task).stem} and datapoint {Path(datapoint['image']).stem}")

        datapoint_basename = os.path.basename(datapoint['image']).split('.nii.gz')[0]
        annot_dir = os.path.join(task, '480pVolumetricallyExtractedData', 'Annotations', datapoint_basename)
        img_dir = os.path.join(task, '480pVolumetricallyExtractedData', 'JPEGImages', datapoint_basename)
        os.makedirs(annot_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        self.nifti_to_frames(
            os.path.join(task, datapoint['image']),
            os.path.join(img_dir)
        )
        self.nifti_to_annotation_frames(
            os.path.join(task, datapoint['label']),
            os.path.join(annot_dir)
        )


if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init()

    N_WORKERS = 70
    workers = [Processor.remote(i) for i in range(N_WORKERS)]

    basedir = '/workspaces/data/tom/Totalsegmentator_MSD_format'
    tasks = sorted([os.path.join(basedir, p) for p in os.listdir(basedir)  if os.path.isdir(os.path.join(basedir, p))])

    print(f'Running over {len(tasks)} tasks: {tasks}')

    for task in tasks:

        if os.path.exists(os.path.join(task, 'dataset.json')):
            with open(os.path.join(task, 'dataset.json')) as f:
                config = json.loads(f.read())
        else:
            config = {}
            config['training'] = []
            config['modality'] = {0: 'Unknown'}
            videos = os.listdir(os.path.join(task, 'imagesTr'))
            for video in videos:
                config['training'].append({
                    'image': os.path.join('imagesTr', video),
                    'label': os.path.join('labelsTr', video)
                })

        print(f"Produced {len(config['training'])} examples for {task}")

        # if len(config['modality'].keys()) > 1:
            # print(f"Skipping {task} as modality is {config['modality']}")
            # continue
            
        print(f"converting {len(config['training'])} datapoints from {Path(task).stem}...")

        if len(config['modality'].keys()) > 1:
            new_training_files = []
            for datapoint in config['training']:
                for modality_idx in config['modality'].keys():
                    dirname = os.path.dirname(datapoint['image'])
                    fpath_components = os.path.basename(datapoint['image']).split('.')
                    fpath_components[0] = fpath_components[0]+f'_{int(modality_idx):04d}'
                    new_training_files.append({
                        'image': os.path.join(dirname, '.'.join(fpath_components)),
                        'label': datapoint['label']
                    })
            config['training'] = new_training_files

        handles = [workers[i % N_WORKERS].process_datapoint.remote(task, datapoint) for i, datapoint in enumerate(config['training'])]
        results = [ray.get(handle) for handle in handles]

            


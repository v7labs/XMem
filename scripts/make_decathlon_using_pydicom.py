import ray
import nibabel as nib
import cv2
import numpy as np
import sys
import os
import json

import torch 
import numpy as np
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF

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


    def nifti_to_frames(self, nifti_path, output_folder, output_format, is_annot, fps=25, extra_modalities=[]):
        # Load NIfTI file
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()

        # Handle scans with multiple modalities
        if nifti_img.ndim == 3:
            n_modalities = 1
        elif nifti_img.ndim == 4:
            n_modalities = data.shape[3]
        else:
            raise RuntimeException('Dont know how to deal with nifti with less than 3 dims')

        for modality in list(range(n_modalities))+extra_modalities:

            # Create output directory
            assert nifti_path.endswith('.nii.gz')
            filename = os.path.basename(nifti_path).split('.nii.gz')[0]
            filename_inc_modality = f"{filename}_{modality}"
            frame_dir = os.path.join(output_folder, filename_inc_modality)
            os.makedirs(frame_dir, exist_ok=True)

            if n_modalities > 1 and len(extra_modalities)==0:
                data_3d = data[:, :, :, modality]
            else:
                data_3d = data

            if not is_annot:
                # Normalize data to the range [0, 255]
                normalized_data = 1 + (data_3d - np.min(data_3d)) * (254 / (np.max(data_3d) - np.min(data_3d)))
                normalized_data = normalized_data.astype(np.uint8)
            else:
                normalized_data = data_3d

            height, width, num_slices = normalized_data.shape

            # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use appropriate codec for MOV format
            # video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write each slice as a frame to the video
            for i in range(num_slices):
                frame = normalized_data[:, :, i].T
                if is_annot:
                    frame = annot_frame_to_colour(frame)
                frame = self.to_480(frame)
                frame_name = "{:05d}.".format(i) + output_format
                frame_dest = os.path.join(frame_dir, frame_name)
                if not is_annot:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(frame_dest, frame)

            # # Release video writer resources
            # video_writer.release()


    def process_datapoint(self, task, datapoint):

        print(f"\t\tWorker #{self.idx} is working on task {task} and datapoint {datapoint['image']}")

        with open(os.path.join(task, 'dataset.json')) as f:
                config = json.loads(f.read())

        annot_dir = os.path.join(task, '480pGoodLabelsRotated', 'Annotations')
        img_dir = os.path.join(task, '480pGoodLabelsRotated', 'JPEGImages')
        os.makedirs(annot_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        img_path = os.path.join(task, datapoint['image'])
        self.nifti_to_frames(img_path, img_dir, 'jpg', is_annot=False)

        img_path = os.path.join(task, datapoint['label'])
        # duplicate annotations for tasks with multiple modalities
        n_modalities = len(config['modality'].keys())
        if n_modalities > 1:
            self.nifti_to_frames(img_path, annot_dir, 'png', is_annot=True, extra_modalities=list(range(1, n_modalities)))
        else:
            self.nifti_to_frames(img_path, annot_dir, 'png', is_annot=True)




if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init()

    N_WORKERS = 45
    workers = [Processor.remote(i) for i in range(N_WORKERS)]

    basedir = './MedicalDecathlon'
    # tasks = [os.path.join(basedir, p) for p in os.listdir(basedir)  if os.path.isdir(os.path.join(basedir, p))]

    print(f'Running over {len(tasks)} tasks: {tasks}')

    for task in tasks:

        with open(os.path.join(task, 'dataset.json')) as f:
            config = json.loads(f.read())

        config['training'] = config['training'][:10]
            
        print(f"converting {len(config['training'])} datapoints from {task}...")

        handles = [workers[i % N_WORKERS].process_datapoint.remote(task, datapoint) for i, datapoint in enumerate(config['training'])]
        results = [ray.get(handle) for handle in handles]

            


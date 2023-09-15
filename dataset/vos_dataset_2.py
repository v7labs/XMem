from torch.utils.data.dataset import Dataset
from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed
import os
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import time
from tqdm import tqdm
import json

# ! im_mean is probably wrong for my dataset


class BinaryMaskVOSDataset(Dataset):
    def __init__(self, im_root, gt_root, frames_per_datapoint, max_jump=3, dataset_info=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.frames_per_datapoint = frames_per_datapoint
        self.max_num_obj = 1
        self.max_jump = max_jump

        self._setup_transforms()

        self.videos = []
        self.frames = {}

        if dataset_info is None:
            vid_list = sorted(os.listdir(self.im_root))
            # assert False, 'putting this here to ensure were supplying dataset info ' + im_root
        else:
            # print(f'Found supplied dataset_info file {dataset_info} so using that')
            dataset_info = json.loads(open(dataset_info, 'r').read())
            vid_list = sorted([os.path.basename(p['image']) for p in dataset_info['train']])

        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < self.frames_per_datapoint:
                print(f"Skipping '{vid}' as it has {len(frames)} frames and self.frames_per_datapoint is {self.frames_per_datapoint}")
                continue
            self.videos.append(vid)

        if len(self.videos) == 0:
            raise RuntimeError(f'Cant have a dataset with no videos: {self.im_root}')
        
        if len(self.videos) < len(vid_list):
            print(f"Accepted {len(self.videos)} videos out of {len(vid_list)} candidates in {self.im_root}")



    def _setup_transforms(self, finetune=True):

        self.image_resize = transforms.Compose([
            Resize(480, antialias=False, interpolation=TF.InterpolationMode.BILINEAR)
        ])

        self.mask_resize = transforms.Compose([
            Resize(480, antialias=False, interpolation=TF.InterpolationMode.NEAREST)
        ])

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.NEAREST, fill=0), # ! TODO: This is probably causing the tiling effect
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])


    def _sample_frames_idx(self, k, n, max_jump=3):
        # choose k frame idxs from n possible frames
        # assert that idxs are no more than max_jump away
        # this logic is directly from xmem paper
        # we can definitely simplify

        if k > n:
            print(f"Requested {k} frames from {n} so replicating first frame.")
            return [0]*(k-n) + list(range(n))

        frames_idx = [np.random.randint(n)]
        acceptable_set = set(
            range(
                max(0, frames_idx[-1] - max_jump),
                min(n, frames_idx[-1] + max_jump + 1)
            )
        ).difference(
            set(frames_idx)
        )
        while(len(frames_idx) < k):
            if len(list(acceptable_set)) == 0:
                print(f"Choosing {k} from {n} with {max_jump}")
            idx = np.random.choice(list(acceptable_set))
            frames_idx.append(idx)
            new_set = set(
                range(
                    max(0, frames_idx[-1] - max_jump),
                    min(n, frames_idx[-1]+ max_jump + 1)
                )
            )
            acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
        frames_idx = sorted(frames_idx)
        if np.random.rand() < 0.5:
            frames_idx = frames_idx[::-1]

        return frames_idx


    def _apply_transformations(self, images, masks):
        transformed_images = []
        transformed_masks = []

        sequence_seed = np.random.randint(2147483647)
        for image, mask in zip(images, masks):

            image = self.image_resize(image)
            mask = self.mask_resize(mask)

            reseed(sequence_seed)
            image = self.all_im_dual_transform(image)
            image = self.all_im_lone_transform(image)
            reseed(sequence_seed)
            mask = self.all_gt_dual_transform(mask)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            image = self.pair_im_dual_transform(image)
            image = self.pair_im_lone_transform(image)
            reseed(pairwise_seed)
            mask = self.pair_gt_dual_transform(mask)

            image = self.final_im_transform(image)
            mask = np.array(mask)

            transformed_images.append(image)
            transformed_masks.append(mask)

        return transformed_images, transformed_masks

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            return self._getitem(idx)
        except Exception as e:
            video = self.videos[idx]

            # just error reporting and debugging from here onwards

            info = {}
            info['name'] = video

            vid_im_path = os.path.join(self.im_root, video)
            vid_gt_path = os.path.join(self.gt_root, video)

            print(f"Exception with idx:{idx} and video: {vid_im_path}")

            # first filter to non-empty frames
            frames = sorted(os.listdir(vid_im_path))
            masks = sorted(os.listdir(vid_gt_path))

            # print(f"loaded up a video with {len(frames)} frames")
            non_empty_frames = []
            for idx, (frame, mask) in enumerate(zip(frames, masks)):
                assert frame.split('.')[0] == mask.split('.')[0]
                mask_path = os.path.join(vid_gt_path, mask)
                mask_obj = cv2.imread(mask_path)
                if np.any(mask_obj != 0):
                    non_empty_frames.append(idx)

            # pad idxs by 1 to include empty start and end frames to teach model how to start/end tracking
            start_idx = max(0, min(non_empty_frames) - 1)
            end_idx = min(len(frames)-1, max(non_empty_frames) + 1)

            breakpoint()

            raise e

    def _getitem(self, idx):
        video = self.videos[idx]

        info = {}
        info['name'] = video

        vid_im_path = os.path.join(self.im_root, video)
        vid_gt_path = os.path.join(self.gt_root, video)

        # first filter to non-empty frames
        frames = sorted(os.listdir(vid_im_path))
        masks = sorted(os.listdir(vid_gt_path))

        # print(f"loaded up a video with {len(frames)} frames")
        non_empty_frames = []
        for idx, (frame, mask) in enumerate(zip(frames, masks)):
            assert frame.split('.')[0] == mask.split('.')[0]
            mask_path = os.path.join(vid_gt_path, mask)
            mask_obj = cv2.imread(mask_path)
            if np.any(mask_obj != 0):
                non_empty_frames.append(idx)

        # pad idxs by 1 to include empty start and end frames to teach model how to start/end tracking
        start_idx = max(0, min(non_empty_frames) - 1)
        end_idx = min(len(frames)-1, max(non_empty_frames) + 1)

        # now do slightly more involved sampling within the non-empty region
        frames = [frames[i] for i in range(start_idx, end_idx+1)]
        # print(f"cutdown that video to {len(frames)} frames")
        # if len(frames) < self.frames_per_datapoint:
            # print(f"Warning: {video} has {len(frames)} non-empty frames but you are trying to select {self.frames_per_datapoint}")
        this_max_jump = min(len(frames), self.max_jump)
        frames_idx = self._sample_frames_idx(self.frames_per_datapoint, len(frames), max_jump=this_max_jump)
        
        for idx in frames_idx:
            if (idx < 0) or (idx >= len(frames)):
                print(f"ooops {len(frames)} frames but {frames_idx} idxs")
                breakpoint()
                
        selected_frames = [frames[idx] for idx in frames_idx]
        # print(f"and further sampled to {len(selected_frames)}")

        # read in the frames and apply transformations
        images = [Image.fromarray(cv2.imread(os.path.join(vid_im_path, frame))) for frame in selected_frames]
        masks = [Image.open(os.path.join(vid_gt_path, frame[:-4]+'.png')).convert('P') for frame in selected_frames]

        images, masks = self._apply_transformations(images, masks)
        
        image_tensor = torch.stack(images, 0)
        mask_tensor = np.stack(masks, 0)

        info['num_objects'] = 1

        # generate one-hot ground truth
        cls_gt = np.zeros((self.frames_per_datapoint, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        this_mask = (mask_tensor!=0)
        cls_gt[this_mask] = 1     # map every mask to 1 (formerly label_idx + 1)
        first_frame_gt[0, 0] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)


        # resize everything to 480

        data = {
            'rgb': image_tensor,  # (frames_per_datapoint, 384, 384) stacked frames tensor
            'first_frame_gt': first_frame_gt,   # (1, max_num_objects, 384, 384) tensor - mask for each object in first frame, one hot encoded
            'cls_gt': cls_gt,   # not one hot encoded, just a stacked list of index masks
            'selector': selector, # selector[i] = 1 if the i'th object in first_frame_gt is to be used, 0 otherwise
            'info': info # What?    # contains n_objects, name, maybe frames
        }
    
        return data


if __name__ == '__main__':
    dataset = BinaryMaskVOSDataset(
        '/workspaces/xmem_training/MedicalDecathlon/Task03_Liver/480pVolumetricallyExtractedData/trimmed-test/JPEGImages',
        '/workspaces/xmem_training/MedicalDecathlon/Task03_Liver/480pVolumetricallyExtractedData/trimmed-test/Annotations',
        frames_per_datapoint=3
    )
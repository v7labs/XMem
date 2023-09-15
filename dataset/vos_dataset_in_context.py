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
import uuid

from in_context_learning.dataset.dataset_factory import registry


class InContextVOSDataset(Dataset):
    def __init__(self, dataset_name, selection_method, frames_per_datapoint, draw_exclusively_from_support=False):
        self.dataset_name = dataset_name
        self.selection_method = selection_method
        self.frames_per_datapoint = frames_per_datapoint
        self.max_num_obj = 1
        self.draw_exclusively_from_support=draw_exclusively_from_support

        self._setup_transforms()

        self.iterator = self.infinite_iterator()

    def infinite_iterator(self):
        while True:
            # support set size is frames per datapoint - 1 since we append the target image
            self.dataset = registry[self.dataset_name](support_set_size=self.frames_per_datapoint-1, support_set_selection=self.selection_method, override_val=self.draw_exclusively_from_support)
            for x in self.dataset:
                yield x

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
            # im_normalization,
        ])


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
        return 1000

    def __getitem__(self, idx):
        try:
            image, mask, support_images, support_masks = next(self.iterator)
        except StopIteration:
            raise RuntimeError('Stop Iteration called on self iterator')

        # read in the frames and apply transformations
        images = support_images + [image]
        masks = support_masks + [mask]
        images = [Image.fromarray(image) for image in images]
        masks = [Image.fromarray(mask) for mask in masks]

        images, masks = self._apply_transformations(images, masks)
        
        image_tensor = torch.stack(images, 0)
        mask_tensor = np.stack(masks, 0)

        # is this really needed now?
        info = {}
        info['name'] = f"{str(uuid.uuid4())}.jpg"
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
    dataset = InContextVOSDataset(
        'coco-chair-all',
        'clip',
        frames_per_datapoint=3
    )
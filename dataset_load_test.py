# from dataset.vos_dataset_2 import BinaryMaskVOSDataset
from dataset.vos_dataset_in_context import InContextVOSDataset

from torch.utils.data import DataLoader, ConcatDataset
import os
import time

dataset = InContextVOSDataset(
    'voc-person-all',
    'clip',
    frames_per_datapoint=3
)

print(dataset[0])

# medical_root = '/workspaces/data/tom/Totalsegmentator_MSD_format'
# tasks = [p for p in os.listdir(medical_root) if os.path.isdir(os.path.join(medical_root, p))]

# datasets = []
# start = time.time()
# for task in tasks:
#     dataset = BinaryMaskVOSDataset(
#         os.path.join(medical_root, task, '480pVolumetricallyExtractedData', 'JPEGImages'), 
#         os.path.join(medical_root, task, '480pVolumetricallyExtractedData', 'Annotations'), 
#         frames_per_datapoint=8,
#         max_jump=5,
#         dataset_info=os.path.join(medical_root, task, 'dataset.json')
#     )

#     datasets.append(dataset)

# train_dataset = ConcatDataset(datasets)

# print(f"Took {time.time() - start:.2f}s to produce dataset of length {len(train_dataset)}")

# dataloader = DataLoader(train_dataset, 16, num_workers=32, drop_last=True)

# for idx, b in enumerate(dataloader):
#     print(idx)
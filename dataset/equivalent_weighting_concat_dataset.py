import torch
from torch.utils.data import Dataset

class EquivalentWeightingConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.n_datasets = len(datasets)

    def __len__(self):
        return self.n_datasets * max([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        dataset_idx = idx % self.n_datasets
        dataset = self.datasets[dataset_idx]
        
        idx = idx // self.n_datasets
        idx = idx % len(dataset)

        return dataset[idx]
    


if __name__ == '__main__':
    # testing
    class IntDataset(Dataset):
        def __init__(self, n, name):
            self.n = n
            self.name = name

        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            return idx, self.name
        
    d1 = IntDataset(3, 'Dataset_A')
    d2 = IntDataset(10, 'Dataset_B')
    d3 = IntDataset(4, 'Dataset_C')

    d = EquivalentWeightingConcatDataset([d1, d2, d3])

    print(f"d is of length {len(d)}")
    for i in range(len(d)):
        print(i, '=>', d[i])
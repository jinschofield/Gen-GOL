import torch
from torch.utils.data import Dataset

class RandomPatternDataset(Dataset):
    """
    Dataset of random binary Game of Life boards for baseline training.
    """
    def __init__(self, num_samples: int = 10000, grid_size: int = 32):
        self.num_samples = num_samples
        self.grid_size = grid_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # sample uniform random bits (0 or 1)
        x = torch.bernoulli(torch.full((1, self.grid_size, self.grid_size), 0.5))
        # random conditioning label for CF training: 0 or 1
        c = torch.randint(0, 2, (), dtype=torch.long)
        return x, c

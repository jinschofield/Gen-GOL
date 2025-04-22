"""
Dataset loader for Conway's Game of Life patterns stored as .npy arrays.
Supports on-the-fly augmentations: rotations, flips, translations.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class GolDataset(Dataset):
    def __init__(self, data_dir, augment=True, max_translation=None):
        """
        Args:
            data_dir (str): path to directory containing .npy pattern files
            augment (bool): apply random rotations, flips, translations
            max_translation (int, optional): max shift in pixels; default size//4
        """
        self.data_dir = data_dir
        self.paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.lower().endswith('.npy')
        ]
        if len(self.paths) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        sample = np.load(self.paths[0])
        if sample.ndim != 2:
            raise ValueError("Expected 2D arrays")
        self.size = sample.shape[0]
        self.augment = augment
        self.max_translation = max_translation or self.size // 4

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx])
        if self.augment:
            arr = self._apply_augment(arr)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
        return tensor

    def _apply_augment(self, arr):
        # Random rotation
        k = np.random.randint(0, 4)
        arr = np.rot90(arr, k)
        # Random flips
        if np.random.rand() < 0.5:
            arr = np.fliplr(arr)
        if np.random.rand() < 0.5:
            arr = np.flipud(arr)
        # Random translation
        tx = np.random.randint(-self.max_translation, self.max_translation + 1)
        ty = np.random.randint(-self.max_translation, self.max_translation + 1)
        arr = self._translate(arr, tx, ty)
        return arr

    def _translate(self, arr, tx, ty):
        H, W = arr.shape
        out = np.zeros_like(arr)
        # source start indices
        xs = max(0, -tx)
        ys = max(0, -ty)
        # target start
        xt = max(0, tx)
        yt = max(0, ty)
        # copy dimensions
        h = H - abs(tx)
        w = W - abs(ty)
        if h > 0 and w > 0:
            out[xt:xt + h, yt:yt + w] = arr[xs:xs + h, ys:ys + w]
        return out

"""
Dataset loader for Conway's Game of Life patterns stored as .npy arrays.
Supports on-the-fly augmentations: rotations, flips, translations.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.gol_simulator import simulate
import warnings
try:
    from utils.metrics import detect_period
except ImportError:
    detect_period = None

class GolDataset(Dataset):
    def __init__(self, data_dir, augment=True, max_translation=None, noise_prob=0.0):
        """
        Args:
            data_dir (str): path to directory containing .npy pattern files
            augment (bool): apply random rotations, flips, translations
            max_translation (int, optional): max shift in pixels; default size//4
            noise_prob (float, optional): probability of random cell flips as augmentation
        """
        self.data_dir = data_dir
        # collect all .npy paths
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith('.npy'):
                    all_files.append(os.path.join(root, f))
        if not all_files:
            raise ValueError(f"No .npy files found in {data_dir}")
        # auto-label by simulation: 0=die, 1=survive
        self.paths = []
        self.labels = []
        for p in all_files:
            arr = np.load(p).astype(np.uint8)
            hist = simulate(arr, steps=200)
            lbl = 1 if hist and hist[-1].sum() > 0 else 0
            self.paths.append(p)
            self.labels.append(lbl)
        sample = np.load(self.paths[0])
        if sample.ndim != 2:
            raise ValueError("Expected 2D arrays")
        self.size = sample.shape[0]
        self.augment = augment
        self.max_translation = max_translation or self.size // 4
        # probability of random cell flips as augmentation
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        arr = np.load(path)
        if self.augment:
            arr = self._apply_augment(arr)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
        label = self.labels[idx]
        return tensor, torch.tensor(label, dtype=torch.long)

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
        # random cell-level noise flips
        if self.noise_prob > 0:
            mask = np.random.rand(*arr.shape) < self.noise_prob
            arr = arr.copy()
            arr[mask] = 1 - arr[mask]
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

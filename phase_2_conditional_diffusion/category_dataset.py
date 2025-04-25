#!/usr/bin/env python3
"""
Dataset for Phase 2: load 32×32 patterns and their multi-class life-type labels from CSV.
CSV columns: [filepath, category]
Categories are mapped to integer indices 0..(K-1).
"""
import csv
from torch.utils.data import Dataset
import torch
import numpy as np

class CategoryDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # header
            for path, cat in reader:
                self.samples.append((path, cat))
        # build category mapping
        cats = sorted({cat for _, cat in self.samples})
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
        self.idx_to_cat = {idx: cat for cat, idx in self.cat_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat = self.samples[idx]
        arr = np.load(path).astype(np.uint8)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
        label = self.cat_to_idx[cat]
        return tensor, torch.tensor(label, dtype=torch.long)

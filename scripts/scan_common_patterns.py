#!/usr/bin/env python3
"""
Scan generated quota-model samples for common subpatterns in still-life and period-2 oscillators.
"""
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.unet import UNet
from models.diffusion import Diffusion

# --- CONFIGURATION ---
CKPT         = 'finished_models/model_final_quota.pt'
DEVICE       = 'cuda'  # or 'cpu'
TIMESTEPS    = 200
NUM_SAMPLES  = 1000
WINDOW_SIZES = [3, 4, 5, 6, 7, 8, 9]  # scan window sizes from 3x3 up to 9x9
STRIDE       = 1
TOP_K        = 9       # top patterns to display

# --- load model & diffusion ---
model = UNet(dropout=0.0, num_classes=5).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
diff = Diffusion(
    timesteps=TIMESTEPS,
    device=DEVICE,
    schedule='cosine',
    guidance_scale=1.0
)

# --- sampling utility ---
def sample_arrs(condition):
    """Generate NUM_SAMPLES binary grids conditioned on a class label or unconditionally."""
    c = None if condition is None else torch.full((NUM_SAMPLES,), condition, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        x = diff.sample(model, shape=(NUM_SAMPLES,1,32,32), c=c)
    arrs = (x.cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
    return arrs

# --- pattern extraction ---
def extract_patterns(arrs, win_size):
    patterns = {}
    h, w = arrs.shape[1], arrs.shape[2]
    for arr in arrs:
        for i in range(0, h - win_size + 1, STRIDE):
            for j in range(0, w - win_size + 1, STRIDE):
                sub = arr[i:i+win_size, j:j+win_size]
                key = tuple(sub.flatten().tolist())
                patterns[key] = patterns.get(key, 0) + 1
    return patterns

# --- visualization ---
def visualize_patterns(patterns, win_size, cond_name):
    """Plot top-K patterns in a grid and save to PNG."""
    # get top K
    top = sorted(patterns.items(), key=lambda x: -x[1])[:TOP_K]
    # setup grid
    ncol = int(np.ceil(np.sqrt(TOP_K)))
    nrow = int(np.ceil(TOP_K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
    axes = axes.flatten()
    # plot each pattern
    for idx, (key, count) in enumerate(top):
        pat = np.array(key).reshape(win_size, win_size)
        axes[idx].imshow(pat, cmap='gray')
        axes[idx].set_title(str(count))
        axes[idx].axis('off')
    # blank out unused
    for ax in axes[len(top):]:
        ax.axis('off')
    fig.suptitle(f"{cond_name}: top {TOP_K} patterns ({win_size}x{win_size})")
    fname = f"patterns_{cond_name}_{win_size}x{win_size}.png"
    fig.tight_layout()
    fig.savefig(fname)
    print(f"Saved {fname}")

# --- main workflow ---
def main():
    # scan still-life
    for cond_name, label in [('still_life', 1), ('oscillator_period_2', 2)]:
        print(f"[scan] Generating samples for {cond_name}…")
        arrs = sample_arrs(label)
        for ws in WINDOW_SIZES:
            print(f"[scan] Extracting {ws}x{ws} patterns…")
            pats = extract_patterns(arrs, ws)
            visualize_patterns(pats, ws, cond_name)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Extract and visualize the most common entire-grid 'other' patterns
(period !=1 or 2) from conditioned (alive) generations by the quota-trained model.
Filters for at least two adjacent live cells (including diagonals).
"""
import os, sys, argparse, csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# allow local imports
sys.path.append(os.path.dirname(__file__))
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import detect_period
from utils.gol_simulator import simulate


def main():
    p = argparse.ArgumentParser(description="Common entire-grid 'other' patterns")
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='path to quota-trained model checkpoint')
    p.add_argument('--class_label', type=int, default=1,
                   help='conditioning label for alive class')
    p.add_argument('--timesteps',   type=int, default=200,
                   help='diffusion timesteps')
    p.add_argument('--num_samples', type=int, default=300,
                   help='number of samples to generate')
    p.add_argument('--grid_size',   type=int, default=32,
                   help='grid height/width')
    p.add_argument('--threshold',   type=float, default=0.5,
                   help='binarization threshold')
    p.add_argument('--top_k',       type=int, default=5,
                   help='number of top patterns to display')
    p.add_argument('--device',      type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',     type=str, default='./common_other_grids',
                   help='output directory')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model
    device = args.device
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=device,
        schedule='linear',
        guidance_scale=1.0
    )

    # sample conditioned alive outputs
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        c = torch.full((args.num_samples,), args.class_label,
                       device=device, dtype=torch.long)
        print("Sampling conditioned alive outputs...")
        samples = diffusion.sample(model, shape, c=c)

    # binarize
    grids = (samples.squeeze(1).cpu().numpy() > args.threshold).astype(np.uint8)

    # filter 'other' patterns: not period-1 or 2, alive, adjacency, neighbor-rule validity, and change
    filtered = []
    for g in grids:
        hist = simulate(g, steps=args.timesteps)
        per = detect_period(hist)
        if per not in (1, 2) and hist[-1].sum() > 0:
            final = hist[-1]
            arr = final.astype(bool)
            # check adjacency (including diagonals)
            has_adj = (
                (arr[:, :-1] & arr[:, 1:]).any() or
                (arr[:-1, :] & arr[1:, :]).any() or
                (arr[:-1, :-1] & arr[1:, 1:]).any() or
                (arr[:-1, 1:] & arr[1:, :-1]).any()
            )
            if not has_adj:
                continue
            # neighbor rule: alive survive 2-3, birth on 3
            padded = np.pad(final, 1, mode='constant', constant_values=0)
            ncount = (
                padded[:-2,:-2] + padded[:-2,1:-1] + padded[:-2,2:] +
                padded[1:-1,:-2] + padded[1:-1,2:] +
                padded[2:  ,:-2] + padded[2:,1:-1] + padded[2:,2:]
            )
            surv = arr & ((ncount >= 2) & (ncount <= 3))
            birth = (~arr) & (ncount == 3)
            if not (surv | birth).any():
                continue
            # ensure pattern changes in next step
            next_frame = simulate(final, steps=1)[-1]
            if np.array_equal(next_frame, final):
                continue
            filtered.append(tuple(final.flatten().tolist()))
    print(f"Filtered {len(filtered)} 'other' patterns out of {len(grids)}")
    if not filtered:
        print("No 'other' patterns found, exiting.")
        return

    # count duplicates
    ctr = Counter(filtered)
    top = ctr.most_common(args.top_k)

    # save CSV of pattern counts
    csv_path = os.path.join(args.out_dir, 'common_other_grids.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern', 'count'])
        for pat, cnt in top:
            writer.writerow([''.join(map(str, pat)), cnt])
    print(f"Saved CSV: {csv_path}")

    # plot patterns
    fig, axs = plt.subplots(1, len(top), figsize=(2*len(top), 2))
    for ax, (pat, cnt) in zip(axs, top):
        arr = np.array(pat).reshape(args.grid_size, args.grid_size)
        ax.imshow(arr, cmap='gray_r')
        ax.set_title(str(cnt))
        ax.axis('off')
    fig.suptitle("Top 'other' entire-grid patterns")
    out_png = os.path.join(args.out_dir, 'common_other_grids.png')
    fig.tight_layout(rect=[0,0,1,0.9])
    fig.savefig(out_png)
    print(f"Saved PNG: {out_png}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Extract and visualize the most common subpatterns from
conditioned generations (quota-trained model).
Scans sliding windows from 3x3 up to 9x9 with stride 1.
"""
import os, sys, argparse, csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# ensure local imports
sys.path.append(os.path.dirname(__file__))
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import detect_period
from utils.gol_simulator import simulate


def main():
    p = argparse.ArgumentParser(description="Common subpatterns")
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='path to quota-trained model checkpoint')
    p.add_argument('--class_label', type=int, default=1,
                   help='conditioning label for still-life class')
    p.add_argument('--period', type=int, default=1,
                   help='pattern period to extract (1=still-life,2=oscillator P2, etc)')
    p.add_argument('--timesteps',   type=int, default=200,
                   help='diffusion timesteps')
    p.add_argument('--num_samples', type=int, default=300,
                   help='number of samples to generate')
    p.add_argument('--grid_size',   type=int, default=32,
                   help='grid height/width')
    p.add_argument('--threshold',   type=float, default=0.5,
                   help='binarization threshold')
    p.add_argument('--min_window',  type=int, default=3,
                   help='minimum sliding window size')
    p.add_argument('--max_window',  type=int, default=None,
                   help='maximum sliding window size (default: grid_size)')
    p.add_argument('--stride',      type=int, default=1,
                   help='sliding window stride')
    p.add_argument('--top_k',       type=int, default=5,
                   help='number of top patterns to display per window')
    p.add_argument('--device',      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',     type=str, default='./common_subpatterns',
                   help='output directory')
    args = p.parse_args()
    # extend max_window to full grid if not set or too large
    if args.max_window is None or args.max_window > args.grid_size:
        args.max_window = args.grid_size

    os.makedirs(args.out_dir, exist_ok=True)

    # load model
    device = args.device
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # diffusion
    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=device,
        schedule='linear',
        guidance_scale=1.0
    )

    # generate conditioned samples
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        c = torch.full((args.num_samples,), args.class_label,
                       device=device, dtype=torch.long)
        print("Sampling conditioned outputs...")
        samples = diffusion.sample(model, shape, c=c)

    # binarize
    grids = (samples.squeeze(1).cpu().numpy() > args.threshold).astype(np.uint8)

    # filter by simulation period and validate Game of Life rules
    filtered = []
    for g in grids:
        hist = simulate(g, steps=args.timesteps)
        per = detect_period(hist)
        if per == args.period and hist[-1].sum() > 0:
            final = hist[-1]
            if args.period == 1:
                # still-life neighbor rule: alive cells have 2 or 3 neighbors; dead cells not exactly 3
                padded = np.pad(final, 1, mode='constant', constant_values=0)
                ncount = (
                    padded[:-2,:-2] + padded[:-2,1:-1] + padded[:-2,2:] +
                    padded[1:-1,:-2] + padded[1:-1,2:] +
                    padded[2:  ,:-2] + padded[2:,1:-1] + padded[2:,2:]
                )
                alive = final.astype(bool)
                if not ((ncount[alive] >= 2) & (ncount[alive] <= 3)).all():
                    continue
                if (ncount[~alive] == 3).any():
                    continue
            elif args.period == 2:
                # verify true period-2 toggle
                next_frame = simulate(final, steps=1)[-1]
                prev = hist[-2] if len(hist) > 1 else None
                if prev is None or not np.array_equal(next_frame, prev):
                    continue
            filtered.append(g)
    print(f"Filtered {len(filtered)} period-{args.period} samples out of {len(grids)}")
    if not filtered:
        print(f"No period-{args.period} patterns found, exiting.")
        return
    patterns = np.stack(filtered)

    # count subpatterns
    from collections import Counter
    counters = {w: Counter() for w in range(args.min_window, args.max_window+1)}
    for g in patterns:
        for w in counters:
            windows = sliding_window_view(g, (w, w))
            for i in range(0, windows.shape[0], args.stride):
                for j in range(0, windows.shape[1], args.stride):
                    patch = windows[i, j]
                    # skip patches without at least two adjacent alive cells (including diagonals)
                    arr = patch.astype(bool)
                    has_neighbors = (
                        (arr[:, :-1] & arr[:, 1:]).any() or  # horizontal
                        (arr[:-1, :] & arr[1:, :]).any() or  # vertical
                        (arr[:-1, :-1] & arr[1:, 1:]).any() or  # diag \ line
                        (arr[:-1, 1:] & arr[1:, :-1]).any()    # diag / line
                    )
                    if not has_neighbors:
                        continue
                    key = tuple(patch.flatten().tolist())
                    counters[w][key] += 1

    # save and plot top patterns
    for w, ctr in counters.items():
        top = ctr.most_common(args.top_k)
        if not top:
            continue
        # CSV
        csv_path = os.path.join(args.out_dir, f"common_{w}x{w}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pattern', 'count'])
            for pat, cnt in top:
                writer.writerow([''.join(map(str, pat)), cnt])
        # plot
        fig, axs = plt.subplots(1, len(top), figsize=(2*len(top), 2))
        for ax, (pat, cnt) in zip(axs, top):
            arr = np.array(pat).reshape(w, w)
            ax.imshow(arr, cmap='gray_r')
            ax.set_title(str(cnt))
            ax.axis('off')
        fig.suptitle(f"Top-{args.top_k} patterns {w}x{w}")
        out_png = os.path.join(args.out_dir, f"common_{w}x{w}.png")
        fig.tight_layout(rect=[0,0,1,0.9])
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved patterns for window {w}x{w} to {csv_path}, {out_png}")


if __name__ == '__main__':
    main()

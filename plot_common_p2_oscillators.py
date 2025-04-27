#!/usr/bin/env python3
"""
Extract and visualize the most common period-2 oscillator subpatterns from
conditioned (alive) generations by the quota-trained model.
Scans sliding windows from min_window up to max_window (default: grid_size).
Skips patches without adjacent alive cells (including diagonals).
"""
import os, sys, argparse, csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from collections import Counter

# allow local imports
sys.path.append(os.path.dirname(__file__))
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import detect_period
from utils.gol_simulator import simulate

def main():
    p = argparse.ArgumentParser(description="Top P2 oscillator subpatterns")
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='path to quota-trained model checkpoint')
    p.add_argument('--class_label', type=int, default=1,
                   help='conditioning label for alive class')
    p.add_argument('--period',      type=int, default=2,
                   help='pattern period to extract (default: 2)')
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
    p.add_argument('--device',      type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='compute device')
    p.add_argument('--out_dir',     type=str, default='./common_p2_oscillators',
                   help='output directory')
    args = p.parse_args()

    # adjust max_window
    if args.max_window is None or args.max_window > args.grid_size:
        args.max_window = args.grid_size

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

    # generate conditioned alive samples
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        c = torch.full((args.num_samples,), args.class_label,
                       device=device, dtype=torch.long)
        print("Sampling conditioned alive outputs...")
        samples = diffusion.sample(model, shape, c=c)

    # binarize
    grids = (samples.squeeze(1).cpu().numpy() > args.threshold).astype(np.uint8)

    # filter by simulation period
    filtered = []
    for g in grids:
        hist = simulate(g, steps=args.timesteps)
        per = detect_period(hist)
        if per == args.period and hist[-1].sum() > 0:
            filtered.append(g)
    print(f"Filtered {len(filtered)} period-{args.period} patterns out of {len(grids)}")
    if not filtered:
        print(f"No period-{args.period} patterns found, exiting.")
        return
    patterns = np.stack(filtered)

    # count subpatterns
    counters = {w: Counter() for w in range(args.min_window, args.max_window+1)}
    for g in patterns:
        for w, ctr in counters.items():
            windows = sliding_window_view(g, (w, w))
            for i in range(0, windows.shape[0], args.stride):
                for j in range(0, windows.shape[1], args.stride):
                    patch = windows[i, j]
                    # require adjacent alive neighbors
                    arr = patch.astype(bool)
                    has_neighbors = (
                        (arr[:, :-1] & arr[:, 1:]).any() or
                        (arr[:-1, :] & arr[1:, :]).any() or
                        (arr[:-1, :-1] & arr[1:, 1:]).any() or
                        (arr[:-1, 1:] & arr[1:, :-1]).any()
                    )
                    if not has_neighbors:
                        continue
                    key = tuple(patch.flatten().tolist())
                    ctr[key] += 1

    # save and plot top patterns
    for w, ctr in counters.items():
        top = ctr.most_common(args.top_k)
        if not top:
            continue
        # CSV
        csv_path = os.path.join(args.out_dir, f"common_p2_{w}x{w}.csv")
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
        fig.suptitle(f"Top-{args.top_k} period-{args.period} patterns ({w}x{w})")
        out_png = os.path.join(args.out_dir, f"common_p2_{w}x{w}.png")
        fig.tight_layout(rect=[0,0,1,0.9])
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved period-{args.period} window {w}x{w}: {csv_path}, {out_png}")

if __name__ == '__main__':
    main()

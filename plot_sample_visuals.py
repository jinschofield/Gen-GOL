#!/usr/bin/env python3
"""
Generate sample visuals for still-life, period-2 oscillators, and other life patterns:
- 100 static PNGs of still-life final frames
- 100 side-by-side PNGs of the last two frames for period-2 oscillators
- 100 animated GIFs of 'other' patterns over their history
"""
import os, sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

# ensure local imports
sys.path.append(os.path.dirname(__file__))
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import detect_period
from utils.gol_simulator import simulate


def classify_pattern(hist):
    final = hist[-1]
    per = detect_period(hist)
    if per == 1:
        return 'still-life'
    elif per == 2:
        return 'oscillator_p2'
    else:
        return 'other'


def save_png(grid, path):
    plt.figure(figsize=(2,2))
    plt.imshow(grid, cmap='gray_r')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_side_by_side(frames, path):
    fig, axs = plt.subplots(1,2,figsize=(4,2))
    for ax, grid in zip(axs, frames):
        ax.imshow(grid, cmap='gray_r')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_gif(frames, path, duration=0.5):
    imgs = [(frame*255).astype(np.uint8) for frame in frames]
    imageio.mimsave(path, imgs, format='GIF', duration=duration)


def main():
    p = argparse.ArgumentParser(description="Sample visuals for GoL diffusion outputs")
    p.add_argument('--checkpoint', type=str, required=True,
                   help='trained model checkpoint')
    p.add_argument('--timesteps',  type=int, default=200,
                   help='diffusion timesteps')
    p.add_argument('--num_samples', type=int, default=1000,
                   help='number of samples to generate')
    p.add_argument('--threshold',  type=float, default=0.5,
                   help='binarization threshold')
    p.add_argument('--grid_size',  type=int, default=32,
                   help='grid height/width')
    p.add_argument('--out_dir',    type=str, default='./sample_visuals',
                   help='output directory')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for cat in ['still-life','oscillator_p2','other']:
        os.makedirs(os.path.join(args.out_dir, cat), exist_ok=True)

    # load model and diffusion
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    diffusion = Diffusion(timesteps=args.timesteps, device=device)

    # sample
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        x = diffusion.sample(model, shape, c=None)
    grids = (x.squeeze(1).cpu().numpy() > args.threshold).astype(np.uint8)

    # simulate and classify
    classes = {'still-life': [], 'oscillator_p2': [], 'other': []}
    histories = []
    for g in grids:
        hist = simulate(g, steps=args.timesteps)
        cat = classify_pattern(hist)
        if len(classes[cat]) < 100:
            classes[cat].append(hist)
        if all(len(classes[c]) >= 100 for c in classes):
            break

    # warn if any short
    for c, lst in classes.items():
        if len(lst) < 100:
            print(f"Warning: only found {len(lst)} samples for {c}")

    # save outputs
    # still-life
    for i, hist in enumerate(classes['still-life'][:100]):
        save_png(hist[-1], os.path.join(args.out_dir, 'still-life', f"{i:03d}.png"))
    # oscillators P2
    for i, hist in enumerate(classes['oscillator_p2'][:100]):
        save_side_by_side([hist[-2], hist[-1]],
                          os.path.join(args.out_dir, 'oscillator_p2', f"{i:03d}.png"))
    # other life as GIF
    for i, hist in enumerate(classes['other'][:100]):
        save_gif([ (frame>0).astype(np.uint8) for frame in hist ],
                 os.path.join(args.out_dir, 'other', f"{i:03d}.gif"))

    print("Saved sample visuals in", args.out_dir)

if __name__ == '__main__':
    main()

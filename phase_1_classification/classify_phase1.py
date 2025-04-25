#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip, clips_array
from PIL import Image
import csv
import math

# allow imports from the parent Gen-GOL repo
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(script_dir, 'utils'))

from models.unet import UNet
from models.diffusion import Diffusion
from utils.gol_simulator import simulate
import detectors


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Generate and classify GoL samples')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to finished_models.pt/model_final.pt')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=16)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'])
    parser.add_argument('--sample_method', type=str, default='ancestral', choices=['ancestral','ddim'])
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(args.device)
    state = torch.load(args.model_checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    diffusion = Diffusion(timesteps=args.timesteps,
                         device=args.device,
                         schedule=args.schedule)

    # sample
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    if args.sample_method == 'ancestral':
        samples = diffusion.sample(model, shape, c=None)
    else:
        samples = diffusion.ddim_sample(model, shape, eta=args.eta, c=None)
    samples = torch.clamp(samples, 0.0, 1.0)
    bin_samples = (samples > args.threshold).float().cpu().numpy()

    # counters
    counts = {'still_life': 0, 'oscillator': 0,
              'glider': 0, 'spaceship': 0,
              'others': 0, 'died_out': 0}
    gifs_dir = os.path.join(args.out_dir, 'gifs')
    os.makedirs(gifs_dir, exist_ok=True)
    # collect clips for grid
    clips = []

    # classify each sample
    for idx in range(bin_samples.shape[0]):
        grid = bin_samples[idx, 0]
        history = simulate(grid, steps=args.timesteps)
        # check final state
        if history[-1].sum() == 0:
            counts['died_out'] += 1
        else:
            # detect period
            period = None
            first = history[0]
            for t in range(1, len(history)):
                if np.array_equal(history[t], first):
                    period = t
                    break
            if period == 1:
                counts['still_life'] += 1
            elif period and period > 1:
                counts['oscillator'] += 1
            else:
                counts['others'] += 1
        # detect glider/spaceship
        if detectors.detect_gliders(grid):
            counts['glider'] += 1
        if detectors.detect_spaceships(grid):
            counts['spaceship'] += 1
        # build clip for this sample
        frames = []
        for frame in history:
            gray = (frame * 255).astype(np.uint8)
            rgb = np.stack([gray] * 3, axis=2)
            frames.append(rgb)
        clip = ImageSequenceClip(frames, fps=5)
        clips.append(clip)
        # save gif if alive
        if history[-1].sum() > 0:
            gif_path = os.path.join(gifs_dir, f'sample_{idx}.gif')
            clip.write_gif(gif_path, program='imageio')

    # save summary CSV
    csv_path = os.path.join(args.out_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'count'])
        for cat, cnt in counts.items():
            writer.writerow([cat, cnt])

    # assemble all clips into a grid GIF
    num_clips = len(clips)
    rows = int(math.sqrt(num_clips)) or 1
    cols = math.ceil(num_clips / rows)
    w, h = clips[0].size
    fps = clips[0].fps
    duration = clips[0].duration
    num_frames = int(duration * fps)
    blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
    blank_clip = ImageSequenceClip([blank_frame] * num_frames, fps=fps)
    grid_clips = []
    for i in range(rows):
        row_clips = []
        for j in range(cols):
            idx2 = i * cols + j
            row_clips.append(clips[idx2] if idx2 < num_clips else blank_clip)
        grid_clips.append(row_clips)
    grid = clips_array(grid_clips)
    grid_path = os.path.join(args.out_dir, 'grid.gif')
    grid.write_gif(grid_path, program='imageio')
    print(f'Grid GIF saved to {grid_path}')

    print('Classification complete:')
    for cat, cnt in counts.items():
        print(f'{cat}: {cnt}')
    print(f'GIFs and summary saved in {args.out_dir}')


if __name__ == '__main__':
    main()

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

    # classification across conditions: unconditioned, conditioned alive, conditioned dead
    conditions = [(None, 'unconditioned'), (1, 'conditioned_alive'), (0, 'conditioned_dead')]
    all_results = {}
    for c, label in conditions:
        # tag directory and filenames with grid size and no type-of-life conditioning
        cond_tag = f"{args.grid_size}_{label}_notypecond"
        cond_dir = os.path.join(args.out_dir, cond_tag)
        os.makedirs(cond_dir, exist_ok=True)
        gifs_dir = os.path.join(cond_dir, 'gifs')
        os.makedirs(gifs_dir, exist_ok=True)
        # sample under condition c
        shape = (args.num_samples, 1, args.grid_size, args.grid_size)
        if args.sample_method == 'ancestral':
            samples = diffusion.sample(model, shape, c=c)
        else:
            samples = diffusion.ddim_sample(model, shape, eta=args.eta, c=c)
        samples = torch.clamp(samples, 0.0, 1.0)
        bin_samples = (samples > args.threshold).float().cpu().numpy()
        # classify
        counts = {'still_life': 0, 'oscillator': 0,
                  'glider': 0, 'spaceship': 0,
                  'others': 0, 'died_out': 0}
        clips = []
        for idx in range(bin_samples.shape[0]):
            grid = bin_samples[idx, 0]
            history = simulate(grid, steps=args.timesteps)
            # count
            if history[-1].sum() == 0:
                counts['died_out'] += 1
            else:
                per = None
                first = history[0]
                for t in range(1, len(history)):
                    if np.array_equal(history[t], first):
                        per = t
                        break
                if per == 1:
                    counts['still_life'] += 1
                elif per and per > 1:
                    counts['oscillator'] += 1
                else:
                    counts['others'] += 1
            if detectors.detect_gliders(grid):
                counts['glider'] += 1
            if detectors.detect_spaceships(grid):
                counts['spaceship'] += 1
            # build clip
            frames = [np.stack([((f*255).astype(np.uint8))]*3, axis=2) for f in history]
            clip = ImageSequenceClip(frames, fps=5)
            clips.append(clip)
            if history[-1].sum() > 0:
                clip.write_gif(os.path.join(gifs_dir, f'sample_{idx}.gif'), program='imageio')
        # summary CSV
        summary_path = os.path.join(cond_dir, f'summary_{label}_sz{args.grid_size}_notypecond.csv')
        with open(summary_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['category', 'count'])
            for k, v in counts.items():
                w.writerow([k, v])
        # grid GIF
        num = len(clips)
        rows = int(math.sqrt(num)) or 1
        cols = math.ceil(num / rows)
        w_img, h_img = clips[0].size
        fps = clips[0].fps
        dur = clips[0].duration
        nf = int(dur * fps)
        blank = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        blank_clip = ImageSequenceClip([blank]*nf, fps=fps)
        grid_clips = [[clips[i*cols+j] if i*cols+j < num else blank_clip for j in range(cols)] for i in range(rows)]
        grid_gif = clips_array(grid_clips)
        grid_gif.write_gif(os.path.join(cond_dir, f'grid_{label}_sz{args.grid_size}_notypecond.gif'), program='imageio')
        all_results[label] = counts
    # compute differences from unconditioned
    cats = set().union(*(r.keys() for r in all_results.values()))
    diff_path = os.path.join(args.out_dir, f'diff_counts_sz{args.grid_size}_notypecond.csv')
    with open(diff_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['category'] + list(all_results.keys()) + ['diff_alive', 'diff_dead']
        w.writerow(header)
        for cat in cats:
            row = [cat] + [all_results[l].get(cat, 0) for l in all_results]
            row.append(all_results['conditioned_alive'].get(cat, 0) - all_results['unconditioned'].get(cat, 0))
            row.append(all_results['conditioned_dead'].get(cat, 0) - all_results['unconditioned'].get(cat, 0))
            w.writerow(row)
    print('Completed classification under all conditions. Diffs in', diff_path)


if __name__ == '__main__':
    main()

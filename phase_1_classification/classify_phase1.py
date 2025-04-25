#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip, clips_array
from PIL import Image, ImageDraw, ImageFont
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
    parser.add_argument('--timesteps', type=int, default=200,
                        help='Number of GoL steps per sample')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'])
    parser.add_argument('--sample_method', type=str, default='ancestral', choices=['ancestral','ddim'])
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--scale', type=int, default=20,
                        help='Upscaling factor for frames (pixels per cell)')
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
        # sample under condition c
        shape = (args.num_samples, 1, args.grid_size, args.grid_size)
        # prepare conditioning tensor for model embedding
        if c is None:
            c_tensor = None
        else:
            c_tensor = torch.full((args.num_samples,), c, device=args.device, dtype=torch.long)
        if args.sample_method == 'ancestral':
            samples = diffusion.sample(model, shape, c=c_tensor)
        else:
            samples = diffusion.ddim_sample(model, shape, eta=args.eta, c=c_tensor)
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
                # detect period of final state
                last = history[-1]
                for p in range(1, len(history)):
                    if np.array_equal(history[-1-p], last):
                        per = p
                        break
                if per == 1:
                    counts['still_life'] += 1
                elif per and per > 1:
                    counts['oscillator'] += 1
                    # track specific period counts
                    period_key = f'oscillator_period_{per}'
                    if period_key not in counts:
                        counts[period_key] = 0
                    counts[period_key] += 1
                else:
                    counts['others'] += 1
            if detectors.detect_gliders(grid):
                counts['glider'] += 1
            if detectors.detect_spaceships(grid):
                counts['spaceship'] += 1
            # determine classification label for overlay
            if history[-1].sum() == 0:
                label_str = 'died_out'
            else:
                per2 = None
                # detect label period based on final state
                last2 = history[-1]
                for p2 in range(1, len(history)):
                    if np.array_equal(history[-1-p2], last2):
                        per2 = p2
                        break
                if per2 == 1:
                    label_str = 'still_life'
                elif per2 and per2 > 1:
                    label_str = f'oscillator_period_{per2}'
                else:
                    label_str = 'others'
            if detectors.detect_gliders(grid):
                label_str = 'glider'
            elif detectors.detect_spaceships(grid):
                label_str = 'spaceship'
            # build labeled clip
            frames = [np.stack([((f*255).astype(np.uint8))]*3, axis=2) for f in history]
            # overlay text label onto each frame via PIL
            pil_frames = []
            # scale up frames for visibility
            scale = args.scale
            # choose larger font for labels
            font_size = int(scale * 1.5)
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', font_size)
            except Exception:
                font = ImageFont.load_default()
            # border width between grid cells
            border_width = max(2, scale // 10)
            for f_rgb in frames:
                img = Image.fromarray(f_rgb)
                img = img.resize((img.width*scale, img.height*scale), Image.NEAREST)
                draw = ImageDraw.Draw(img)
                # compute text width and height via textbbox
                bbox = draw.textbbox((0, 0), label_str, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x = (img.width - text_w) // 2
                y = img.height - text_h - 2
                draw.rectangle([(x-1, y-1), (x+text_w+1, y+text_h+1)], fill=(0,0,0))
                draw.text((x, y), label_str, fill=(255,255,255), font=font)
                # draw border around frame
                draw.rectangle([(0, 0), (img.width-1, img.height-1)], outline=(0,0,0), width=border_width)
                pil_frames.append(np.array(img))
            clip = ImageSequenceClip(pil_frames, fps=5)
            clips.append(clip)
        # add aggregated life category (all samples except died_out)
        counts['life'] = args.num_samples - counts.get('died_out', 0)
        # summary CSV
        summary_path = os.path.join(cond_dir, f'summary_{label}_sz{args.grid_size}_notypecond.csv')
        with open(summary_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['category', 'count'])
            for k, v in counts.items():
                w.writerow([k, v])
        # print counts to console including period-specific oscillators
        print(f"\n{label} counts:")
        for cat in sorted(counts.keys()):
            print(f"{cat}: {counts[cat]}")
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
        # assemble grid of bordered clips
        grid_gif = clips_array(grid_clips)
        grid_gif.write_gif(os.path.join(cond_dir, f'grid_{label}_sz{args.grid_size}_notypecond.gif'), program='imageio')
        all_results[label] = counts
    # compute differences from unconditioned, include life category
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

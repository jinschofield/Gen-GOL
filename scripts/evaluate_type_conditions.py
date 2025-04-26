#!/usr/bin/env python3
"""
Evaluate distribution across death, still life, 2-period oscillators, others
for dataset, unconditioned, conditioned-live, conditioned-dead.
Outputs CSV and bar plots in a single figure.
"""
import os, sys, glob, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# add repo to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from models.unet import UNet
from models.diffusion import Diffusion
from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid

cats = ['died_out', 'still_life', 'oscillator_period_2', 'others']

# map raw classify_grid output to our four buckets
def map_cat(raw):
    return raw if raw in cats else 'others'


def load_dataset_props(data_dir, timesteps):
    files = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
    cnt = {c: 0 for c in cats}
    for p in files:
        arr = np.load(p).astype(np.uint8)
        key = map_cat(classify_grid(arr, timesteps))
        cnt[key] += 1
    total = sum(cnt.values())
    return {c: cnt[c] / total * 100.0 for c in cats}


def sample_props(model, diff, device, num, timesteps, condition):
    # condition: None, 1 for live, 0 for dead
    if condition is None:
        c = None
    else:
        c = torch.full((num,), int(condition), dtype=torch.long, device=device)
    with torch.no_grad():
        x = diff.sample(model, shape=(num,1,32,32), c=c)
    arrs = (x.cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
    cnt = {c: 0 for c in cats}
    for arr in arrs:
        key = map_cat(classify_grid(arr, timesteps))
        cnt[key] += 1
    return {c: cnt[c] / num * 100.0 for c in cats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='model checkpoint')
    p.add_argument('--data_dir', default='data', help='directory of .npy files')
    p.add_argument('--output_csv', default='type_comparison.csv')
    p.add_argument('--output_fig', default='type_comparison.png')
    p.add_argument('--num_samples', type=int, default=1000)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--schedule', default='cosine')
    p.add_argument('--guidance_scale', type=float, default=1.0)
    args = p.parse_args()

    # dataset
    ds = load_dataset_props(args.data_dir, args.timesteps)

    # model
    model = UNet(dropout=0.0, num_classes=2).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    diff = Diffusion(timesteps=args.timesteps,
                     schedule=args.schedule,
                     guidance_scale=args.guidance_scale,
                     device=args.device)

    # samples
    uncond = sample_props(model, diff, args.device, args.num_samples, args.timesteps, None)
    cond_live = sample_props(model, diff, args.device, args.num_samples, args.timesteps, 1)
    cond_dead = sample_props(model, diff, args.device, args.num_samples, args.timesteps, 0)

    # assemble rows
    scenarios = ['dataset', 'unconditioned', 'cond_live', 'cond_dead']
    props = {'dataset': ds, 'unconditioned': uncond, 'cond_live': cond_live, 'cond_dead': cond_dead}

    # write CSV
    header = ['scenario'] + [f"{c}_pct" for c in cats]
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for sc in scenarios:
            row = [sc] + [f"{props[sc][c]:.2f}" for c in cats]
            w.writerow(row)
    print(f"CSV written to {args.output_csv}")

    # plot
    x = np.arange(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10,6))
    for i, c in enumerate(cats):
        vals = [props[sc][c] for sc in scenarios]
        ax.bar(x + (i-1.5)*width, vals, width, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Percentage')
    ax.set_title('Type distribution by condition')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output_fig)
    plt.show()

if __name__ == '__main__':
    main()

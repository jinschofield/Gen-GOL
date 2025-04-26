#!/usr/bin/env python3
"""
Evaluate living vs. dead proportions for dataset and diffusion model (unconditioned & conditional).
Generates a CSV and bar charts comparing dataset distribution, unconditioned sampling,
conditioning on live and conditioning on dead.
"""
import os, sys, glob, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# ensure repo root on path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from models.unet import UNet
from models.diffusion import Diffusion
from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid


def load_dataset_pct(data_dir, timesteps):
    files = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
    cnt = {'dead':0, 'live':0}
    for p in files:
        arr = np.load(p).astype(np.uint8)
        cat = classify_grid(arr, timesteps)
        if cat == 'died_out': cnt['dead'] += 1
        else: cnt['live'] += 1
    tot = cnt['dead'] + cnt['live']
    return {k: cnt[k]/tot*100.0 for k in cnt}


def sample_pct(model, diffusion, device, num, timesteps, condition):
    if condition is None:
        c = None
    else:
        c = torch.full((num,), int(condition), dtype=torch.long, device=device)
    with torch.no_grad():
        x = diffusion.sample(model, shape=(num,1,32,32), c=c)
    arrs = (x.cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
    cnt = {'dead':0, 'live':0}
    for arr in arrs:
        cat = classify_grid(arr, timesteps)
        if cat == 'died_out': cnt['dead'] += 1
        else: cnt['live'] += 1
    return {k: cnt[k]/num*100.0 for k in cnt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_csv', default='living_dead_comparison.csv')
    p.add_argument('--num_samples', type=int, default=1000)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--schedule', default='cosine')
    p.add_argument('--guidance_scale', type=float, default=1.0)
    args = p.parse_args()

    # dataset distribution
    ds = load_dataset_pct(args.data_dir, args.timesteps)

    # load model
    model = UNet(dropout=0.0, num_classes=2).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    diff = Diffusion(timesteps=args.timesteps,
                     schedule=args.schedule,
                     guidance_scale=args.guidance_scale,
                     device=args.device)

    # sample proportions
    uncond = sample_pct(model, diff, args.device, args.num_samples, args.timesteps, None)
    cond_live = sample_pct(model, diff, args.device, args.num_samples, args.timesteps, 1)
    cond_dead = sample_pct(model, diff, args.device, args.num_samples, args.timesteps, 0)

    # compute deltas
    delta_live_live = cond_live['live'] - uncond['live']
    delta_dead_dead = cond_dead['dead'] - uncond['dead']
    pctchg_live_live = delta_live_live / uncond['live'] * 100.0 if uncond['live'] else 0.0
    pctchg_dead_dead = delta_dead_dead / uncond['dead'] * 100.0 if uncond['dead'] else 0.0

    # write CSV
    rows = [
        ('dataset', ds['live'], ds['dead'], 0.0, 0.0),
        ('unconditioned', uncond['live'], uncond['dead'], 0.0, 0.0),
        ('cond_live', cond_live['live'], cond_live['dead'], delta_live_live, pctchg_live_live),
        ('cond_dead', cond_dead['live'], cond_dead['dead'], - (uncond['live'] - cond_dead['live']), pctchg_dead_dead)
    ]
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario','live_pct','dead_pct','delta_pct','pct_change_pct'])
        for r in rows:
            w.writerow(r)

    # plot
    scenarios = [r[0] for r in rows]
    live_p = [r[1] for r in rows]
    dead_p = [r[2] for r in rows]
    delta_p = [r[3] for r in rows]

    x = np.arange(len(scenarios))
    width = 0.35
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(8,10))
    # proportions
    ax1.bar(x - width/2, live_p, width, label='live')
    ax1.bar(x + width/2, dead_p, width, label='dead')
    ax1.set_xticks(x); ax1.set_xticklabels(scenarios)
    ax1.set_title('Proportions')
    ax1.legend()

    # absolute change
    ax2.bar(x, delta_p)
    ax2.set_xticks(x); ax2.set_xticklabels(scenarios)
    ax2.set_title('Delta live % from unconditioned')

    plt.tight_layout()
    plt.savefig('living_dead_comparison.png')
    plt.show()


if __name__ == '__main__':
    main()

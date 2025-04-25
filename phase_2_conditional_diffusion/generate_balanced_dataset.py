#!/usr/bin/env python3
"""
Generate a balanced set of random 32×32 Game of Life patterns until a target count per category is reached.
Outputs .npy files in an output folder and a CSV of [filepath, category].
Categories: died_out, still_life, oscillator_period_2, glider, others
"""
import os, sys, csv, argparse, random
import numpy as np

# ensure imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'utils'))
sys.path.insert(0, os.path.join(repo_root, 'phase_1_classification', 'utils'))

from utils.gol_simulator import simulate
from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid


def main():
    p = argparse.ArgumentParser(description='Generate balanced GoL patterns per category')
    p.add_argument('--output_dir', default='balanced_data_32x32',
                   help='Folder to save generated .npy patterns')
    p.add_argument('--label_csv', default='balanced_labels_32x32.csv',
                   help='CSV file listing [filepath, category]')
    p.add_argument('--target_count', type=int, default=1000,
                   help='Number of patterns per category')
    p.add_argument('--timesteps', type=int, default=200,
                   help='Simulation timesteps for classification')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cats = ['died_out', 'still_life', 'oscillator_period_2', 'glider', 'others']
    counts = {c:0 for c in cats}
    saved = []
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating up to {args.target_count} patterns for each category...")
    while min(counts.values()) < args.target_count:
        # generate random binary grid
        arr = (np.random.rand(32,32) > 0.5).astype(np.uint8)
        cat = classify_grid(arr, timesteps=args.timesteps)
        # remap any unexpected to 'others'
        if cat not in cats:
            cat = 'others'
        if counts[cat] < args.target_count:
            idx = counts[cat]
            fname = f"{cat}_{idx:04d}.npy"
            fpath = os.path.join(args.output_dir, fname)
            np.save(fpath, arr)
            saved.append((fpath, cat))
            counts[cat] += 1
            print(f" {cat}: {counts[cat]}/{args.target_count}", end='\r')

    # write labels CSV
    with open(args.label_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filepath','category'])
        for pth, c in saved:
            w.writerow([pth, c])

    print("\nDone generating balanced dataset.")
    print("Final counts:")
    for c in cats:
        print(f"  {c}: {counts[c]}")


if __name__ == '__main__':
    main()

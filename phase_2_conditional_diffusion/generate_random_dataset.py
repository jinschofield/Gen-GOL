#!/usr/bin/env python3
"""
Generate a dataset of 32×32 GoL patterns by random sampling.
Collect exactly N samples (no per-category quotas).
Cells start alive with given threshold probability.
"""
import os, sys, csv, argparse, random, time
import numpy as np

# ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'utils'))
sys.path.insert(0, os.path.join(repo_root, 'phase_2_conditional_diffusion'))

from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid


def main():
    p = argparse.ArgumentParser(description='Random GoL dataset generator')
    p.add_argument('--output_dir', type=str, default='random_data_32x32',
                   help='Folder to save .npy patterns')
    p.add_argument('--label_csv', type=str, default='random_labels_32x32.csv',
                   help='CSV file listing [filepath, category]')
    p.add_argument('--num_samples', type=int, default=4000,
                   help='Total number of samples to generate')
    p.add_argument('--threshold', type=float, default=0.2,
                   help='Probability a cell starts alive')
    p.add_argument('--timesteps', type=int, default=200,
                   help='Simulation timesteps for classification')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    saved = []
    print(f"Generating {args.num_samples} samples at threshold={args.threshold}")

    for i in range(args.num_samples):
        arr = (np.random.rand(32,32) < args.threshold).astype(np.uint8)
        cat = classify_grid(arr, timesteps=args.timesteps)
        fname = f"{i:06d}_{cat}.npy"
        fpath = os.path.join(args.output_dir, fname)
        np.save(fpath, arr)
        saved.append((fpath, cat))
        if (i+1) % 100 == 0 or (i+1) == args.num_samples:
            print(f"\rCompleted {i+1}/{args.num_samples}", end='', flush=True)
    print("\nSampling complete.")

    with open(args.label_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'category'])
        for pth, cat in saved:
            writer.writerow([pth, cat])
    print(f"Saved {len(saved)} entries to {args.label_csv}")


if __name__ == '__main__':
    main()

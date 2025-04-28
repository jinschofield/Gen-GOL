#!/usr/bin/env python3
"""
Generate a dataset of 32×32 GoL patterns by natural random sampling.
Collect exactly N samples for each of: died_out, still_life, oscillator_period_2, others.
"""
import os, sys, csv, argparse, random, time
import numpy as np

# ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'utils'))
sys.path.insert(0, os.path.join(repo_root, 'phase_1_classification', 'utils'))

from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid


def main():
    p = argparse.ArgumentParser(description='Natural balanced GoL dataset')
    p.add_argument('--output_dir', type=str, default='quota_data_32x32',
                   help='Folder to save .npy patterns')
    p.add_argument('--label_csv', type=str, default='quota_labels_32x32.csv',
                   help='CSV file listing [filepath, category]')
    p.add_argument('--target_count', type=int, default=1000,
                   help='Samples per category for died, still life, oscillator')
    p.add_argument('--threshold', type=float, default=0.2,
                   help='Probability a cell starts alive')
    p.add_argument('--timesteps', type=int, default=200,
                   help='Simulation timesteps for classification')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # include other life-forms beyond period-2 oscillators and still lifes
    cats = ['died_out', 'still_life', 'oscillator_period_2', 'others']
    counts = {c: 0 for c in cats}
    saved = []
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Sampling up to {args.target_count} per category with threshold={args.threshold}")
    while any(counts[c] < args.target_count for c in cats):
        # generate random initial grid
        arr = (np.random.rand(32,32) < args.threshold).astype(np.uint8)
        raw_cat = classify_grid(arr, timesteps=args.timesteps)
        # group non-targets into 'others'
        cat = raw_cat if raw_cat in ['died_out', 'still_life', 'oscillator_period_2'] else 'others'
        if counts[cat] < args.target_count:
            idx = counts[cat]
            fname = f"{cat}_{idx:04d}.npy"
            fpath = os.path.join(args.output_dir, fname)
            np.save(fpath, arr)
            saved.append((fpath, cat))
            counts[cat] += 1
            # progress print
            status = " | ".join([f"{c}:{counts[c]}/{args.target_count}" for c in cats])
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\r[{ts}] {status}", end='', flush=True)
    print("\nDone sampling.")

    # write labels CSV
    with open(args.label_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filepath', 'category'])
        for pth, cat in saved:
            w.writerow([pth, cat])
    print(f"Saved {len(saved)} samples listed in {args.label_csv}")


if __name__ == '__main__':
    main()

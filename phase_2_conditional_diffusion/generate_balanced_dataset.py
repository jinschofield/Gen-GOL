#!/usr/bin/env python3
"""
Generate a balanced set of random 32×32 Game of Life patterns until a target count per category is reached.
Outputs .npy files in an output folder and a CSV of [filepath, category].
Categories: died_out, still_life, oscillator_period_2, glider, others
"""
import os, sys, csv, argparse, random
import numpy as np
import time

# ensure imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'utils'))
sys.path.insert(0, os.path.join(repo_root, 'phase_1_classification', 'utils'))

from utils.gol_simulator import simulate
from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid
from phase_1_classification.utils.detectors import GLIDER_OFFSETS


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
    p.add_argument('--alive_prob', type=float, default=0.1,
                   help='Probability a cell is alive in random noise')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cats = ['died_out', 'still_life', 'oscillator_period_2', 'glider', 'others']
    counts = {c:0 for c in cats}
    saved = []
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating exactly {args.target_count} patterns per category via targeted sampling...")
    # helper to place a shape defined by offsets
    def place_offsets(arr, offsets):
        max_i = max(di for di, _ in offsets)
        max_j = max(dj for _, dj in offsets)
        i0 = random.randint(0, arr.shape[0]-1-max_i)
        j0 = random.randint(0, arr.shape[1]-1-max_j)
        for di, dj in offsets:
            arr[i0+di, j0+dj] = 1
        return arr

    # simple shape definitions
    STILL_LIFE_OFFSETS = [[(0,0),(0,1),(1,0),(1,1)]]  # block
    OSC2_OFFSETS = [[(0,1),(1,1),(2,1)], [(1,0),(1,1),(1,2)]]  # blinkers

    for cat in cats:
        count = 0
        while count < args.target_count:
            arr = np.zeros((32,32), dtype=np.uint8)
            if cat == 'died_out':
                i, j = random.randint(0,31), random.randint(0,31)
                arr[i, j] = 1
            elif cat == 'still_life':
                offsets = random.choice(STILL_LIFE_OFFSETS)
                place_offsets(arr, offsets)
            elif cat == 'oscillator_period_2':
                offsets = random.choice(OSC2_OFFSETS)
                place_offsets(arr, offsets)
            elif cat == 'glider':
                offsets = random.choice(GLIDER_OFFSETS)
                place_offsets(arr, offsets)
            else:
                # random noise with lower alive-probability
                arr = (np.random.rand(32,32) < args.alive_prob).astype(np.uint8)
            # verify classification matches
            cls = classify_grid(arr, timesteps=args.timesteps)
            cls = cls if cls in cats else 'others'
            if cls != cat:
                continue
            # save valid pattern
            fname = f"{cat}_{count:04d}.npy"
            fpath = os.path.join(args.output_dir, fname)
            np.save(fpath, arr)
            saved.append((fpath, cat))
            counts[cat] += 1
            count += 1
            # progress print per category
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\r[{timestamp}] {cat}: {counts[cat]}/{args.target_count}", end='', flush=True)

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

#!/usr/bin/env python3
"""
Compare category distributions between training dataset and generated samples.
Outputs CSV with dataset pct, generated pct, and percent change per category.
"""
import os
import sys
import argparse
import glob
import csv
import numpy as np

# ensure repo root in path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from utils.gol_simulator import simulate
from phase_1_classification.utils.detectors import detect_gliders, detect_spaceships

def classify_dataset(data_dir, timesteps):
    files = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
    total = len(files)
    counts = {}
    for p in files:
        arr = np.load(p).astype(np.uint8)
        history = simulate(arr, steps=timesteps)
        # death
        if history[-1].sum() == 0:
            cat = 'died_out'
        else:
            # detect period
            per = None
            first = history[0]
            for t in range(1, len(history)):
                if np.array_equal(history[t], first):
                    per = t
                    break
            if per == 1:
                cat = 'still_life'
            elif per and per > 1:
                cat = f'oscillator_period_{per}'
                # also general oscillator
                counts['oscillator'] = counts.get('oscillator', 0) + 1
            else:
                cat = 'others'
        counts[cat] = counts.get(cat, 0) + 1
        # glider/spaceship override
        grid = history[0]
        if detect_gliders(grid):
            counts['glider'] = counts.get('glider', 0) + 1
        if detect_spaceships(grid):
            counts['spaceship'] = counts.get('spaceship', 0) + 1
    # handle general oscillator count for specific periods
    # above loop increments 'oscillator' once per period>1
    return counts, total


def load_generated(summary_csv):
    gen_counts = {}
    with open(summary_csv) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            k, v = row[0], int(row[1])
            gen_counts[k] = v
    gen_total = sum(gen_counts.values())
    return gen_counts, gen_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Path to dataset .npy files')
    p.add_argument('--gen_summary', required=True, help='Path to generated summary CSV')
    p.add_argument('--timesteps', type=int, default=300)
    p.add_argument('--output_csv', default='compare_distribution.csv')
    args = p.parse_args()

    ds_counts, ds_tot = classify_dataset(args.data_dir, args.timesteps)
    gen_counts, gen_tot = load_generated(args.gen_summary)

    # balanced dataset percentages (50% dead, 50% living)
    ds_counts_live = ds_tot - ds_counts.get('died_out', 0)
    ds_pct = {}
    ds_pct['died_out'] = 50.0
    for k, v in ds_counts.items():
        if k != 'died_out':
            ds_pct[k] = v / ds_counts_live * 50.0
    # generated percentages (actual)
    gen_pct = {k: v/gen_tot*100.0 for k, v in gen_counts.items()}
    # compute living-only percentages (exclude 'died_out')
    ds_tot_live = ds_tot - ds_counts.get('died_out', 0)
    gen_tot_live = gen_tot - gen_counts.get('died_out', 0)
    ds_pct_live = {k: v/ds_tot_live*100.0 for k, v in ds_counts.items() if k != 'died_out'}
    gen_pct_live = {k: v/gen_tot_live*100.0 for k, v in gen_counts.items() if k != 'died_out'}

    categories = sorted(set(ds_pct) | set(gen_pct))
    rows = []
    for cat in categories:
        d = ds_pct.get(cat, 0.0)
        g = gen_pct.get(cat, 0.0)
        ch = g - d
        rows.append((cat, d, g, ch))

    # write CSV
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category','dataset_pct','generated_pct','pct_change'])
        for cat, d, g, ch in rows:
            w.writerow([cat, f"{d:.2f}", f"{g:.2f}", f"{ch:.2f}"])

    # print to console for all categories
    print("Comparison of distributions (including dead):")
    for cat, d, g, ch in rows:
        print(f"{cat}: dataset={d:.2f}%, generated={g:.2f}%, change={ch:.2f}%")
    # print living-only comparison
    print("\nLiving-only distributions (dead excluded):")
    categories_live = sorted(set(ds_pct_live) | set(gen_pct_live))
    for cat in categories_live:
        d = ds_pct_live.get(cat, 0.0)
        g = gen_pct_live.get(cat, 0.0)
        ch = g - d
        print(f"{cat}: dataset={d:.2f}%, generated={g:.2f}%, change={ch:.2f}%")
    # write living-only CSV
    base, ext = os.path.splitext(args.output_csv)
    living_csv = f"{base}_living{ext}"
    with open(living_csv, 'w', newline='') as f2:
        w2 = csv.writer(f2)
        w2.writerow(['category','dataset_pct','generated_pct','pct_change'])
        for cat in categories_live:
            d = ds_pct_live.get(cat, 0.0)
            g = gen_pct_live.get(cat, 0.0)
            ch = g - d
            w2.writerow([cat, f"{d:.2f}", f"{g:.2f}", f"{ch:.2f}"])
    print(f"Living-only comparison saved to {living_csv}")

if __name__ == '__main__':
    main()

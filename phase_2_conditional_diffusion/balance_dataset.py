#!/usr/bin/env python3
"""
Balance labeled Phase 2 dataset to equal proportions per category.
Usage:
  python balance_dataset.py \
    --label_csv phase2_training_labels_32x32.csv \
    --output_csv phase2_training_labels_balanced_32x32.csv \
    [--mode downsample|oversample] [--seed 42] [--target_count 1000]
"""
import os, csv, argparse, random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Balance Phase 2 label CSV to equal category counts")
    parser.add_argument('--label_csv', required=True, help='Input CSV with [filepath, category]')
    parser.add_argument('--output_csv', default='phase2_training_labels_balanced_32x32.csv',
                        help='Output balanced CSV')
    parser.add_argument('--mode', choices=['downsample', 'oversample'], default='downsample',
                        help='(deprecated) unused; use --target_count for exact per-category count')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_count', type=int, default=1000,
                        help='Desired number of samples per category')
    args = parser.parse_args()

    random.seed(args.seed)
    # define desired categories
    target_cats = ['died_out','still_life','oscillator_period_2','glider','others']
    # load and remap labels
    samples = []
    with open(args.label_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for path, cat in reader:
            key = cat if cat in target_cats else 'others'
            samples.append((path, key))
    # initialize groups
    groups = {c: [] for c in target_cats}
    for path, cat in samples:
        groups[cat].append(path)
    # build balanced list: exact target_count per category
    balanced = []
    for cat in target_cats:
        paths = groups[cat]
        if not paths:
            raise RuntimeError(f'No samples found for category {cat}')
        if len(paths) >= args.target_count:
            chosen = random.sample(paths, args.target_count)
        else:
            chosen = random.choices(paths, k=args.target_count)
        balanced.extend((p, cat) for p in chosen)
    random.shuffle(balanced)
    # write CSV
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'category'])
        for p, cat in balanced:
            writer.writerow([p, cat])
    # summary
    print(f"Balanced CSV saved to {args.output_csv}")
    # counts per category
    new_counts = {c: sum(1 for _,cat in balanced if cat == c) for c in target_cats}
    print("Counts per category:")
    for c in target_cats:
        print(f"  {c}: {new_counts[c]}")
    # aggregated
    total = len(balanced)
    living = sum(v for c, v in new_counts.items() if c != 'died_out')
    oscillators = sum(v for c, v in new_counts.items() if c.startswith('oscillator_'))
    print(f"All living: {living}/{total}")
    print(f"All oscillators: {oscillators}/{total}")

if __name__ == '__main__':
    main()

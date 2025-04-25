#!/usr/bin/env python3
"""
Balance labeled Phase 2 dataset to equal proportions per category.
Usage:
  python balance_dataset.py \
    --label_csv phase2_training_labels_32x32.csv \
    --output_csv phase2_training_labels_balanced_32x32.csv \
    [--mode downsample|oversample] [--seed 42]
"""
import os, csv, argparse, random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Balance Phase 2 label CSV to equal category counts")
    parser.add_argument('--label_csv', required=True, help='Input CSV with [filepath, category]')
    parser.add_argument('--output_csv', default='phase2_training_labels_balanced_32x32.csv',
                        help='Output balanced CSV')
    parser.add_argument('--mode', choices=['downsample', 'oversample'], default='downsample',
                        help='Downsample larger groups or oversample smaller groups')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    # load labels
    samples = []
    with open(args.label_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for path, cat in reader:
            samples.append((path, cat))
    # group by category
    groups = defaultdict(list)
    for path, cat in samples:
        groups[cat].append(path)
    cats = sorted(groups.keys())
    counts = {c: len(groups[c]) for c in cats}
    # determine target count
    min_n = min(counts.values())
    max_n = max(counts.values())
    target = min_n if args.mode == 'downsample' else max_n
    # build balanced list
    balanced = []
    for cat in cats:
        paths = groups[cat]
        if args.mode == 'downsample':
            chosen = random.sample(paths, target)
        else:
            chosen = [random.choice(paths) for _ in range(target)]
        for p in chosen:
            balanced.append((p, cat))
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
    new_counts = {c: sum(1 for _,cat in balanced if cat == c) for c in cats}
    print("Counts per category:")
    for c in cats:
        print(f"  {c}: {new_counts[c]}")
    # aggregated
    total = len(balanced)
    living = sum(v for c, v in new_counts.items() if c != 'died_out')
    oscillators = sum(v for c, v in new_counts.items() if c.startswith('oscillator_'))
    print(f"All living: {living}/{total}")
    print(f"All oscillators: {oscillators}/{total}")

if __name__ == '__main__':
    main()

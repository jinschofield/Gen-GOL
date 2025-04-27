#!/usr/bin/env python3
"""
Plot normalized living-category distributions, scaling living-only percentages to 100.
Reads the CSV of living-only percentages (from plot_fig2) and produces a bar chart and a new CSV.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Normalize living-category percentages to sum to 100."
    )
    parser.add_argument(
        '--metrics_csv', type=str,
        default='./figures/figure2_metrics_percentages_living.csv',
        help='input CSV of living-only metrics (category, train_percent, gen_percent)'
    )
    parser.add_argument(
        '--out_csv', type=str,
        default='./figures/figure2_metrics_living_normalized.csv',
        help='output CSV of normalized living metrics'
    )
    parser.add_argument(
        '--out_fig', type=str,
        default='./figures/figure2_living_normalized.png',
        help='output bar chart of normalized living metrics'
    )
    args = parser.parse_args()

    # load data
    rows = []
    with open(args.metrics_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # extract alive reference
    alive = next(r for r in rows if r['category'] == 'Alive')
    alive_train = float(alive['train_percent'])
    alive_gen = float(alive['gen_percent'])
    # filter out Alive row
    filtered = [r for r in rows if r['category'] != 'Alive']
    cats = [r['category'] for r in filtered]
    train_norm = [float(r['train_percent']) / alive_train * 100.0 for r in filtered]
    gen_norm   = [float(r['gen_percent'])   / alive_gen   * 100.0 for r in filtered]

    # write normalized CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'train_percent_norm', 'gen_percent_norm'])
        for cat, tn, gn in zip(cats, train_norm, gen_norm):
            writer.writerow([cat, f"{tn:.1f}", f"{gn:.1f}"])
    print(f"Saved normalized metrics CSV to: {args.out_csv}")

    # plot bar chart
    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - w/2, train_norm, w, label='Train', color='skyblue')
    ax.bar(x + w/2, gen_norm,   w, label='Gen',   color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha='right')
    ax.set_ylabel('Normalized Percentage of Alive (%)')
    ax.set_title('Living-category Distributions (Normalized to 100)')
    for i, (tn, gn) in enumerate(zip(train_norm, gen_norm)):
        ax.text(i - w/2, tn, f"{tn:.1f}%", ha='center', va='bottom')
        ax.text(i + w/2, gn, f"{gn:.1f}%", ha='center', va='bottom')
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.out_fig)
    print(f"Saved normalized living plot to: {args.out_fig}")

if __name__ == '__main__':
    main()

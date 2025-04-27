#!/usr/bin/env python3
"""
Plot Figure 2: compare training dataset vs unconditioned generation metrics.
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# ensure Gen-GOL path for imports
sys.path.append(os.path.dirname(__file__))
from evaluate import load_train_patterns
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples


def main():
    parser = argparse.ArgumentParser(
        description="Figure 2: Training vs Unconditioned Generation"
    )
    parser.add_argument('--data_dir', type=str, default='phase_2_conditional_diffusion/random_data_32x32',
                        help='directory of .npy training patterns')
    parser.add_argument('--checkpoint', type=str, default='finished_models/model_final_random.pt',
                        help='path to trained model checkpoint')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='diffusion timesteps')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='number of samples to generate')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='binarization threshold')
    parser.add_argument('--out_dir', type=str, default='./figures',
                        help='output directory for figure and metrics')
    args = parser.parse_args()
    print(f"Configuration: data_dir={args.data_dir}, checkpoint={args.checkpoint}, timesteps={args.timesteps}, num_samples={args.num_samples}, threshold={args.threshold}, out_dir={args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Created output directory: {args.out_dir}")

    # load train patterns
    print("Loading training patterns...")
    train_patterns = load_train_patterns(args.data_dir)
    print(f"Loaded {len(train_patterns)} training patterns.")
    N_train = len(train_patterns)
    H = train_patterns[0].shape[0]

    # compute training metrics
    # stack into tensor (N,1,H,W)
    train_tensor = torch.tensor(
        np.stack([p.astype(np.float32) for p in train_patterns])[:,None,:,:],
        dtype=torch.float32
    )
    print("Computing training metrics...")
    train_results = evaluate_samples(
        train_tensor, train_patterns,
        max_steps=args.timesteps, threshold=args.threshold
    )

    # load trained model
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(args.device if hasattr(args, 'device') else 'cpu')
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    # diffusion
    diffusion = Diffusion(
        timesteps=args.timesteps, schedule='linear', guidance_scale=1.0
    )

    # sample unconditioned
    shape = (args.num_samples, 1, H, H)
    print(f"Sampling {args.num_samples} unconditioned samples...")
    with torch.no_grad():
        gen = diffusion.sample(model, shape, c=None)
    print("Sampling complete.")
    gen = torch.clamp(gen, 0.0, 1.0)
    print("Computing generated metrics...")
    gen_results = evaluate_samples(
        gen, train_patterns,
        max_steps=args.timesteps, threshold=args.threshold
    )

    # define categories
    dead = 'died_out'
    types = ['still_life', 'oscillator_p2', 'survived_unknown']
    # helper: percentage per category
    def pct(res, key): return res.get(key,0)/res['total']*100.0
    # compute pct for types and dead
    train_pct = {key: pct(train_results, key) for key in types + [dead]}
    gen_pct   = {key: pct(gen_results,   key) for key in types + [dead]}
    # compute alive grouping (sum of types)
    train_pct['alive'] = sum(train_pct[k] for k in types)
    gen_pct['alive']   = sum(gen_pct[k]   for k in types)
    # define plot order
    plot_keys = ['alive'] + types

    # compute absolute counts for five categories
    counts_keys = ['alive'] + types + [dead]
    train_counts = {k: train_results.get(k, 0) for k in counts_keys}
    gen_counts   = {k: gen_results.get(k,   0) for k in counts_keys}
    deltas       = {k: gen_counts[k] - train_counts[k] for k in counts_keys}

    # plot absolute counts
    print("Plotting absolute counts...")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(counts_keys))
    w = 0.35
    ax.bar(x - w/2, [train_counts[k] for k in counts_keys], w, label='Train', color='skyblue')
    ax.bar(x + w/2, [gen_counts[k]   for k in counts_keys], w, label='Gen',   color='salmon')
    labels = ['Alive', 'Still Life', 'Oscillator P2', 'Other', 'Dead']
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count')
    # annotate counts and deltas
    for i, k in enumerate(counts_keys):
        ax.text(i - w/2, train_counts[k], str(train_counts[k]), ha='center', va='bottom')
        ax.text(i + w/2, gen_counts[k],   str(gen_counts[k]),   ha='center', va='bottom')
        ax.text(i + w/2, gen_counts[k], f'Δ {deltas[k]}', ha='center', va='top', color='black')
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, 'figure2_absolute.png')
    fig.savefig(fig_path)
    print(f"Saved figure: {fig_path}")

    # save raw metrics (absolute counts)
    out_csv = os.path.join(args.out_dir, 'figure2_metrics_absolute.csv')
    print(f"Writing metrics CSV to: {out_csv}")
    import csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'train_count', 'gen_count', 'delta_count'])
        for k in counts_keys:
            writer.writerow([k, train_counts[k], gen_counts[k], deltas[k]])
    print(f"Saved metrics CSV: {out_csv}")


if __name__ == '__main__':
    main()

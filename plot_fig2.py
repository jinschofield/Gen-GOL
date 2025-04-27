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

    # compute absolute counts for five key categories
    total_train = train_results['total']
    train_dead = train_results.get('died_out', 0)
    train_alive = total_train - train_dead
    train_sl = train_results.get('still_life', 0)
    train_osc2 = train_results.get('oscillator_p2', 0)
    train_other = train_alive - train_sl - train_osc2

    total_gen = gen_results['total']
    gen_dead = gen_results.get('died_out', 0)
    gen_alive = total_gen - gen_dead
    gen_sl = gen_results.get('still_life', 0)
    gen_osc2 = gen_results.get('oscillator_p2', 0)
    gen_other = gen_alive - gen_sl - gen_osc2

    # prepare labels and ordered counts
    categories = ['Alive', 'Still Life', 'Oscillator P2', 'Other', 'Dead']
    train_counts = [train_alive, train_sl, train_osc2, train_other, train_dead]
    gen_counts   = [gen_alive,   gen_sl,   gen_osc2,   gen_other,   gen_dead]
    deltas       = [g - t for t, g in zip(train_counts, gen_counts)]

    # plot absolute counts
    print("Plotting absolute counts...")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, train_counts, w, label='Train', color='skyblue')
    ax.bar(x + w/2, gen_counts,   w, label='Gen',   color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Count')
    # annotate counts and deltas
    for i, (t, g, d) in enumerate(zip(train_counts, gen_counts, deltas)):
        ax.text(i - w/2, t, str(t), ha='center', va='bottom')
        ax.text(i + w/2, g, str(g), ha='center', va='bottom')
        ax.text(i + w/2, g, f'Δ {d}', ha='center', va='top', color='black')
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
        for name, t, g, d in zip(categories, train_counts, gen_counts, deltas):
            writer.writerow([name, t, g, d])
    print(f"Saved metrics CSV: {out_csv}")


if __name__ == '__main__':
    main()

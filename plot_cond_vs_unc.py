#!/usr/bin/env python3
"""
Compare unconditioned vs conditioned generation distributions for trained model.
Produces percentage distribution bar charts and CSV metrics.
"""
import os, argparse, csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples

def main():
    parser = argparse.ArgumentParser(
        description="Compare unconditioned vs conditioned generation distributions"
    )
    parser.add_argument('--checkpoint', type=str, default='finished_models/model_final_random.pt',
                        help='trained model checkpoint')
    parser.add_argument('--timesteps', type=int, default=200, help='diffusion timesteps')
    parser.add_argument('--num_samples', type=int, default=300, help='number of samples to generate')
    parser.add_argument('--threshold', type=float, default=0.5, help='binarization threshold')
    parser.add_argument('--class_label', type=int, default=1, choices=[0,1],
                        help='conditioning label: 1=survive, 0=died')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'], help='noise schedule')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='CFG scale')
    parser.add_argument('--grid_size', type=int, default=32, help='grid height/width')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default='./figures_cond', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    # diffusion
    diffusion = Diffusion(
        timesteps=args.timesteps, schedule=args.schedule, guidance_scale=args.guidance_scale
    )

    # sampling
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        samples_unc = diffusion.sample(model, shape, c=None)
        c = torch.full((args.num_samples,), args.class_label, device=args.device, dtype=torch.long)
        samples_cond = diffusion.sample(model, shape, c=c)

    # clamp and evaluate
    samples_unc = torch.clamp(samples_unc, 0.0, 1.0)
    samples_cond = torch.clamp(samples_cond, 0.0, 1.0)
    results_unc = evaluate_samples(samples_unc, [], max_steps=args.timesteps, threshold=args.threshold)
    results_cond = evaluate_samples(samples_cond, [], max_steps=args.timesteps, threshold=args.threshold)

    # compute counts
    total_unc = results_unc['total']
    dead_unc = results_unc.get('died_out', 0)
    alive_unc = total_unc - dead_unc
    sl_unc = results_unc.get('still_life', 0)
    osc2_unc = results_unc.get('oscillator_p2', 0)
    other_unc = alive_unc - sl_unc - osc2_unc

    total_cond = results_cond['total']
    dead_cond = results_cond.get('died_out', 0)
    alive_cond = total_cond - dead_cond
    sl_cond = results_cond.get('still_life', 0)
    osc2_cond = results_cond.get('oscillator_p2', 0)
    other_cond = alive_cond - sl_cond - osc2_cond

    categories = ['Alive', 'Still Life', 'Oscillator P2', 'Other', 'Dead']
    counts_unc_list = [alive_unc, sl_unc, osc2_unc, other_unc, dead_unc]
    counts_cond_list = [alive_cond, sl_cond, osc2_cond, other_cond, dead_cond]

    # percentages
    perc_unc = [c/total_unc*100.0 for c in counts_unc_list]
    perc_cond = [c/total_cond*100.0 for c in counts_cond_list]

    # plot overall percentages
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, perc_unc, w, label='Unconditioned', color='skyblue')
    ax.bar(x + w/2, perc_cond, w, label='Conditioned', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    for i, (u, v) in enumerate(zip(perc_unc, perc_cond)):
        ax.text(i - w/2, u, f"{u:.1f}%", ha='center', va='bottom')
        ax.text(i + w/2, v, f"{v:.1f}%", ha='center', va='bottom')
    ax.legend()
    plt.tight_layout()
    out_fig1 = os.path.join(args.out_dir, 'cond_vs_unc_percentages.png')
    fig.savefig(out_fig1)
    print(f"Saved figure: {out_fig1}")

    # save overall metrics
    out_csv1 = os.path.join(args.out_dir, 'cond_vs_unc_metrics_percentages.csv')
    with open(out_csv1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'unc_percent', 'cond_percent'])
        for cat, u, v in zip(categories, perc_unc, perc_cond):
            writer.writerow([cat, f"{u:.1f}", f"{v:.1f}"])
    print(f"Saved metrics CSV: {out_csv1}")

    # living-only percentages
    living_cats = categories[:-1]
    perc_unc_live = perc_unc[:-1]
    perc_cond_live = perc_cond[:-1]
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(living_cats))
    ax.bar(x - w/2, perc_unc_live, w, label='Unconditioned', color='skyblue')
    ax.bar(x + w/2, perc_cond_live, w, label='Conditioned', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(living_cats, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    for i, (u, v) in enumerate(zip(perc_unc_live, perc_cond_live)):
        ax.text(i - w/2, u, f"{u:.1f}%", ha='center', va='bottom')
        ax.text(i + w/2, v, f"{v:.1f}%", ha='center', va='bottom')
    ax.legend()
    plt.tight_layout()
    out_fig2 = os.path.join(args.out_dir, 'cond_vs_unc_percentages_living.png')
    fig.savefig(out_fig2)
    print(f"Saved figure: {out_fig2}")

    # save living-only metrics
    out_csv2 = os.path.join(args.out_dir, 'cond_vs_unc_metrics_percentages_living.csv')
    with open(out_csv2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'unc_percent', 'cond_percent'])
        for cat, u, v in zip(living_cats, perc_unc_live, perc_cond_live):
            writer.writerow([cat, f"{u:.1f}", f"{v:.1f}"])
    print(f"Saved metrics CSV: {out_csv2}")

    # compute and plot delta percentages overall
    delta_percentages = [v - u for u, v in zip(perc_unc, perc_cond)]
    print("Plotting overall delta percentages...")
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(categories))
    ax.bar(x, delta_percentages, color='purple')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Δ Percentage (Cond − Uncond)')
    for i, d in enumerate(delta_percentages):
        va = 'bottom' if d >= 0 else 'top'
        ax.text(i, d, f"{d:.1f}%", ha='center', va=va)
    plt.tight_layout()
    out_fig_delta = os.path.join(args.out_dir, 'cond_vs_unc_delta_percentages.png')
    fig.savefig(out_fig_delta)
    print(f"Saved delta figure: {out_fig_delta}")

    # save delta CSV overall
    out_csv_delta = os.path.join(args.out_dir, 'cond_vs_unc_metrics_delta.csv')
    with open(out_csv_delta, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'delta_percent'])
        for cat, d in zip(categories, delta_percentages):
            writer.writerow([cat, f"{d:.1f}"])
    print(f"Saved delta metrics CSV: {out_csv_delta}")

    # living-only delta percentages
    delta_live = [v - u for u, v in zip(perc_unc_live, perc_cond_live)]
    print("Plotting living-only delta percentages...")
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(living_cats))
    ax.bar(x, delta_live, color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(living_cats, rotation=45, ha='right')
    ax.set_ylabel('Δ Percentage (Cond − Uncond)')
    for i, d in enumerate(delta_live):
        va = 'bottom' if d >= 0 else 'top'
        ax.text(i, d, f"{d:.1f}%", ha='center', va=va)
    plt.tight_layout()
    out_fig_delta_live = os.path.join(args.out_dir, 'cond_vs_unc_delta_percentages_living.png')
    fig.savefig(out_fig_delta_live)
    print(f"Saved living-only delta figure: {out_fig_delta_live}")

    # save living-only delta CSV
    out_csv_delta_live = os.path.join(args.out_dir, 'cond_vs_unc_metrics_delta_living.csv')
    with open(out_csv_delta_live, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'delta_percent'])
        for cat, d in zip(living_cats, delta_live):
            writer.writerow([cat, f"{d:.1f}"])
    print(f"Saved living-only delta metrics CSV: {out_csv_delta_live}")

if __name__ == '__main__':
    main()

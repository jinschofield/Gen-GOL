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
    parser.add_argument('--condition_labels', type=int, nargs='+', default=[0,1],
                        help='conditioning labels (e.g., 0 1 for each label)')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'], help='noise schedule')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='CFG scale')
    parser.add_argument('--grid_size', type=int, default=32, help='grid height/width')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default='./figures_cond', help='output directory')
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='directory of .npy training patterns for novelty check')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    # ensure diffusion on same device as model
    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=args.device,
        schedule=args.schedule,
        guidance_scale=args.guidance_scale
    )

    # sampling
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        print("Sampling unconditioned...")
        samples_unc = diffusion.sample(model, shape, c=None)
        cond_samples = {}
        for cl in args.condition_labels:
            print(f"Sampling conditioned (label={cl})...")
            c_vec = torch.full((args.num_samples,), cl, device=args.device, dtype=torch.long)
            cond_samples[cl] = diffusion.sample(model, shape, c=c_vec)

    # clamp samples
    samples_unc = torch.clamp(samples_unc, 0.0, 1.0)
    for cl in cond_samples:
        cond_samples[cl] = torch.clamp(cond_samples[cl], 0.0, 1.0)

    # load training patterns for novelty
    train_patterns = []
    if args.train_data_dir:
        files = sorted(os.listdir(args.train_data_dir))
        for fname in files:
            if fname.endswith('.npy'):
                train_patterns.append(np.load(os.path.join(args.train_data_dir, fname)))
    print(f"Loaded {len(train_patterns)} training patterns for novelty check")

    # evaluate and compute novelty fractions (rotation/flip)
    results_unc = evaluate_samples(
        samples_unc, train_patterns,
        max_steps=args.timesteps, threshold=args.threshold
    )
    novel_unc = results_unc.get('novel_frac', 0.0)
    results_cond = {}  # init dict for full conditional results
    novel_by_label = {}
    for cl in args.condition_labels:
        res_c = evaluate_samples(
            cond_samples[cl], train_patterns,
            max_steps=args.timesteps, threshold=args.threshold
        )
        novel_by_label[cl] = res_c.get('novel_frac', 0.0)
        # store full evaluation results for this condition
        results_cond[cl] = res_c
    # print novelty rates
    print(f"Novelty fraction — Unconditioned: {novel_unc:.3f}")
    for cl in args.condition_labels:
        print(f"Novelty fraction — Conditioned {cl}: {novel_by_label[cl]:.3f}")
    # save novelty metrics CSV
    out_csv = os.path.join(args.out_dir, 'cond_vs_unc_novelty.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'novel_frac'])
        writer.writerow(['unconditioned', f"{novel_unc:.3f}"])
        for cl in args.condition_labels:
            writer.writerow([f'condition_{cl}', f"{novel_by_label[cl]:.3f}"])
    print(f"Saved novelty metrics CSV: {out_csv}")
    # plot novelty bar chart
    labels = ['Unconditioned'] + [f'Cond {cl}' for cl in args.condition_labels]
    values = [novel_unc * 100] + [novel_by_label[cl] * 100 for cl in args.condition_labels]
    fig, ax = plt.subplots(figsize=(6,4))
    colors = ['skyblue'] + ['salmon'] * len(args.condition_labels)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Novelty (%)')
    ax.set_title('Novelty Rate')
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    out_fig = os.path.join(args.out_dir, 'cond_vs_unc_novelty.png')
    fig.savefig(out_fig)
    print(f"Saved novelty figure: {out_fig}")

    # compute counts
    total_unc = results_unc['total']
    dead_unc = results_unc.get('died_out', 0)
    alive_unc = total_unc - dead_unc
    sl_unc = results_unc.get('still_life', 0)
    osc2_unc = results_unc.get('oscillator_p2', 0)
    other_unc = alive_unc - sl_unc - osc2_unc

    total_cond = {}
    dead_cond = {}
    alive_cond = {}
    sl_cond = {}
    osc2_cond = {}
    other_cond = {}
    for cl in args.condition_labels:
        res_c_full = results_cond[cl]
        total_cond[cl] = res_c_full['total']
        dead_cond[cl] = res_c_full.get('died_out', 0)
        alive_cond[cl] = total_cond[cl] - dead_cond[cl]
        sl_cond[cl] = res_c_full.get('still_life', 0)
        osc2_cond[cl] = res_c_full.get('oscillator_p2', 0)
        other_cond[cl] = alive_cond[cl] - sl_cond[cl] - osc2_cond[cl]

    categories = ['Alive', 'Still Life', 'Oscillator P2', 'Other', 'Dead']
    counts_unc_list = [alive_unc, sl_unc, osc2_unc, other_unc, dead_unc]
    counts_cond_list = {}
    for cl in args.condition_labels:
        counts_cond_list[cl] = [alive_cond[cl], sl_cond[cl], osc2_cond[cl], other_cond[cl], dead_cond[cl]]

    # percentages
    perc_unc = [c/total_unc*100.0 for c in counts_unc_list]
    perc_cond = {}
    for cl in args.condition_labels:
        perc_cond[cl] = [c/total_cond[cl]*100.0 for c in counts_cond_list[cl]]

    # plot overall percentages
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, perc_unc, w, label='Unconditioned', color='skyblue')
    for i, cl in enumerate(args.condition_labels):
        ax.bar(x + w/2 + i*w, perc_cond[cl], w, label=f'Conditioned {cl}', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    for i, (u, v) in enumerate(zip(perc_unc, perc_cond[args.condition_labels[0]])):
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
        writer.writerow(['category', 'unc_percent'] + [f'cond_{cl}_percent' for cl in args.condition_labels])
        for cat, u, *v in zip(categories, perc_unc, *[perc_cond[cl] for cl in args.condition_labels]):
            writer.writerow([cat, f"{u:.1f}"] + [f"{vi:.1f}" for vi in v])
    print(f"Saved metrics CSV: {out_csv1}")

    # living-only percentages
    living_cats = categories[:-1]
    perc_unc_live = perc_unc[:-1]
    perc_cond_live = {}
    for cl in args.condition_labels:
        perc_cond_live[cl] = perc_cond[cl][:-1]
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(living_cats))
    ax.bar(x - w/2, perc_unc_live, w, label='Unconditioned', color='skyblue')
    for i, cl in enumerate(args.condition_labels):
        ax.bar(x + w/2 + i*w, perc_cond_live[cl], w, label=f'Conditioned {cl}', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(living_cats, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    for i, (u, v) in enumerate(zip(perc_unc_live, perc_cond_live[args.condition_labels[0]])):
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
        writer.writerow(['category', 'unc_percent'] + [f'cond_{cl}_percent' for cl in args.condition_labels])
        for cat, u, *v in zip(living_cats, perc_unc_live, *[perc_cond_live[cl] for cl in args.condition_labels]):
            writer.writerow([cat, f"{u:.1f}"] + [f"{vi:.1f}" for vi in v])
    print(f"Saved metrics CSV: {out_csv2}")

    # compute and plot delta percentages overall
    delta_percentages = {}
    for cl in args.condition_labels:
        delta_percentages[cl] = [v - u for u, v in zip(perc_unc, perc_cond[cl])]
    print("Plotting overall delta percentages...")
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(categories))
    for i, cl in enumerate(args.condition_labels):
        ax.bar(x + i*w, delta_percentages[cl], w, label=f'Conditioned {cl}', color='purple')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Δ Percentage (Cond − Uncond)')
    for i, v in enumerate(delta_percentages[args.condition_labels[0]]):
        va = 'bottom' if v >= 0 else 'top'
        ax.text(i, v, f"{v:.1f}%", ha='center', va=va)
    ax.legend()
    plt.tight_layout()
    out_fig_delta = os.path.join(args.out_dir, 'cond_vs_unc_delta_percentages.png')
    fig.savefig(out_fig_delta)
    print(f"Saved delta figure: {out_fig_delta}")

    # save delta CSV overall
    out_csv_delta = os.path.join(args.out_dir, 'cond_vs_unc_metrics_delta.csv')
    with open(out_csv_delta, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category'] + [f'delta_{cl}_percent' for cl in args.condition_labels])
        for cat, *v in zip(categories, *[delta_percentages[cl] for cl in args.condition_labels]):
            writer.writerow([cat] + [f"{vi:.1f}" for vi in v])
    print(f"Saved delta metrics CSV: {out_csv_delta}")

    # living-only delta percentages
    delta_live = {}
    for cl in args.condition_labels:
        delta_live[cl] = [v - u for u, v in zip(perc_unc_live, perc_cond_live[cl])]
    print("Plotting living-only delta percentages...")
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(living_cats))
    for i, cl in enumerate(args.condition_labels):
        ax.bar(x + i*w, delta_live[cl], w, label=f'Conditioned {cl}', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(living_cats, rotation=45, ha='right')
    ax.set_ylabel('Δ Percentage (Cond − Uncond)')
    for i, v in enumerate(delta_live[args.condition_labels[0]]):
        va = 'bottom' if v >= 0 else 'top'
        ax.text(i, v, f"{v:.1f}%", ha='center', va=va)
    ax.legend()
    plt.tight_layout()
    out_fig_delta_live = os.path.join(args.out_dir, 'cond_vs_unc_delta_percentages_living.png')
    fig.savefig(out_fig_delta_live)
    print(f"Saved living-only delta figure: {out_fig_delta_live}")

    # save living-only delta CSV
    out_csv_delta_live = os.path.join(args.out_dir, 'cond_vs_unc_metrics_delta_living.csv')
    with open(out_csv_delta_live, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category'] + [f'delta_{cl}_percent' for cl in args.condition_labels])
        for cat, *v in zip(living_cats, *[delta_live[cl] for cl in args.condition_labels]):
            writer.writerow([cat] + [f"{vi:.1f}" for vi in v])
    print(f"Saved living-only delta metrics CSV: {out_csv_delta_live}")

if __name__ == '__main__':
    main()

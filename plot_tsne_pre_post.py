#!/usr/bin/env python3
"""
TSNE comparison of pre- and post- reverse-step noise for diffusion samples.
Generates two figures:
1) Living vs Death clusters (pre vs post)
2) Detailed life-type clusters (life, death, still-life, p2, other) pre vs post
"""
import os, sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.metrics import detect_period
from utils.gol_simulator import simulate

sys.path.append(os.path.dirname(__file__))
from models.unet import UNet
from models.diffusion import Diffusion


def parse_label(x):
    if x.lower() == 'none':
        return None
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid condition label: {x}")


def classify_pattern(grid):
    s = grid.sum()
    if s == 0:
        return 'death'
    hist = simulate(grid, steps=50)
    per = detect_period(hist)
    if per == 1:
        return 'still-life'
    elif per == 2:
        return 'oscillator_p2'
    else:
        return 'other_life'


def main():
    p = argparse.ArgumentParser(description="TSNE pre vs post noise")
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--grid_size', type=int, default=32)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--condition_labels', type=parse_label, nargs='+', default=[None, 0, 1],
                   help='labels: None for uncond, then each cond label')
    p.add_argument('--out_dir', type=str, default='./tsne_noise',
                   help='output dir')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    diffusion = Diffusion(timesteps=args.timesteps, device=device)

    all_feat = []
    labels_basic = []
    labels_type = []
    prepost = []
    conds = []

    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    for cl in args.condition_labels:
        # classify on final full samples for labeling
        with torch.no_grad():
            x_final = diffusion.sample(model, shape, c=None if cl is None else torch.full((args.num_samples,), cl, device=device, dtype=torch.long))
        bin_final = (x_final.squeeze(1).cpu().numpy() > args.threshold).astype(np.uint8)
        types_final = [classify_pattern(g) for g in bin_final]
        basic_final = ['death' if t=='death' else 'life' for t in types_final]

        # sample one-step pre/post features
        with torch.no_grad():
            x = torch.randn(shape, device=device)
            t_step = torch.full((args.num_samples,), args.timesteps - 1, device=device, dtype=torch.long)
            x_pre = x
            x_post = diffusion.p_sample(model, x, t_step,
                                       c=None if cl is None else torch.full((args.num_samples,), cl, device=device, dtype=torch.long))
        # flatten arrays
        pre = x_pre.squeeze(1).cpu().numpy().reshape(args.num_samples, -1)
        post = x_post.squeeze(1).cpu().numpy().reshape(args.num_samples, -1)
        # accumulate features and labels
        all_feat.append(pre)
        labels_basic += basic_final
        labels_type += types_final
        prepost += ['pre'] * args.num_samples
        conds += [f"cond_{cl}" for _ in range(args.num_samples)]
        all_feat.append(post)
        labels_basic += basic_final
        labels_type += types_final
        prepost += ['post'] * args.num_samples
        conds += [f"cond_{cl}" for _ in range(args.num_samples)]

    X = np.vstack(all_feat)
    print("Label distribution (basic):", Counter(labels_basic))
    print("Label distribution (types):", Counter(labels_type))
    # reduce dims before TSNE
    X_pca = PCA(n_components=50).fit_transform(X)
    X_tsne = TSNE(n_components=2).fit_transform(X_pca)
    print("Computed t-SNE embeddings.")

    # Plot 1: living vs death
    plt.figure(figsize=(6,6))
    for pp in ['pre','post']:
        for lb in ['life','death']:
            idx = [(pp_, lb_) == (pp, lb) for pp_, lb_ in zip(prepost, labels_basic)]
            plt.scatter(X_tsne[idx,0], X_tsne[idx,1], label=f"{lb}_{pp}", alpha=0.6, s=5)
    plt.legend()
    plt.title('t-SNE: Living vs Death (pre vs post)')
    plt.savefig(os.path.join(args.out_dir, 'tsne_life_death.png'))
    print(f"Saved living vs death t-SNE to {os.path.join(args.out_dir, 'tsne_life_death.png')}")

    # Plot 2: detailed types
    plt.figure(figsize=(6,6))
    types_unique = ['death','still-life','oscillator_p2','other_life']
    markers = {'pre':'o','post':'x'}
    for pp in ['pre','post']:
        for lb in types_unique:
            idx = [(pp_, lb_) == (pp, lb) for pp_, lb_ in zip(prepost, labels_type)]
            plt.scatter(X_tsne[idx,0], X_tsne[idx,1], label=f"{lb}_{pp}", marker=markers[pp], alpha=0.6, s=5)
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('t-SNE: Life Types (pre vs post)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'tsne_life_types.png'))
    print(f"Saved life types t-SNE to {os.path.join(args.out_dir, 'tsne_life_types.png')}")

if __name__ == '__main__':
    main()

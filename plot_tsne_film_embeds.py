#!/usr/bin/env python3
"""
TSNE on UNet FiLM embeddings: compare time + class embeddings pre- and post-conditioning.
"""
import os, sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# project imports
sys.path.append(os.path.dirname(__file__))
from models.unet import UNet


def parse_label(x):
    if x.lower() == 'none':
        return None
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid condition label: {x}")


def main():
    p = argparse.ArgumentParser("TSNE on UNet FiLM embeddings pre/post conditioning")
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--condition_labels', type=parse_label, nargs='+', default=[None,0,1])
    p.add_argument('--out_dir', type=str, default='./tsne_film')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_embeds = []
    prepost = []
    conds = []

    for cl in args.condition_labels:
        # sample random timesteps
        t = torch.randint(0, args.timesteps, (args.num_samples,), device=device)
        # pre-conditioning time embedding
        with torch.no_grad():
            emb_pre = model.time_emb(t)
        # post-conditioning: add class embedding if provided
        if cl is None:
            emb_post = emb_pre.clone()
        else:
            c = torch.full((args.num_samples,), cl, device=device, dtype=torch.long)
            with torch.no_grad():
                class_emb = model.class_emb(c)
            emb_post = emb_pre + class_emb
        # collect
        all_embeds.append(emb_pre.cpu().numpy())
        prepost += ['pre'] * args.num_samples
        conds += [f"cond_{cl}"] * args.num_samples
        all_embeds.append(emb_post.cpu().numpy())
        prepost += ['post'] * args.num_samples
        conds += [f"cond_{cl}"] * args.num_samples

    X = np.vstack(all_embeds)
    print("Embedding matrix shape:", X.shape)
    X_tsne = TSNE(n_components=2).fit_transform(X)
    print("Computed t-SNE on FiLM embeddings.")

    # BASIC grouping: death vs alive
    basic = ['death' if c=='cond_0' else 'alive' for c in conds]
    plt.figure(figsize=(8,8))
    markers = {'pre':'o','post':'x'}
    for lbl in ['death','alive']:
        for pp in ['pre','post']:
            idx = [(pp_, b)==(pp, lbl) for pp_, b in zip(prepost, basic)]
            plt.scatter(
                X_tsne[idx,0], X_tsne[idx,1],
                label=f"{lbl}_{pp}", marker=markers[pp], alpha=0.6, s=5
            )
    plt.legend(bbox_to_anchor=(1,1))
    plt.title("t-SNE: basic grouping (death vs alive)")
    plt.tight_layout()
    out_basic = os.path.join(args.out_dir, 'tsne_film_basic.png')
    plt.savefig(out_basic)
    print(f"Saved basic t-SNE to {out_basic}")

    # DETAILED grouping: pre/post clusters per condition
    plt.figure(figsize=(8,8))
    markers = {'pre':'o','post':'x'}
    for cond in sorted(set(conds)):
        for pp in ['pre','post']:
            idx = [(pp_, c_)==(pp,cond) for pp_,c_ in zip(prepost,conds)]
            plt.scatter(
                X_tsne[idx,0], X_tsne[idx,1],
                label=f"{cond}_{pp}", marker=markers[pp], alpha=0.6, s=5
            )
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('t-SNE: UNet FiLM embeddings pre/post conditioning')
    plt.tight_layout()
    out_file = os.path.join(args.out_dir, 'tsne_film_prepost.png')
    plt.savefig(out_file)
    print(f"Saved FiLM embeddings t-SNE to {out_file}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Figure 7: t-SNE on diffusion intermediates colored by living vs death and by life-type clusters.
Figure 8: Cosine similarity heatmap of class embeddings.
"""
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# add repo to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from models.unet import UNet
from models.diffusion import Diffusion
from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid

# CONFIGURATION
CKPT         = 'finished_models/model_final_quota.pt'
DEVICE       = 'cuda'   # or 'cpu'
TIMESTEPS    = 200
NUM_SAMPLES  = 500      # TSNE speed
INTERVAL     = 50       # record every 50 timesteps
TSNE_PARAMS  = dict(n_components=2, perplexity=30, random_state=0)

# CLASS NAMES
cats_types = ['died_out','still_life','oscillator_period_2','others']

# load model + diffusion
model = UNet(dropout=0.0, num_classes=5).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

diff = Diffusion(timesteps=TIMESTEPS,
                 device=DEVICE,
                 schedule='cosine',
                 guidance_scale=1.0)


def get_intermediates(condition=None):
    """Run diffusion, record intermediates and return list of numpy arrays and final binary grids."""
    c = None if condition is None else torch.full((NUM_SAMPLES,), condition, dtype=torch.long, device=DEVICE)
    x = torch.randn((NUM_SAMPLES,1,32,32), device=DEVICE)
    intermediates = []
    for t in reversed(range(0, TIMESTEPS+1, INTERVAL)):
        intermediates.append(x.clone().cpu().numpy().reshape(NUM_SAMPLES, -1))
        t_tensor = torch.full((NUM_SAMPLES,), t, dtype=torch.long, device=DEVICE)
        x = diff.p_sample(model, x, t_tensor, c)
    return intermediates, x.clone().cpu().numpy().squeeze(1)  # final float grids


def plot_tsne(data, labels, title, fname):
    """Run TSNE on flattened data and plot colored by labels."""
    tsne = TSNE(**TSNE_PARAMS)
    embed = tsne.fit_transform(data)
    plt.figure(figsize=(6,6))
    for lab in np.unique(labels):
        idx = labels==lab
        plt.scatter(embed[idx,0], embed[idx,1], label=str(lab), alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")


def figure7():
    # unconditioned (living vs dead)
    uncond_inter, final_arrs = get_intermediates(condition=None)
    # classify final as live/dead
    bin_arrs = (final_arrs>0.5).astype(np.uint8)
    ld_labels = np.array(['live' if classify_grid(a,0)!='died_out' else 'dead' for a in bin_arrs])
    # repeat labels for each intermediate
    ld_labels_rep = np.tile(ld_labels, len(uncond_inter))
    data_flat = np.vstack(uncond_inter)
    plot_tsne(data_flat, ld_labels_rep, 'Living vs Dead Clusters', 'fig7_l vs d.png')

    # conditioned types
    cond_all = {'still_life':1, 'oscillator_period_2':2, 'others':3}
    all_data, all_labels = [], []
    for name, cond in cond_all.items():
        inter, final_arrs = get_intermediates(condition=cond)
        bin_arrs = (final_arrs>0.5).astype(np.uint8)
        type_labels = np.array([classify_grid(a,0) for a in bin_arrs])
        all_data.append(np.vstack(inter))
        all_labels.append(np.repeat(type_labels, len(inter)))
    data2 = np.vstack(all_data)
    labels2 = np.concatenate(all_labels)
    plot_tsne(data2, labels2, 'Type Clusters under Conditioning', 'fig7_types.png')


def figure8():
    # cosine similarity of class embeddings
    embeddings = model.class_emb.weight.detach().cpu().numpy()
    sim = cosine_similarity(embeddings)
    plt.figure(figsize=(5,5))
    plt.imshow(sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cats_types)), cats_types, rotation=45)
    plt.yticks(range(len(cats_types)), cats_types)
    plt.title('Embedding Cosine Similarity')
    plt.tight_layout()
    plt.savefig('fig8_embedding_similarity.png')
    print('Saved fig8_embedding_similarity.png')


def main():
    figure7()
    figure8()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Figure 8: Cosine similarity of class embeddings for each condition.
"""
import os, sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# ensure project path
sys.path.append(os.path.dirname(__file__))
from models.unet import UNet

LABEL_NAMES = ['Death', 'Alive', 'Still Life', 'Oscillator P2', 'Other Life']


def main():
    parser = argparse.ArgumentParser(
        description="Figure 8: Class embedding cosine similarity"
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='path to trained model checkpoint'
    )
    parser.add_argument(
        '--out_dir', type=str, default='./fig8',
        help='directory to save figure and CSV'
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # extract class embeddings
    W = model.class_emb.weight.data.cpu().numpy()  # shape (5, D)
    # normalize rows
    W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)
    # cosine similarity matrix
    cosim = W_norm @ W_norm.T

    # plot heatmap
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cosim, vmin=-1, vmax=1, cmap='coolwarm')
    # ticks and labels
    ax.set_xticks(np.arange(len(LABEL_NAMES)))
    ax.set_yticks(np.arange(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(LABEL_NAMES)
    # annotate values
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, f"{cosim[i,j]:.2f}",
                    ha='center', va='center', color='black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title('Cosine Similarity of Class Embeddings')
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, 'figure8_class_embedding_similarity.png')
    fig.savefig(fig_path)
    print(f"Saved figure: {fig_path}")
    plt.show()

    # save CSV
    import csv
    csv_path = os.path.join(args.out_dir, 'figure8_class_embedding_cosine.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + LABEL_NAMES)
        for name, row in zip(LABEL_NAMES, cosim):
            writer.writerow([name] + [f"{v:.4f}" for v in row])
    print(f"Saved CSV: {csv_path}")


if __name__ == '__main__':
    main()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples
import argparse


def load_train_patterns(data_dir):
    patterns = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.npy'):
            arr = np.load(os.path.join(data_dir, fname))
            patterns.append(arr.astype(np.uint8))
    return patterns


def save_grid(samples, path, rows=4, cols=4):
    # samples: (N,1,H,W) binary
    N, _, H, W = samples.shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        if i < N:
            ax.imshow(samples[i,0], cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='./eval_outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # load train patterns for novelty
    train_patterns = load_train_patterns(args.data_dir)
    # prepare model
    # infer grid size from first pattern
    H = train_patterns[0].shape[0]
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    # diffusion
    diffusion = Diffusion(timesteps=args.timesteps, device=args.device)
    # sample
    with torch.no_grad():
        samples = diffusion.sample(model, (args.num_samples,1,H,H))
    # binarize
    bin_samples = (samples > 0.5).float()
    # save grid
    save_grid(bin_samples.cpu().numpy(), os.path.join(args.out_dir, 'samples.png'))
    # evaluate
    results = evaluate_samples(samples, train_patterns)
    # print results
    for k, v in results.items():
        print(f"{k}: {v}")
    # save metrics
    with open(os.path.join(args.out_dir, 'metrics.txt'), 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

if __name__ == '__main__':
    main()

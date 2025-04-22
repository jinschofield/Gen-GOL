import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples, detect_period
from utils.gol_simulator import simulate
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
        # remove ticks but keep frame for border
        ax.set_xticks([])
        ax.set_yticks([])
        # add border around each sample
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
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
    parser.add_argument('--threshold', type=float, default=0.5, help='binary threshold for sample activation')
    args = parser.parse_args()

    threshold = args.threshold

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
    # clamp outputs to [0,1] for meaningful thresholding
    samples = torch.clamp(samples, 0.0, 1.0)
    # binarize using specified threshold
    bin_samples = (samples > threshold).float()
    # visualize living configurations
    living_idxs = []
    for i in range(bin_samples.shape[0]):
        g = bin_samples[i,0].cpu().numpy().astype(np.uint8)
        hist = simulate(g, steps=args.timesteps)
        per = detect_period(hist)
        if per is not None:
            living_idxs.append(i)
    if living_idxs:
        living_samples = bin_samples[living_idxs]
        m = len(living_idxs)
        rows = int(np.ceil(m ** 0.5))
        cols = int(np.ceil(m / rows))
        save_grid(living_samples.cpu().numpy(), os.path.join(args.out_dir, 'living_samples.png'), rows, cols)
        print(f"Saved {m} living configurations to 'living_samples.png'")
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
    # Random baseline for comparison
    # generate random samples
    rand_samples = torch.rand_like(samples)
    rand_samples = torch.clamp(rand_samples, 0.0, 1.0)
    rand_bin = (rand_samples > threshold).float()
    # save random grid
    save_grid(rand_bin.cpu().numpy(), os.path.join(args.out_dir, 'random_samples.png'))
    # evaluate random baseline
    rand_results = evaluate_samples(rand_samples, train_patterns)
    print("\nRandom baseline results:")
    for k, v in rand_results.items():
        print(f"{k}: {v}")
    # save random metrics
    with open(os.path.join(args.out_dir, 'random_metrics.txt'), 'w') as f:
        for k, v in rand_results.items():
            f.write(f"{k}: {v}\n")

if __name__ == '__main__':
    main()

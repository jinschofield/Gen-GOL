import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def save_grid(samples, path, rows=4, cols=4, labels=None):
    # samples: (N,1,H,W) binary
    N, _, H, W = samples.shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    # ensure axes is a 2D array for consistent indexing
    axes = np.array(axes).reshape(rows, cols)
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
        # annotate index label if provided
        if labels is not None and i < len(labels):
            ax.text(0.05, 0.9, str(labels[i]), color='red', fontsize=8, transform=ax.transAxes)
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
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'], help='noise schedule')
    parser.add_argument('--sample_method', type=str, default='ancestral', choices=['ancestral','ddim'], help='sampling method')
    parser.add_argument('--eta', type=float, default=0.0, help='eta for ddim sampling')
    parser.add_argument('--threshold', type=float, default=0.5, help='binary threshold for sample activation')
    parser.add_argument('--class_label', type=int, default=1, choices=[0,1], help='0=die, 1=survive conditioning flag')
    parser.add_argument('--animate', action='store_true', help='animate a sample history')
    parser.add_argument('--anim_idx', type=int, default=0, help='sample index to animate')
    parser.add_argument('--anim_steps', type=int, default=None, help='simulation steps for animation (defaults to timesteps)')
    parser.add_argument('--anim_last', type=int, default=None, help='only animate the last N frames of history')
    parser.add_argument('--baseline_model', type=str, default=None,
                        help='path to untrained model checkpoint for baseline comparison')
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
    # diffusion with chosen noise schedule
    diffusion = Diffusion(timesteps=args.timesteps, device=args.device, schedule=args.schedule)
    # build condition tensor
    c = torch.full((args.num_samples,), args.class_label, device=args.device, dtype=torch.long)
    # sample
    with torch.no_grad():
        # generate with conditional label
        shape = (args.num_samples, 1, H, H)
        if args.sample_method == 'ancestral':
            samples = diffusion.sample(model, shape, c)
        else:
            samples = diffusion.ddim_sample(model, shape, eta=args.eta, c=c)
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
        print(f"Living sample indices: {living_idxs}")
        living_samples = bin_samples[living_idxs]
        m = len(living_idxs)
        rows = int(np.ceil(m ** 0.5))
        cols = int(np.ceil(m / rows))
        save_grid(living_samples.cpu().numpy(), os.path.join(args.out_dir, 'living_samples.png'), rows, cols, labels=living_idxs)
        print(f"Saved {m} living configurations to 'living_samples.png'")
        # sample same number of died-out configs for comparison
        died_idxs = [i for i in range(bin_samples.shape[0]) if i not in living_idxs]
        if died_idxs:
            # if fewer died than living, allow repeats
            died_sel = np.random.choice(died_idxs, size=m, replace=len(died_idxs) < m)
            died_samples = bin_samples[died_sel]
            print(f"Died sample indices: {died_sel.tolist()}")
            save_grid(died_samples.cpu().numpy(), os.path.join(args.out_dir, 'died_samples.png'), rows, cols, labels=died_sel)
            print(f"Saved {m} died configurations to 'died_samples.png'")

    # save grid
    save_grid(bin_samples.cpu().numpy(), os.path.join(args.out_dir, 'samples.png'))
    # evaluate
    results = evaluate_samples(samples, train_patterns, max_steps=args.timesteps, threshold=args.threshold)
    # print results
    print("\nTrained+conditioned (samples.png, metrics.txt):")
    for k, v in results.items():
        print(f"{k}: {v}")
    # save metrics
    with open(os.path.join(args.out_dir, 'metrics.txt'), 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    # evaluate trained model without condition
    with torch.no_grad():
        if args.sample_method == 'ancestral':
            samples_nc = diffusion.sample(model, shape, c=None)
        else:
            samples_nc = diffusion.ddim_sample(model, shape, eta=args.eta, c=None)
    samples_nc = torch.clamp(samples_nc, 0.0, 1.0)
    bin_nc = (samples_nc > threshold).float()
    save_grid(bin_nc.cpu().numpy(), os.path.join(args.out_dir, 'samples_no_cond.png'))
    results_nc = evaluate_samples(samples_nc, train_patterns, max_steps=args.timesteps, threshold=args.threshold)
    print("\nTrained+unconditioned (samples_no_cond.png, metrics_no_cond.txt):")
    for k, v in results_nc.items():
        print(f"{k}: {v}")
    with open(os.path.join(args.out_dir, 'metrics_no_cond.txt'), 'w') as f:
        for k, v in results_nc.items():
            f.write(f"{k}: {v}\n")
    # compute improvement due to conditioning for trained
    improvement_trained = {k: results.get(k,0) - results_nc.get(k,0) for k in results_nc}
    print("\nImprovements from conditioning (trained):")
    for k, v in improvement_trained.items():
        print(f"{k}: {v}")

    # evaluate untrained baseline if provided
    if args.baseline_model:
        # skip if checkpoint not found
        if not os.path.isfile(args.baseline_model):
            print(f"Baseline model checkpoint '{args.baseline_model}' not found, skipping baseline evaluation.")
        else:
            base_model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(args.device)
            base_state = torch.load(args.baseline_model, map_location=args.device)
            base_model.load_state_dict(base_state)
            base_model.eval()
            with torch.no_grad():
                if args.sample_method == 'ancestral':
                    base_samples = diffusion.sample(base_model, shape, c)
                else:
                    base_samples = diffusion.ddim_sample(base_model, shape, eta=args.eta, c=c)
            base_samples = torch.clamp(base_samples, 0.0, 1.0)
            base_results = evaluate_samples(base_samples, train_patterns, max_steps=args.timesteps, threshold=args.threshold)
            print("\nUntrained+conditioned (baseline_samples.png, baseline_metrics.txt):")
            for k, v in base_results.items():
                print(f"{k}: {v}")
            # save baseline metrics
            with open(os.path.join(args.out_dir, 'baseline_metrics.txt'), 'w') as f:
                for k, v in base_results.items():
                    f.write(f"{k}: {v}\n")

            # evaluate untrained model without condition
            with torch.no_grad():
                if args.sample_method == 'ancestral':
                    base_nc = diffusion.sample(base_model, shape, c=None)
                else:
                    base_nc = diffusion.ddim_sample(base_model, shape, eta=args.eta, c=None)
            base_nc = torch.clamp(base_nc, 0.0, 1.0)
            bin_base_nc = (base_nc > threshold).float()
            save_grid(bin_base_nc.cpu().numpy(), os.path.join(args.out_dir, 'baseline_samples_no_cond.png'))
            base_results_nc = evaluate_samples(base_nc, train_patterns, max_steps=args.timesteps, threshold=args.threshold)
            print("\nUntrained+unconditioned (baseline_samples_no_cond.png, baseline_metrics_no_cond.txt):")
            for k, v in base_results_nc.items():
                print(f"{k}: {v}")
            with open(os.path.join(args.out_dir, 'baseline_metrics_no_cond.txt'), 'w') as f:
                for k, v in base_results_nc.items():
                    f.write(f"{k}: {v}\n")
            # compute improvement due to conditioning for untrained
            improvement_untrained = {k: base_results.get(k,0) - base_results_nc.get(k,0) for k in base_results_nc}
            print("\nImprovements from conditioning (untrained):")
            for k, v in improvement_untrained.items():
                print(f"{k}: {v}")

    # Random baseline for comparison
    # generate random samples
    rand_samples = torch.rand_like(samples)
    rand_samples = torch.clamp(rand_samples, 0.0, 1.0)
    rand_bin = (rand_samples > threshold).float()
    # save random grid
    save_grid(rand_bin.cpu().numpy(), os.path.join(args.out_dir, 'random_samples.png'))
    # evaluate random baseline
    rand_results = evaluate_samples(rand_samples, train_patterns, max_steps=args.timesteps, threshold=args.threshold)
    print("\nRandom baseline results:")
    for k, v in rand_results.items():
        print(f"{k}: {v}")
    # save random metrics
    with open(os.path.join(args.out_dir, 'random_metrics.txt'), 'w') as f:
        for k, v in rand_results.items():
            f.write(f"{k}: {v}\n")

    if args.animate:
        total = bin_samples.shape[0]
        idx = min(args.anim_idx, total-1)
        if args.anim_idx >= total:
            print(f"anim_idx {args.anim_idx} out of range (0 to {total-1}), using {idx}")
        g = bin_samples[idx,0].cpu().numpy()
        # determine simulation steps: anim_steps or full timesteps
        sim_steps = args.anim_steps if args.anim_steps is not None else args.timesteps
        full_history = simulate(g, steps=sim_steps)
        # trim to last N frames if requested
        if args.anim_last is not None:
            n = min(args.anim_last, len(full_history))
            history = full_history[-n:]
        else:
            history = full_history
        # print status of animated sample
        if full_history[-1].sum() == 0:
            print(f"Animated sample {idx} died out")
        else:
            print(f"Animated sample {idx} survived")
        fig, ax = plt.subplots()
        im = ax.imshow(history[0], cmap='gray_r', vmin=0, vmax=1)
        def update(i):
            im.set_data(history[i])
            return [im]
        anim = animation.FuncAnimation(fig, update, frames=len(history), interval=200, blit=True)
        anim_path = os.path.join(args.out_dir, f'anim_{idx}.gif')
        anim.save(anim_path, writer='pillow', fps=5)
        print(f"Saved animation: {anim_path}")

if __name__ == '__main__':
    main()

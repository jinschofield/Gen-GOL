#!/usr/bin/env python3
"""
Compare unconditioned vs conditioned (same class) percentages per category for multiple models.
"""
import os, argparse, csv
import torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples


def compute_counts(results):
    total = results.get('total', 0)
    died = results.get('died_out', 0)
    sl = results.get('still_life', 0)
    osc2 = results.get('oscillator_p2', 0)
    alive = total - died
    other = alive - sl - osc2
    return {'total': total, 'alive': alive, 'still_life': sl,
            'oscillator_p2': osc2, 'other': other, 'dead': died}


def plot_same_category(categories, perc_unc, perc_cond_diag, model_name, out_dir):
    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - w/2, perc_unc, w, label='Unconditioned', color='skyblue')
    ax.bar(x + w/2, perc_cond_diag, w, label='Conditioned', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{model_name}: Uncond vs Cond (same class)')
    for i,(u,c) in enumerate(zip(perc_unc, perc_cond_diag)):
        ax.text(i - w/2, u, f"{u:.1f}%", ha='center', va='bottom')
        ax.text(i + w/2, c, f"{c:.1f}%", ha='center', va='bottom')
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'{model_name}_same_category.png')
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")
    # CSV
    csv_path = os.path.join(out_dir, f'{model_name}_same_category.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'unc_pct', 'cond_pct'])
        for cat, u, c in zip(categories, perc_unc, perc_cond_diag):
            writer.writerow([cat, f"{u:.1f}", f"{c:.1f}"])
    print(f"Saved CSV: {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Plot same-class cond vs uncond for categories and models")
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='list of model checkpoint paths')
    p.add_argument('--model_names', nargs='+', default=None,
                   help='names for each checkpoint output folder')
    p.add_argument('--cond_labels', nargs='+', type=int, default=None,
                   help='class labels to condition on (must match categories count)')
    p.add_argument('--num_samples', type=int, default=300)
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--schedule', choices=['linear','cosine'], default='linear')
    p.add_argument('--guidance_scale', type=float, default=1.0)
    p.add_argument('--grid_size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--train_data_dir', default=None,
                   help='directory of .npy for novelty check')
    p.add_argument('--out_dir', default='./figures_same_category')
    args = p.parse_args()

    if args.model_names:
        assert len(args.checkpoints) == len(args.model_names), \
            "--model_names must match number of checkpoints"
        names = args.model_names
    else:
        names = [os.path.splitext(os.path.basename(c))[0] for c in args.checkpoints]

    categories = ['Alive', 'Still Life', 'Oscillator P2', 'Other', 'Dead']
    count_keys = ['alive', 'still_life', 'oscillator_p2', 'other', 'dead']

    for ckpt, name in zip(args.checkpoints, names):
        out_model = os.path.join(args.out_dir, name)
        os.makedirs(out_model, exist_ok=True)
        device = args.device

        # load model
        model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # determine cond_labels
        if args.cond_labels is not None:
            cond_labels = args.cond_labels
        else:
            try:
                cond_labels = list(range(model.class_emb.num_embeddings))
            except AttributeError:
                raise RuntimeError("Model has no class_emb, cannot condition")
        assert len(cond_labels) == len(categories), \
            "Number of cond_labels must equal number of categories"

        # diffusion
        diff = Diffusion(timesteps=args.timesteps, device=device,
                         schedule=args.schedule, guidance_scale=args.guidance_scale)
        shape = (args.num_samples, 1, args.grid_size, args.grid_size)

        with torch.no_grad():
            samp_unc = torch.clamp(diff.sample(model, shape, c=None), 0, 1)
            cond_samples = {
                cl: torch.clamp(
                    diff.sample(model, shape,
                                c=torch.full((args.num_samples,), cl,
                                              device=device, dtype=torch.long)),
                    0, 1)
                for cl in cond_labels
            }

        # load train patterns
        train_patterns = []
        if args.train_data_dir and os.path.isdir(args.train_data_dir):
            for fn in sorted(os.listdir(args.train_data_dir)):
                if fn.endswith('.npy'):
                    train_patterns.append(np.load(os.path.join(args.train_data_dir, fn)))
        print(f"Loaded {len(train_patterns)} training patterns (novelty)")

        # evaluate
        res_unc = evaluate_samples(samp_unc, train_patterns,
                                   max_steps=args.timesteps, threshold=args.threshold)
        res_cond = {cl: evaluate_samples(cond_samples[cl], train_patterns,
                                         max_steps=args.timesteps, threshold=args.threshold)
                    for cl in cond_labels}

        # compute counts & percentages
        cnt_unc = compute_counts(res_unc)
        cnt_cond = {cl: compute_counts(res_cond[cl]) for cl in cond_labels}
        total_unc = cnt_unc['total']
        perc_unc = [cnt_unc[k]/total_unc*100 if total_unc>0 else 0 for k in count_keys]

        # same-class cond percentages
        perc_cond_diag = []
        for idx, cl in enumerate(cond_labels):
            total_c = cnt_cond[cl]['total']
            val = cnt_cond[cl][count_keys[idx]]/total_c*100 if total_c>0 else 0
            perc_cond_diag.append(val)

        plot_same_category(categories, perc_unc, perc_cond_diag, name, out_model)

if __name__ == '__main__':
    main()

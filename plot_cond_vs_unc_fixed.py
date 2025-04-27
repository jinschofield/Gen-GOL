#!/usr/bin/env python3
"""
Compare unconditioned vs conditioned diffusion outputs for a GoL model.
Produces: novelty plot/CSV, overall category percentages plot/CSV, delta percentages plot/CSV.
"""
import os
import argparse
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples


def compute_counts(results):
    total = results.get('total', 0)
    dead  = results.get('died_out', 0)
    sl    = results.get('still_life', 0)
    osc2  = results.get('oscillator_p2', 0)
    alive = total - dead
    other = alive - sl - osc2
    return {'total': total, 'alive': alive, 'still_life': sl, 'osc_p2': osc2, 'other': other, 'dead': dead}


def plot_bar_comparison(categories, vals_unc, vals_cond_list, labels, colors, ylabel, title, out_path):
    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(categories)*1), 6))
    ax.bar(x - w/2, vals_unc, w, label=labels[0], color=colors[0])
    for i, vals_cond in enumerate(vals_cond_list, 1):
        ax.bar(x + w/2 + (i-1)*w, vals_cond, w, label=labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, u in enumerate(vals_unc):
        ax.text(i - w/2, u, f"{u:.1f}%", ha='center', va='bottom')
    for i, vals_cond in enumerate(vals_cond_list, 1):
        for j, v in enumerate(vals_cond):
            ax.text(j + w/2 + (i-1)*w, v, f"{v:.1f}%", ha='center', va='bottom')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Compare cond vs uncond generations")
    p.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p.add_argument('--timesteps', type=int, default=200)
    p.add_argument('--num_samples', type=int, default=300)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--condition_labels', type=int, nargs='+', default=[0,1])
    p.add_argument('--schedule', choices=['linear','cosine'], default='linear')
    p.add_argument('--guidance_scale', type=float, default=1.0)
    p.add_argument('--grid_size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', default='./figures_cond')
    p.add_argument('--train_data_dir', default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # load model
    model = UNet(1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    diff = Diffusion(timesteps=args.timesteps, device=device,
                     schedule=args.schedule, guidance_scale=args.guidance_scale)

    shape = (args.num_samples, 1, args.grid_size, args.grid_size)
    with torch.no_grad():
        print("Sampling unconditioned...")
        samp_unc = torch.clamp(diff.sample(model, shape, c=None), 0.0, 1.0)
        cond_samples = {}
        for cl in args.condition_labels:
            print(f"Sampling conditioned (label={cl})...")
            c_vec = torch.full((args.num_samples,), cl, device=device, dtype=torch.long)
            cond_samples[cl] = torch.clamp(diff.sample(model, shape, c=c_vec), 0.0, 1.0)

    # load training patterns
    train_patterns = []
    if args.train_data_dir and os.path.isdir(args.train_data_dir):
        for fn in sorted(os.listdir(args.train_data_dir)):
            if fn.endswith('.npy'):
                train_patterns.append(np.load(os.path.join(args.train_data_dir, fn)))
    print(f"Loaded {len(train_patterns)} training patterns")

    # evaluate novelty
    res_unc = evaluate_samples(samp_unc, train_patterns, max_steps=args.timesteps, threshold=args.threshold)
    nov_unc = res_unc.get('novel_frac',0.0)*100
    res_cond = {}
    nov_cond_list = []
    for cl in args.condition_labels:
        rc = evaluate_samples(cond_samples[cl], train_patterns, max_steps=args.timesteps, threshold=args.threshold)
        res_cond[cl] = rc
        nov_cond_list.append(rc.get('novel_frac',0.0)*100)

    # novelty CSV & plot
    nov_csv = os.path.join(args.out_dir,'novelty.csv')
    with open(nov_csv,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['cond','novelty_pct'])
        w.writerow(['unconditioned',f"{nov_unc:.2f}"])
        for cl,nv in zip(args.condition_labels,nov_cond_list):
            w.writerow([f'condition_{cl}',f"{nv:.2f}"])
    print(f"Saved novelty CSV: {nov_csv}")

    fig, ax = plt.subplots(figsize=(6,4))
    labs = ['Unconditioned'] + [f'Cond {cl}' for cl in args.condition_labels]
    vals = [nov_unc] + nov_cond_list
    cols = ['skyblue'] + [cm.tab10(i) for i in range(len(nov_cond_list))]
    ax.bar(labs, vals, color=cols)
    ax.set_ylabel('Novelty (%)')
    ax.set_title('Novelty Rate')
    for i,v in enumerate(vals): ax.text(i,v,f"{v:.1f}%",ha='center',va='bottom')
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir,'novelty.png')); plt.close()

    # counts & percentages
    cnt_unc  = compute_counts(res_unc)
    cnt_cond = {cl: compute_counts(res_cond[cl]) for cl in args.condition_labels}

    cats = ['Alive','Still Life','Oscillator P2','Other','Dead']
    vals_unc = [cnt_unc['alive'],cnt_unc['still_life'],cnt_unc['osc_p2'],cnt_unc['other'],cnt_unc['dead']]
    perc_unc = [(v/cnt_unc['total']*100 if cnt_unc['total']>0 else 0) for v in vals_unc]

    perc_cond = []
    for cl in args.condition_labels:
        c = cnt_cond[cl]
        v = [c['alive'],c['still_life'],c['osc_p2'],c['other'],c['dead']]
        perc_cond.append([(x/c['total']*100 if c['total']>0 else 0) for x in v])

    # overall percentages
    plot_bar_comparison(cats, perc_unc, perc_cond,
                        ['Unconditioned']+[f'Cond {cl}' for cl in args.condition_labels],
                        ['skyblue']+[cm.tab10(i) for i in range(len(perc_cond))],
                        'Percentage (%)', 'Category Percentages',
                        os.path.join(args.out_dir,'percentages.png'))
    pct_csv = os.path.join(args.out_dir,'percentages.csv')
    with open(pct_csv,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['category','unc_pct']+[f'cond_{cl}_pct' for cl in args.condition_labels])
        for i,cat in enumerate(cats): w.writerow([cat,f"{perc_unc[i]:.1f}"]+ [f"{perc_cond[j][i]:.1f}" for j in range(len(perc_cond))])
    print(f"Saved percentages CSV: {pct_csv}")

    # delta percentages
    deltas = [[cond - unc for cond,unc in zip(pc,perc_unc)] for pc in perc_cond]
    plot_bar_comparison(cats, [0]*len(cats), deltas,
                        ['Zero']+[f'Delta Cond {cl}' for cl in args.condition_labels],
                        ['white']+[cm.Purples((i+1)/len(deltas)) for i in range(len(deltas))],
                        'Δ Percentage (Cond−Uncond)', 'Delta Percentages',
                        os.path.join(args.out_dir,'delta.png'))
    dlt_csv = os.path.join(args.out_dir,'delta.csv')
    with open(dlt_csv,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['category']+[f'delta_{cl}' for cl in args.condition_labels])
        for i,cat in enumerate(cats): w.writerow([cat]+[f"{deltas[j][i]:.1f}" for j in range(len(deltas))])
    print(f"Saved delta CSV: {dlt_csv}")

if __name__ == '__main__':
    main()

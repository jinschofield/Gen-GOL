#!/usr/bin/env python3
"""
Phase 2 - Step 3:
Evaluate reproducibility of each life-type under conditional diffusion (32×32 grid).
Generates samples per category, classifies them using Phase 1 logic, and compares proportions vs. the dataset.
Outputs a descriptive CSV and prints ranking by reproducibility.
"""
import os, argparse, csv
import numpy as np
import torch
from models.unet import UNet
from models.diffusion import Diffusion
from phase_2_conditional_diffusion.category_dataset import CategoryDataset
from utils.gol_simulator import simulate
from phase_1_classification.utils.detectors import detect_gliders, detect_spaceships


def classify_grid(arr, timesteps):
    history = simulate(arr, steps=timesteps)
    if history[-1].sum() == 0:
        return 'died_out'
    # find period of final state
    last = history[-1]
    per = None
    for p in range(1, len(history)):
        if np.array_equal(history[-1-p], last):
            per = p
            break
    if per == 1:
        cat = 'still_life'
    elif per and per > 1:
        cat = f'oscillator_period_{per}'
    else:
        cat = 'others'
    first = history[0]
    if detect_gliders(first):
        cat = 'glider'
    elif detect_spaceships(first):
        cat = 'spaceship'
    return cat


def main():
    parser = argparse.ArgumentParser(description='Eval reproducibility for conditional diffusion')
    parser.add_argument('--model_ckpt', type=str, required=True,
                        help='Conditional diffusion checkpoint')
    parser.add_argument('--label_csv', type=str, required=True,
                        help='CSV of training labels [filepath, category]')
    parser.add_argument('--num_gen', type=int, default=1000,
                        help='Samples to generate per category')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='Diffusion timesteps')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['linear','cosine'], help='Noise schedule')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for UNet')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--output_csv', type=str,
                        default='phase2_cond_diffusion_32x32_data_vs_generated_proportions.csv',
                        help='Output CSV name')
    args = parser.parse_args()

    # load dataset labels
    dataset = CategoryDataset(args.label_csv)
    # data distribution
    counts_data = {cat: 0 for cat in dataset.cat_to_idx.keys()}
    for _, cat in dataset.samples:
        counts_data[cat] += 1
    total_data = len(dataset)
    pct_data = {cat: counts_data[cat] / total_data for cat in counts_data}

    # load model
    model = UNet(dropout=args.dropout).to(args.device)
    state = torch.load(args.model_ckpt, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    # diffusion sampler
    diffusion = Diffusion(
        timesteps=args.timesteps,
        schedule=args.schedule,
        guidance_scale=args.guidance_scale,
        device=args.device
    )

    # generate and classify
    counts_gen = {cat: 0 for cat in counts_data}
    for cat, idx in dataset.cat_to_idx.items():
        # sample for each category
        c_tensor = torch.full((args.num_gen,), idx, dtype=torch.long, device=args.device)
        with torch.no_grad():
            x_gen = diffusion.sample(model, shape=(args.num_gen,1,32,32), c=c_tensor)
        # threshold to binary
        arrs = (x_gen.cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
        for arr in arrs:
            cl = classify_grid(arr, args.timesteps)
            if cl in counts_gen:
                counts_gen[cl] += 1
            else:
                counts_gen[cl] = 1

    # compute generated proportions
    pct_gen = {cat: counts_gen.get(cat, 0)/args.num_gen for cat in counts_gen}

    # write CSV
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category','count_data','count_gen','pct_data','pct_gen','delta'])
        for cat in sorted(set(counts_data) | set(counts_gen)):
            cd = counts_data.get(cat,0)
            cg = counts_gen.get(cat,0)
            pd = pct_data.get(cat,0)
            pg = pct_gen.get(cat,0)
            writer.writerow([cat, cd, cg, f"{pd:.4f}", f"{pg:.4f}", f"{(pg-pd):.4f}"])

    # print ranking by reproducibility (abs delta)
    ranking = sorted(pct_gen.keys(), key=lambda c: abs(pct_gen[c]-pct_data.get(c,0)), reverse=True)
    print("Reproducibility ranking (largest delta first):")
    for cat in ranking:
        delta = pct_gen.get(cat,0)-pct_data.get(cat,0)
        print(f"{cat}: Δ = {delta:.4f} (data {pct_data.get(cat,0):.4f}, gen {pct_gen.get(cat,0):.4f})")
    print(f"Results saved to {args.output_csv}")

if __name__ == '__main__':
    main()

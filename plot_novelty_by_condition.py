#!/usr/bin/env python3
"""
Compute novelty rates for unconditioned and each conditioning label.
"""
import os, sys, argparse, csv
import torch
from evaluate import load_train_patterns
from models.unet import UNet
from models.diffusion import Diffusion
from utils.metrics import evaluate_samples

# allow imports from script directory
sys.path.append(os.path.dirname(__file__))

def main():
    p = argparse.ArgumentParser(description="Novelty by condition")
    p.add_argument('--data_dir', type=str, required=True,
                   help='directory of .npy training patterns')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to trained model checkpoint')
    p.add_argument('--condition_labels', type=int, nargs='+', default=[0,1],
                   help='list of conditioning labels')
    p.add_argument('--timesteps', type=int, default=200,
                   help='diffusion timesteps')
    p.add_argument('--num_samples', type=int, default=300,
                   help='number of samples')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='binarization threshold')
    p.add_argument('--schedule', type=str, default='linear',
                   choices=['linear','cosine'],
                   help='noise schedule')
    p.add_argument('--guidance_scale', type=float, default=1.0,
                   help='CFG scale')
    p.add_argument('--grid_size', type=int, default=32,
                   help='grid H/W')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='compute device')
    p.add_argument('--out_dir', type=str, default='./novelty_by_condition',
                   help='output dir for CSV')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # load training patterns
    print("Loading training patterns...")
    train_patterns = load_train_patterns(args.data_dir)
    print(f"Loaded {len(train_patterns)} patterns.")

    # load model
    device = args.device
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,4)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # diffusion setup
    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=device,
        schedule=args.schedule,
        guidance_scale=args.guidance_scale
    )
    shape = (args.num_samples, 1, args.grid_size, args.grid_size)

    # sample unconditioned
    with torch.no_grad():
        print("Sampling unconditioned...")
        samples_unc = diffusion.sample(model, shape, c=None)
    samples_unc = torch.clamp(samples_unc, 0.0, 1.0)
    res_unc = evaluate_samples(samples_unc, train_patterns,
                                max_steps=args.timesteps, threshold=args.threshold)
    novel_unc = res_unc.get('novel_frac', 0.0)
    print(f"Unconditioned novelty: {novel_unc:.3f}")

    # sample and eval for each conditioning label
    results = {'unconditioned': novel_unc}
    for cl in args.condition_labels:
        with torch.no_grad():
            print(f"Sampling conditioned (label={cl})...")
            c = torch.full((args.num_samples,), cl, device=device, dtype=torch.long)
            samples_c = diffusion.sample(model, shape, c=c)
        samples_c = torch.clamp(samples_c, 0.0, 1.0)
        res_c = evaluate_samples(samples_c, train_patterns,
                                 max_steps=args.timesteps, threshold=args.threshold)
        novel_c = res_c.get('novel_frac', 0.0)
        print(f"Conditioned {cl} novelty: {novel_c:.3f}")
        results[f'cond_{cl}'] = novel_c

    # write CSV
    csv_path = os.path.join(args.out_dir, 'novelty_by_condition.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'novel_frac'])
        w.writerow(['unconditioned', f"{novel_unc:.3f}"])
        for cl in args.condition_labels:
            w.writerow([f'condition_{cl}', f"{results[f'cond_{cl}']:.3f}"])
    print(f"Saved CSV: {csv_path}")

if __name__ == '__main__':
    main()

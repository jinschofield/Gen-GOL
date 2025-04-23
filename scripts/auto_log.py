#!/usr/bin/env python3
"""
Wrapper to run training with specified hyperparameters, parse stdout for losses,
and append a summary row to a CSV for easy comparison.
"""
import argparse
import subprocess
import re
import csv
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run train.py and log losses to CSV")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--live_weight', type=float, default=1.0)
    parser.add_argument('--ssim_weight', type=float, default=0.0)
    parser.add_argument('--bce_weight', type=float, default=0.0)
    parser.add_argument('--mae_weight', type=float, default=0.0)
    parser.add_argument('--noise_prob', type=float, default=0.0)
    parser.add_argument('--cf_prob', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--output_csv', default='results.csv',
                        help='Path to CSV file to append results')
    return parser.parse_args()


def run_and_log(args):
    cmd = [
        'python', 'train.py',
        '--data_dir', args.data_dir,
        '--save_dir', args.save_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--timesteps', str(args.timesteps),
        '--live_weight', str(args.live_weight),
        '--ssim_weight', str(args.ssim_weight),
        '--bce_weight', str(args.bce_weight),
        '--mae_weight', str(args.mae_weight),
        '--noise_prob', str(args.noise_prob),
        '--cf_prob', str(args.cf_prob),
        '--val_split', str(args.val_split)
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    train_pattern = re.compile(r"Step\s+(\d+),\s+Loss:\s+([0-9.]+)")
    val_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+Validation\s+Loss:\s+([0-9.]+)")
    train_losses = []
    val_losses = []
    # stream and print
    for line in proc.stdout:
        print(line, end='')
        m = train_pattern.search(line)
        if m:
            train_losses.append(float(m.group(2)))
        mv = val_pattern.search(line)
        if mv:
            val_losses.append(float(mv.group(3)))
    proc.wait()

    final_train = train_losses[-1] if train_losses else None
    final_val   = val_losses[-1]   if val_losses else None

    # ensure output directory exists
    out_dir = os.path.dirname(args.output_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    write_header = not os.path.isfile(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'data_dir','save_dir','epochs','batch_size','lr','timesteps',
                'live_weight','ssim_weight','bce_weight','mae_weight',
                'noise_prob','cf_prob','val_split',
                'final_train_loss','final_val_loss'
            ])
        writer.writerow([
            args.data_dir, args.save_dir, args.epochs, args.batch_size,
            args.lr, args.timesteps,
            args.live_weight, args.ssim_weight, args.bce_weight, args.mae_weight,
            args.noise_prob, args.cf_prob, args.val_split,
            final_train, final_val
        ])
    print(f"Results appended to {args.output_csv}")


if __name__ == '__main__':
    args = parse_args()
    run_and_log(args)

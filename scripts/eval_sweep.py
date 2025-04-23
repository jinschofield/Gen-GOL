#!/usr/bin/env python3
"""
Sweep multiple thresholds and class labels, run evaluate.py, and log conditioning improvements.
"""
import argparse
import subprocess
import re
import csv
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser("Sweep eval thresholds and cond labels")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, required=True, help='trained model checkpoint')
    parser.add_argument('--baseline_model', type=str, required=True, help='untrained model checkpoint')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.01,0.1,0.3,0.5])
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample_method', type=str, default='ancestral', choices=['ancestral','ddim'])
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--output_csv', type=str, default='eval_sweep.csv')
    return parser.parse_args()

def run_eval(data_dir, checkpoint, baseline_model, threshold, class_label, num_samples, timesteps, device, sample_method, eta):
    cmd = [
        'python', 'evaluate.py',
        '--data_dir', data_dir,
        '--checkpoint', checkpoint,
        '--baseline_model', baseline_model,
        '--threshold', str(threshold),
        '--class_label', str(class_label),
        '--num_samples', str(num_samples),
        '--timesteps', str(timesteps),
        '--device', device,
        '--sample_method', sample_method,
        '--eta', str(eta)
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    trained_imp = {}
    trained_nc  = {}
    baseline_imp = {}
    baseline_nc  = {}
    rand_imp = {}
    cur = None
    pat_tc = re.compile(r"Trained\+conditioned")
    pat_tnc = re.compile(r"Trained\+unconditioned")
    pat_bc = re.compile(r"Untrained\+conditioned")
    pat_bnc = re.compile(r"Untrained\+unconditioned")
    pat_rb = re.compile(r"Random baseline results:")
    pat_kv = re.compile(r"(\w+):\s*([\-0-9.]+)")
    for line in proc.stdout:
        if pat_tc.search(line): cur = 'tc'; continue
        if pat_tnc.search(line): cur = 'tnc'; continue
        if pat_bc.search(line): cur = 'bc'; continue
        if pat_bnc.search(line): cur = 'bnc'; continue
        if pat_rb.search(line): cur = 'rand'; continue
        m = pat_kv.search(line)
        if m and cur:
            key, val = m.group(1), float(m.group(2))
            if cur == 'tc': trained_imp[key] = val
            if cur == 'tnc': trained_nc[key] = val
            if cur == 'bc': baseline_imp[key] = val
            if cur == 'bnc': baseline_nc[key] = val
            if cur == 'rand': rand_imp[key] = val
    proc.wait()
    return trained_imp, trained_nc, baseline_imp, baseline_nc, rand_imp

if __name__ == '__main__':
    args = parse_args()
    # ensure CSV
    write_header = not os.path.isfile(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'threshold','class_label','direction',
                'trained_cond','trained_unc','trained_imp',
                'baseline_cond','baseline_unc','baseline_imp',
                'novel_trained_cond','novel_trained_unc',
                'novel_baseline_cond','novel_baseline_unc',
                'novel_random'
            ])
        for th in args.thresholds:
            for cl in [1,0]:
                trc, trnc, bc, bnc, rand = run_eval(
                    args.data_dir, args.checkpoint, args.baseline_model,
                    th, cl, args.num_samples, args.timesteps,
                    args.device, args.sample_method, args.eta
                )
                direction = 'alive' if cl==1 else 'dead'
                # select metric key
                key = 'survived_unknown' if cl==1 else 'died_out'
                # raw counts
                trc_val = trc.get(key, 0.0)
                trnc_val = trnc.get(key, 0.0)
                bc_val   = bc.get(key, 0.0)
                bnc_val  = bnc.get(key, 0.0)
                # improvements
                tr_imp = trc_val - trnc_val
                ba_imp = bc_val - bnc_val
                # novel fractions
                trc_novel = trc.get('novel_frac', 0.0)
                trnc_novel = trnc.get('novel_frac', 0.0)
                bc_novel = bc.get('novel_frac', 0.0)
                bnc_novel = bnc.get('novel_frac', 0.0)
                rand_novel = rand.get('novel_frac', 0.0)
                writer.writerow([
                    th, cl, direction,
                    trc_val, trnc_val, tr_imp,
                    bc_val, bnc_val, ba_imp,
                    trc_novel, trnc_novel,
                    bc_novel, bnc_novel,
                    rand_novel
                ])
    print(f"Sweep complete. Results written to {args.output_csv}")

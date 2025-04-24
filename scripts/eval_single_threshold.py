#!/usr/bin/env python3
"""
Evaluate model at a single threshold without baseline testing, output CSV with trained metrics.
"""
import argparse
import subprocess
import re
import csv
import os
import torch
import sys

def parse_args():
    parser = argparse.ArgumentParser("Single-threshold eval without baseline")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--class_labels', type=int, nargs='+', default=[1,0])
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample_method', type=str, default='ancestral', choices=['ancestral','ddim'])
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--output_csv', type=str, default='eval_single_threshold.csv')
    return parser.parse_args()

def run_eval(data_dir, checkpoint, threshold, class_label, num_samples, timesteps, device, sample_method, eta):
    cmd = [
        sys.executable, '-u', 'evaluate.py',
        '--data_dir', data_dir,
        '--checkpoint', checkpoint,
        '--baseline_model', checkpoint,  # dummy baseline, ignored
        '--threshold', str(threshold),
        '--class_label', str(class_label),
        '--num_samples', str(num_samples),
        '--timesteps', str(timesteps),
        '--device', device,
        '--sample_method', sample_method,
        '--eta', str(eta)
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    trained_cond = {}
    trained_unc = {}
    cur = None
    pat_tc = re.compile(r"Trained\+conditioned")
    pat_tnc = re.compile(r"Trained\+unconditioned")
    pat_kv = re.compile(r"(\w+):\s*([\-0-9.]+)")
    for line in proc.stdout:
        # stream evaluate.py logs live
        print(f"[eval_single] {line}", end="", flush=True)
        if pat_tc.search(line):
            cur = 'tc'; continue
        if pat_tnc.search(line):
            cur = 'tnc'; continue
        m = pat_kv.search(line)
        if m and cur:
            key, val = m.group(1), float(m.group(2))
            if cur == 'tc':
                trained_cond[key] = val
            elif cur == 'tnc':
                trained_unc[key] = val
    proc.wait()
    return trained_cond, trained_unc

if __name__ == '__main__':
    print("[single] Starting single-threshold evaluation", flush=True)
    args = parse_args()
    print(f"[single] Config: checkpoint={args.checkpoint}, threshold={args.threshold}, classes={args.class_labels}", flush=True)
    write_header = not os.path.isfile(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'threshold','class_label','direction',
                'trained_cond_pct','trained_unc_pct','trained_imp_pct',
                'novel_trained_cond_pct','novel_trained_unc_pct',
                'novel_trained_cond_trans_inv_pct','novel_trained_unc_trans_inv_pct'
            ])
        for cl in args.class_labels:
            direction = 'alive' if cl == 1 else 'dead'
            print(f"[single] Start: threshold={args.threshold}, class={direction}", flush=True)
            trc, trnc = run_eval(
                args.data_dir,
                args.checkpoint,
                args.threshold,
                cl,
                args.num_samples,
                args.timesteps,
                args.device,
                args.sample_method,
                args.eta
            )
            key = 'survived_unknown' if cl == 1 else 'died_out'
            n = args.num_samples
            trc_val = trc.get(key, 0.0)
            trnc_val = trnc.get(key, 0.0)
            trc_pct = trc_val / n * 100.0
            trnc_pct = trnc_val / n * 100.0
            imp_pct = trc_pct - trnc_pct
            novel_trc = trc.get('novel_frac', 0.0) * 100.0
            novel_trnc = trnc.get('novel_frac', 0.0) * 100.0
            # translation-invariant novelty
            novel_trc_ti = trc.get('novel_frac_trans_inv', 0.0) * 100.0
            novel_trnc_ti = trnc.get('novel_frac_trans_inv', 0.0) * 100.0
            writer.writerow([
                 args.threshold,
                 cl,
                 direction,
                 trc_pct,
                 trnc_pct,
                 imp_pct,
                 novel_trc,
                 novel_trnc,
                 novel_trc_ti,
                 novel_trnc_ti
             ])
            print(f"[single] Done: threshold={args.threshold}, class={direction}", flush=True)
    print(f"Single-threshold evaluation complete. Results in {args.output_csv}")

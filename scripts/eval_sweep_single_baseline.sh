#!/usr/bin/env bash
set -e
set -x

# Sweep multiple thresholds with a single baseline and write CSV
cd "$(dirname "$0")"/..

# settings
thresholds=(0.01 0.1 0.3 0.5)
trained_ckpt="checkpoints/model_final_ema.pt"
baseline_ckpt="$trained_ckpt"
out_csv="eval_sweep_single_baseline_ema.csv"

# Use Python sweeper instead of shell parsing
python scripts/eval_sweep.py \
  --data_dir data \
  --checkpoint "$trained_ckpt" \
  --baseline_model "$baseline_ckpt" \
  --thresholds "${thresholds[@]}" \
  --num_samples 64 \
  --timesteps 200 \
  --device cuda \
  --sample_method ancestral \
  --eta 0.0 \
  --output_csv "$out_csv"
echo "=== Sweep Completed: $out_csv ==="
cat "$out_csv"
exit 0

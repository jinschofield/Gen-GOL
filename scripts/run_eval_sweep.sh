#!/usr/bin/env bash
# Helper to run eval_sweep.py over thresholds and class labels
# Copy the specified checkpoint as best_so_far
cp checkpoints/model_42000.pt checkpoints/best_so_far.pt
set -e

python scripts/eval_sweep.py \
  --data_dir ./data \
  --checkpoint checkpoints/best_so_far.pt \
  --baseline_model checkpoints/model_init.pt \
  --thresholds 0.01 0.1 0.3 0.5 \
  --num_samples 64 \
  --timesteps 300 \
  --device cuda \
  --sample_method ancestral \
  --eta 0.0 \
  --output_csv eval_sweep.csv

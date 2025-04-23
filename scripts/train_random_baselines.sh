#!/usr/bin/env bash
set -e

# Abort any ongoing merge and pull latest main
cd "$(dirname "$0")"/..

git merge --abort 2>/dev/null || true
git pull origin main --no-edit

# Train random baseline models per threshold
thresholds=(0.01 0.1 0.3 0.5)
for th in "${thresholds[@]}"; do
  echo "▶ Training random baseline for threshold $th"
  python train.py \
    --random_baseline \
    --random_baseline_samples 20000 \
    --grid_size 32 \
    --epochs 200 \
    --batch_size 64 \
    --save_dir checkpoints/random_baseline_${th}
  echo "✔️ Done threshold $th"
done

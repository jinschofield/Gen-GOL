#!/usr/bin/env bash
set -e
set -x

# Sweep single-threshold evals without baseline
cd "$(dirname "$0")"/..

# settings
thresholds=(0.01 0.1 0.3 0.5)
ckpt="finished_models/adversarial_0.01.pt"
out_csv="eval_sweep_no_baseline_adversarial_0.01.csv"

# write header
printf "%s\n" \
  "threshold,class_label,direction,trained_cond_pct,trained_unc_pct,trained_imp_pct,novel_trained_cond_pct,novel_trained_unc_pct,novel_trained_cond_trans_inv_pct,novel_trained_unc_trans_inv_pct" \
  > "$out_csv"

for th in "${thresholds[@]}"; do
  for cl in 1 0; do
    python scripts/eval_single_threshold.py \
      --data_dir data \
      --checkpoint "$ckpt" \
      --threshold "$th" \
      --class_labels "$cl" \
      --num_samples 64 \
      --timesteps 200 \
      --device cuda \
      --sample_method ancestral \
      --eta 0.0 \
      --output_csv /dev/stdout \
    >> "$out_csv"
  done
done

echo "=== Sweep Completed: $out_csv ==="
cat "$out_csv"

exit 0

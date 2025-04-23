#!/usr/bin/env bash
set -e

# Sweep multiple thresholds with a single baseline and write CSV
cd "$(dirname "$0")"/..

# settings
thresholds=(0.01 0.1 0.3 0.5)
trained_ckpt="checkpoints/best_so_far.pt"
baseline_ckpt="checkpoints/model_init.pt"
out_csv="eval_sweep_single_baseline.csv"

# header
printf "%s\n" \
  "threshold,class_label,direction,trained_cond,trained_unc,trained_imp,baseline_cond,baseline_unc,baseline_imp,novel_trained_cond,novel_trained_unc,novel_baseline_cond,novel_baseline_unc,novel_random" \
  > "$out_csv"

# loop through thresholds and class labels
for th in "${thresholds[@]}"; do
  for cl in 1 0; do
    direction="alive"
    [[ $cl -eq 0 ]] && direction="dead"
    echo "[EVAL] threshold=$th class=$cl ($direction)"
    out=$(python evaluate.py \
      --data_dir data \
      --checkpoint "$trained_ckpt" \
      --baseline_model "$baseline_ckpt" \
      --threshold "$th" \
      --class_label "$cl" \
      --num_samples 64 \
      --timesteps 300 \
      --device cuda \
      --sample_method ancestral \
      --eta 0.0)

    # parse metrics
extract() { echo "$out" | grep -Eo "${1}: [-0-9.]+" | tail -n1 | awk '{print $2}'; }
    key=$( [[ $cl -eq 1 ]] && echo "survived_unknown" || echo "died_out" )
    tc=$(extract "Trained+conditioned.*${key}")
    tnc=$(extract "Trained+unconditioned.*${key}")
    bc=$(extract "Untrained+conditioned.*${key}")
    bnc=$(extract "Untrained+unconditioned.*${key}")
    rand_n=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | tail -n1 | awk '{print $2}')

    # improvements
    imp_t=$(awk "BEGIN {print $tc - $tnc}")
    imp_b=$(awk "BEGIN {print $bc - $bnc}")

    # novel fractions
    ntc=$(extract "Trained+conditioned.*novel_frac")
    ntnc=$(extract "Trained+unconditioned.*novel_frac")
    nbc=$(extract "Untrained+conditioned.*novel_frac")
    nbnc=$(extract "Untrained+unconditioned.*novel_frac")

    # write CSV row
echo "$th,$cl,$direction,$tc,$tnc,$imp_t,$bc,$bnc,$imp_b,$ntc,$ntnc,$nbc,$nbnc,$rand_n" >> "$out_csv"
    echo "[DONE] threshold=$th class=$cl"
  done
done

echo "✅ CSV written to $out_csv"

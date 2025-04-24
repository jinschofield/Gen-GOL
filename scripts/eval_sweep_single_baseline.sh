#!/usr/bin/env bash
set -e
set -x

# Sweep multiple thresholds with a single baseline and write CSV
cd "$(dirname "$0")"/..

# settings
thresholds=(0.01 0.1 0.3 0.5)
trained_ckpt="checkpoints/model_final.pt"
baseline_ckpt="$trained_ckpt"
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
      --timesteps 200 \
      --device cuda \
      --sample_method ancestral \
      --eta 0.0 2>&1)
    echo "----- DEBUG OUTPUT BEGIN -----"
    echo "$out"
    echo "----- DEBUG OUTPUT END -----"

    # parse metrics
    # capture die/survive counts in order for trained_cond, trained_unc, baseline_cond, baseline_unc
    key=$( [[ $cl -eq 1 ]] && echo "survived_unknown" || echo "died_out" )
    vals=( $(echo "$out" | grep -Eo "${key}: [-0-9.]+" | cut -d':' -f2) )
    tc=${vals[0]:-0}
    tnc=${vals[1]:-0}
    bc=${vals[2]:-0}
    bnc=${vals[3]:-0}
    rand_n=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | tail -n1 | cut -d':' -f2 || echo "0")

    # improvements
    imp_t=$(awk -v a="$tc" -v b="$tnc" 'BEGIN {print a - b}')
    imp_b=$(awk -v a="$bc" -v b="$bnc" 'BEGIN {print a - b}')

    # novel fractions
    ntc=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | head -1 | cut -d':' -f2)
    ntnc=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | head -2 | tail -1 | cut -d':' -f2)
    nbc=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | head -3 | tail -1 | cut -d':' -f2)
    nbnc=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | head -4 | tail -1 | cut -d':' -f2)

    # write CSV row
    echo "$th,$cl,$direction,$tc,$tnc,$imp_t,$bc,$bnc,$imp_b,$ntc,$ntnc,$nbc,$nbnc,$rand_n" >> "$out_csv"
    echo "[DONE] threshold=$th class=$cl"
  done
done

echo "✅ CSV written to $out_csv"

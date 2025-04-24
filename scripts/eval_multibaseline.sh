#!/usr/bin/env bash
set -e

# Sweep multiple thresholds with corresponding baselines and write CSV
cd "$(dirname "$0")"/..

thresholds=(0.01 0.1 0.3 0.5)
trained_ckpt="checkpoints/best_so_far.pt"
out_csv="eval_sweep_multibase.csv"

# header
echo "threshold,class_label,direction,trained_cond,trained_unc,trained_imp,baseline_cond,baseline_unc,baseline_imp,novel_trained_cond,novel_trained_unc,novel_baseline_cond,novel_baseline_unc,novel_random,novel_trained_cond_trans_inv,novel_trained_unc_trans_inv,novel_baseline_cond_trans_inv,novel_baseline_unc_trans_inv,novel_random_trans_inv" > "$out_csv"

for th in "${thresholds[@]}"; do
  baseline_model="baseline_models/model_final_${th}.pt"
  for cl in 1 0; do
    direction="alive"
    [[ $cl -eq 0 ]] && direction="dead"
    echo "[EVAL] th=$th, class=$cl ($direction)"
    # run evaluate.py
    out=$(python evaluate.py \
      --data_dir data \
      --checkpoint "$trained_ckpt" \
      --baseline_model "$baseline_model" \
      --threshold "$th" \
      --class_label "$cl" \
      --num_samples 64 \
      --timesteps 300 \
      --device cuda \
      --sample_method ancestral \
      --eta 0.0)
    # extract
gc() { echo "$out" | grep -Eo "${1}: [-0-9.]+"; }
    key_tc=$(gc survived_unknown | head -1 | cut -d':' -f2)
    key_tnc=$(gc survived_unknown | tail -1 | cut -d':' -f2)
    key_bc=$(gc died_out | head -1 | cut -d':' -f2)
    key_bnc=$(gc died_out | tail -1 | cut -d':' -f2)
    rk=$(echo "$out" | grep -Eo "novel_frac: [-0-9.]+" | head -1 | cut -d':' -f2)
    # improvements
    imp_t=$(awk "BEGIN {print $key_tc - $key_tnc}")
    imp_b=$(awk "BEGIN {print $key_bc - $key_bnc}")
    # novel fractions from each section
    d1=$(echo "$out" | grep -A1 "Trained+conditioned" | tail -1 | grep -Eo "novel_frac: [-0-9.]+" | cut -d':' -f2)
    d2=$(echo "$out" | grep -A1 "Trained+unconditioned" | tail -1 | grep -Eo "novel_frac: [-0-9.]+" | cut -d':' -f2)
    d3=$(echo "$out" | grep -A1 "Untrained+conditioned" | tail -1 | grep -Eo "novel_frac: [-0-9.]+" | cut -d':' -f2)
    d4=$(echo "$out" | grep -A1 "Untrained+unconditioned" | tail -1 | grep -Eo "novel_frac: [-0-9.]+" | cut -d':' -f2)
    novel_r=$(gc novel_frac)
    # translation-invariant novel fractions
    d1_ti=$(echo "$out" | grep -A1 "Trained+conditioned" | tail -1 | grep -Eo "novel_frac_trans_inv: [-0-9.]+" | cut -d':' -f2)
    d2_ti=$(echo "$out" | grep -A1 "Trained+unconditioned" | tail -1 | grep -Eo "novel_frac_trans_inv: [-0-9.]+" | cut -d':' -f2)
    d3_ti=$(echo "$out" | grep -A1 "Untrained+conditioned" | tail -1 | grep -Eo "novel_frac_trans_inv: [-0-9.]+" | cut -d':' -f2)
    d4_ti=$(echo "$out" | grep -A1 "Untrained+unconditioned" | tail -1 | grep -Eo "novel_frac_trans_inv: [-0-9.]+" | cut -d':' -f2)
    novel_r_ti=$(gc novel_frac_trans_inv)
    # append to CSV
    echo "$th,$cl,$direction,$key_tc,$key_tnc,$imp_t,$key_bc,$key_bnc,$imp_b,$d1,$d2,$d3,$d4,$rk,$d1_ti,$d2_ti,$d3_ti,$d4_ti,$novel_r_ti" >> "$out_csv"
    echo "[DONE] th=$th, class=$cl"
  done
done

echo "✅ Multi‑baseline CSV written to $out_csv"

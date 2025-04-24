#!/usr/bin/env python3
import csv, os, subprocess

"""
Python wrapper to sweep a single threshold without baseline testing.
"""

def main():
    thresholds = [0.3]
    ckpt = "finished_models/adversarial_0.01.pt"
    out_csv = "finished_models/eval_sweep_no_baseline_adversarial_0.01.csv"

    header = [
        "threshold","class_label","direction",
        "trained_cond_pct","trained_unc_pct","trained_imp_pct",
        "novel_trained_cond_pct","novel_trained_unc_pct",
        "novel_trained_cond_trans_inv_pct","novel_trained_unc_trans_inv_pct"
    ]
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for th in thresholds:
            for cl in (1, 0):
                direction = 'alive' if cl == 1 else 'dead'
                cmd = [
                    "python3", "scripts/eval_single_threshold.py",
                    "--data_dir", "data",
                    "--checkpoint", ckpt,
                    "--threshold", str(th),
                    "--class_labels", str(cl),
                    "--num_samples", "64",
                    "--timesteps", "200",
                    "--device", "cuda",
                    "--sample_method", "ancestral",
                    "--eta", "0.0",
                    "--output_csv", "/dev/stdout"
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                for line in proc.stdout.splitlines():
                    if line.startswith("threshold"):  # skip header
                        continue
                    writer.writerow(line.split(","))

    print(f"Sweep complete → {out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import csv, os, subprocess
import sys  # use same interpreter and unbuffered mode

"""
Python wrapper to sweep a single threshold without baseline testing.
"""

def main():
    print("[sweep] Starting no-baseline sweep", flush=True)
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
                print(f"[sweep] Start: threshold={th}, class={direction}", flush=True)
                cmd = [
                    sys.executable, "-u", "scripts/eval_single_threshold.py",
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
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    print("[eval] " + line, end="", flush=True)
                    if line.startswith("threshold"):  # skip header
                        continue
                    writer.writerow(line.strip().split(","))
                proc.wait()
                print(f"[sweep] Done: threshold={th}, class={direction}", flush=True)

    print(f"Sweep complete → {out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 2 - Step 1:
Label 32×32 training dataset patterns (.npy) with life‐type categories using simulation.
Outputs CSV `phase2_training_labels_32x32.csv` in this folder with columns: [filepath, category].
"""
import os
import glob
import csv
import numpy as np
import sys
# ensure repo root & utils on module path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'utils'))  # for gol_simulator
from utils.gol_simulator import simulate
from phase_1_classification.utils.detectors import detect_gliders, detect_spaceships


def classify_grid(arr, timesteps=200):
    history = simulate(arr, steps=timesteps)
    # died out
    if history[-1].sum() == 0:
        return 'died_out'
    # detect period of final state
    last = history[-1]
    per = None
    for p in range(1, len(history)):
        if np.array_equal(history[-1-p], last):
            per = p
            break
    if per == 1:
        cat = 'still_life'
    elif per and per > 1:
        cat = f'oscillator_period_{per}'
    else:
        cat = 'others'
    # override if glider or spaceship detected in final frame
    last_frame = history[-1]
    if detect_gliders(last_frame):
        cat = 'glider'
    elif detect_spaceships(last_frame):
        cat = 'spaceship'
    return cat


def main():
    data_dir = 'data'
    output_csv = 'phase2_training_labels_32x32.csv'
    files = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'category'])
        for p in files:
            arr = np.load(p).astype(np.uint8)
            cat = classify_grid(arr)
            writer.writerow([p, cat])
    print(f"Labeled {len(files)} patterns. Saved to {output_csv}.")


if __name__ == '__main__':
    main()

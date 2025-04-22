import os
import sys
# ensure project root is on sys.path for sibling imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils.gol_simulator import simulate

if __name__ == '__main__':
    data_dir = os.path.dirname(__file__)
    os.makedirs(data_dir, exist_ok=True)
    N = 32  # grid size
    max_samples = 1000
    samples = 0
    tries = 0
    STEPS = 200  # number of simulation steps
    TRIES_MULT = 100  # max attempts multiplier
    # Generate patterns by random init + simulation filtering
    while samples < max_samples and tries < max_samples * TRIES_MULT:
        x0 = (np.random.rand(N, N) < 0.3).astype(np.uint8)
        history = simulate(x0, steps=STEPS)
        # if simulation repeated early, save the last unique state
        if len(history) < STEPS:
            rep_state = history[-1]
            if rep_state.sum() > 0:
                np.save(os.path.join(data_dir, f'pattern_{samples}.npy'), rep_state)
                samples += 1
        tries += 1
    print(f"Generated {samples} patterns (out of {max_samples}) in {data_dir} after {tries} attempts")

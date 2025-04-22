import os
import sys
# ensure project root is on sys.path for sibling imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils.gol_simulator import simulate
from utils.metrics import detect_period

if __name__ == '__main__':
    data_dir = os.path.dirname(__file__)
    os.makedirs(data_dir, exist_ok=True)
    N = 32  # grid size
    max_samples = 1000
    samples = 0
    tries = 0
    # Generate patterns by random init + simulation filtering
    while samples < max_samples and tries < max_samples * 5:
        x0 = (np.random.rand(N, N) < 0.3).astype(np.uint8)
        history = simulate(x0, steps=20)
        per = detect_period(history)
        if per is not None and per >= 1:
            # save the initial pattern
            np.save(os.path.join(data_dir, f'pattern_{samples}.npy'), history[0])
            samples += 1
        tries += 1
    print(f"Generated {samples} patterns in {data_dir}")

"""
Metrics for evaluating generated GoL patterns: oscillator rate, novelty.
"""
import numpy as np
import torch
from collections import defaultdict
from .gol_simulator import simulate


def detect_period(history):
    """Given a history list of grids, return period (1 for still life, >1 if oscillator), or None if no repeat."""
    if len(history) <= 1:
        return None
    first = history[0]
    for i in range(1, len(history)):
        if np.array_equal(history[i], first):
            return i
    return None


def evaluate_samples(samples, train_patterns, max_steps=50):
    """
    Args:
        samples: torch.Tensor (N,1,H,W) continuous outputs; threshold at 0.5 for binarization.
        train_patterns: list of np.ndarray (H,W) training grids for novelty check.
    Returns:
        dict with counts and novel fraction.
    """
    N = samples.shape[0]
    grids = (samples.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
    results = defaultdict(int)
    novel = 0
    # Precompute train set variants
    train_set = set()
    for pat in train_patterns:
        # rotations+flips
        for k in range(4):
            r = np.rot90(pat, k)
            train_set.add(r.tobytes())
            train_set.add(np.fliplr(r).tobytes())
    for g in grids:
        hist = simulate(g, steps=max_steps)
        per = detect_period(hist)
        if per is None:
            results['died_out'] += 1
        elif per == 1:
            results['still_life'] += 1
        else:
            results[f'oscillator_p{per}'] += 1
        # novelty
        if g.tobytes() not in train_set:
            novel += 1
    results['total'] = N
    results['novel_frac'] = novel / N
    return results

"""
Metrics for evaluating generated GoL patterns: oscillator rate, novelty.
"""
import numpy as np
import torch
from collections import defaultdict
from .gol_simulator import simulate


def detect_period(history):
    """Given a history list of grids, return period (1 for still life, >1 if oscillator), or None if no repeat."""
    # detect period relative to final state
    if len(history) <= 1:
        return None
    last = history[-1]
    for p in range(1, len(history)):
        if np.array_equal(history[-1-p], last):
            return p
    return None


def evaluate_samples(samples, train_patterns, max_steps=50, threshold=0.5):
    """
    Args:
        samples: torch.Tensor (N,1,H,W) continuous outputs; threshold for binarization.
        train_patterns: list of np.ndarray (H,W) training grids for novelty check.
        threshold: float, cutoff for binarization.
    Returns:
        dict with counts and both novelty fractions (rot/flip-only and translation-invariant).
    """
    N = samples.shape[0]
    grids = (samples.squeeze(1).cpu().numpy() > threshold).astype(np.uint8)
    results = defaultdict(int)
    # novelty counters: rotation+flip only and translation-invariant
    novel_rf = 0
    novel_ti = 0
    # build rotation+flip set
    rf_set = set()
    for pat in train_patterns:
        for k in range(4):
            r = np.rot90(pat, k)
            rf_set.add(r.tobytes())
            rf_set.add(np.fliplr(r).tobytes())
    for g in grids:
        b = g.tobytes()
        if b not in rf_set:
            novel_rf += 1
        # translation-invariant novelty disabled for debugging
        # if b not in ti_set:
        #     novel_ti += 1
        hist = simulate(g, steps=max_steps)
        per = detect_period(hist)
        if hist[-1].sum() == 0:
            results['died_out'] += 1
        else:
            if per == 1:
                results['still_life'] += 1
            elif per and per > 1:
                results[f'oscillator_p{per}'] += 1
            else:
                results['survived_unknown'] += 1
    results['total'] = N
    results['novel_frac'] = novel_rf / N
    # translation-invariant novel fraction disabled
    results['novel_frac_trans_inv'] = 0.0
    return results

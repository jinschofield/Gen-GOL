"""
Game of Life simulator for evaluating generated patterns.
"""
import numpy as np

def step(grid):
    """Perform one GoL update with toroidal boundary."""
    H, W = grid.shape
    # Count neighbors
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1,0,1) for j in (-1,0,1)
                    if not (i==0 and j==0))
    # Apply rules
    birth = (neighbors == 3) & (grid == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    return (birth | survive).astype(np.uint8)

def simulate(grid, steps=50):
    """Simulate for up to given steps or until extinction. Returns history."""
    history = []
    g = grid.copy()
    for _ in range(steps):
        history.append(g.copy())
        if g.sum() == 0:
            break
        g = step(g)
    return history

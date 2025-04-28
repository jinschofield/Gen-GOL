#!/usr/bin/env python3
"""
Placeholder detection routines for various Game of Life patterns.
"""

import numpy as np

# A basic glider pattern (3x3) offsets for detection
GLIDER_OFFSETS = [
    [(0,2),(1,0),(1,2),(2,1),(2,2)],  # one glider orientation
    # TODO: add other orientations and spaceship patterns
]

# Placeholder spaceship patterns for small ships like light-weight spaceship
SPACESHIP_OFFSETS = []  # TODO: populate with known offsets


def detect_gliders(grid):
    """
    Simple detection: scan the grid for the basic glider pattern.
    :param grid: 2D numpy array of 0/1
    """
    H, W = grid.shape
    for offsets in GLIDER_OFFSETS:
        h = len(grid)
        w = len(grid[0])
        for i in range(h - 2):
            for j in range(w - 2):
                if all(grid[i + di, j + dj] == 1 for di, dj in offsets):
                    return True
    return False


def detect_spaceships(grid):
    """
    Placeholder always returns False until patterns are added.
    """
    return False

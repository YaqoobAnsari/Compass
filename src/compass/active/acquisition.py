"""
Acquisition strategies for active crowdsensing.

Each strategy maps (current reconstruction, observed-coverage, free mask) -> a
TARGET pixel for the next walk. The closed loop then routes a walkable trajectory
to that target. Strategies share an interface so the same loop runs all of them
for a matched-budget comparison.

  * max_variance    — go where the reconstructor is most uncertain (ours).
  * space_filling   — go farthest from all observations (geometric baseline).
  * coverage        — go to the lowest observation-density region.
  * random          — random free pixel (lower bound).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _argmax_free(score: np.ndarray, free: np.ndarray) -> Tuple[int, int]:
    masked = np.where(free, score, -np.inf)
    r, c = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(r), int(c)


def max_variance(std: np.ndarray, observed: np.ndarray, free: np.ndarray, rng) -> Tuple[int, int]:
    # smooth the uncertainty so we target REGIONS, not single noisy pixels
    s = gaussian_filter(np.where(free, std, 0.0), sigma=3.0)
    return _argmax_free(s, free)


def space_filling(std: np.ndarray, observed: np.ndarray, free: np.ndarray, rng) -> Tuple[int, int]:
    # distance to nearest observation; farthest free pixel wins
    dist = distance_transform_edt(~observed)
    return _argmax_free(dist, free)


def coverage(std: np.ndarray, observed: np.ndarray, free: np.ndarray, rng) -> Tuple[int, int]:
    dens = gaussian_filter(observed.astype(float), sigma=8.0)
    return _argmax_free(-dens, free)  # lowest density


def random_target(std: np.ndarray, observed: np.ndarray, free: np.ndarray, rng) -> Tuple[int, int]:
    coords = np.argwhere(free)
    r, c = coords[rng.integers(len(coords))]
    return int(r), int(c)


STRATEGIES = {
    "max_variance": max_variance,
    "space_filling": space_filling,
    "coverage": coverage,
    "random": random_target,
}

"""
Walkable-graph routing for active crowdsensing.

The acquisition map says WHERE we want a measurement; a real crowd worker can
only get there by WALKING ON STREETS. This module plans a budgeted walking
trajectory from the worker's current position to a target pixel using A* over
the free-space mask, then resamples it into timed points — so collection follows
walkable paths (the differentiator vs free-grid acquisition).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..data.trajectory import Trajectory, _astar, _resample_polyline


def random_free_pixel(free: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    coords = np.argwhere(free)
    r, c = coords[rng.integers(len(coords))]
    return int(r), int(c)


def walk_to(
    free: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    n_points: int = 80,
    speed_px_s: float = 1.4,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Trajectory]:
    """Plan an on-street walk start->goal (A*), resampled to n_points. None if unreachable."""
    path = _astar(free, tuple(int(v) for v in start_rc), tuple(int(v) for v in goal_rc))
    if path is None or len(path) < 2:
        return None
    tr = _resample_polyline(np.array(path, float), n_points, speed_px_s)
    tr.kind = "guided_walk"
    return tr


def nearest_reachable(
    free: np.ndarray, target_rc: Tuple[int, int], from_rc: Tuple[int, int]
) -> Tuple[int, int]:
    """Snap a (possibly building) target to the nearest free pixel."""
    if free[int(target_rc[0]), int(target_rc[1])]:
        return int(target_rc[0]), int(target_rc[1])
    coords = np.argwhere(free)
    d = np.hypot(coords[:, 0] - target_rc[0], coords[:, 1] - target_rc[1])
    r, c = coords[int(np.argmin(d))]
    return int(r), int(c)

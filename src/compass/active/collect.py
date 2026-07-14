"""
Turn trajectories into accumulated Observations by sampling the ground-truth
field along the walked path (the crowdsensed measurement act). On RadioMapSeer
the GT is the dense ray-traced map; on real data this is replaced by the actual
recorded RSS. Optional realistic noise via compass.data.noise.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..data.trajectory import Trajectory
from ..recon.base import Observations


def collect_along_trajectories(
    trajs: List[Trajectory],
    gt_dbm: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.0,
    shadow_field: Optional[np.ndarray] = None,
) -> Observations:
    """Sample GT dBm at each (rounded) trajectory pixel; optional measurement noise.

    Args:
        trajs: walked trajectories (rows, cols in pixels).
        gt_dbm: (H,W) ground-truth gain map in dBm.
        noise_std: i.i.d. Gaussian measurement noise (dB).
        shadow_field: optional (H,W) correlated shadowing to add (dB).
    """
    H, W = gt_dbm.shape
    rows, cols, vals = [], [], []
    for tr in trajs:
        r = np.clip(np.round(tr.rows).astype(int), 0, H - 1)
        c = np.clip(np.round(tr.cols).astype(int), 0, W - 1)
        v = gt_dbm[r, c].astype(float)
        if shadow_field is not None:
            v = v + shadow_field[r, c]
        if noise_std > 0 and rng is not None:
            v = v + rng.normal(0, noise_std, size=v.shape)
        rows.append(r)
        cols.append(c)
        vals.append(v)
    if not rows:
        return Observations.empty()
    return Observations(
        np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    )


def observed_mask(obs: Observations, shape) -> np.ndarray:
    """Boolean (H,W) mask of pixels that carry at least one observation."""
    m = np.zeros(shape, dtype=bool)
    if len(obs):
        m[obs.rows, obs.cols] = True
    return m

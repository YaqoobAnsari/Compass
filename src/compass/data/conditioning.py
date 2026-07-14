"""
Assemble model inputs from trajectories + a radio map.

Produces BOTH representations so we can compare them head-to-head:
  * RASTER channels (order-blind): ``sparse_rss`` (normalised), ``mask``,
    ``coverage`` — the v1-style conditioning.
  * ORDERED sequence features (per point): time, dt, position, speed, heading
    (sin/cos), distance-to-TX, radial/tangential velocity, and measured RSS —
    the information a sequence encoder needs (Pillar 1).

The realistic noise model (correlated shadowing + device effects) is applied
along each trajectory, sharing one shadowing field per map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from .conventions import MAP_SIZE, PATHLOSS_MIN_DBM, PATHLOSS_RANGE_DB, RESOLUTION_M
from .noise import MeasurementNoise
from .trajectory import Trajectory, TrajectorySampler

SEQ_FEATURES = [
    "t", "dt", "row_norm", "col_norm", "speed",
    "head_sin", "head_cos", "dist_tx_norm", "v_radial", "v_tangential", "rss_norm",
]


def dbm_to_signed_unit(dbm: np.ndarray) -> np.ndarray:
    """dBm -> [-1, 1] (matches the radio_map target normalisation)."""
    v01 = (np.asarray(dbm, dtype=np.float32) - PATHLOSS_MIN_DBM) / PATHLOSS_RANGE_DB
    return np.clip(v01 * 2.0 - 1.0, -1.0, 1.0)


def sample_clean_dbm(radio_map_dbm: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    ri = np.clip(np.round(rows).astype(int), 0, radio_map_dbm.shape[0] - 1)
    ci = np.clip(np.round(cols).astype(int), 0, radio_map_dbm.shape[1] - 1)
    return radio_map_dbm[ri, ci].astype(np.float32)


def trajectory_features(
    traj: Trajectory, tx_rowcol, measured_dbm: np.ndarray,
    map_size: int = MAP_SIZE,
) -> np.ndarray:
    """(N, len(SEQ_FEATURES)) ordered per-point feature array for one trajectory."""
    r, c, t = traj.rows, traj.cols, traj.t
    dt = np.gradient(t)
    dt[np.abs(dt) < 1e-9] = 1e-6
    vr, vc = np.gradient(r) / dt, np.gradient(c) / dt
    speed = np.hypot(vr, vc)
    heading = np.arctan2(vr, vc)
    tx_r, tx_c = float(tx_rowcol[0]), float(tx_rowcol[1])
    to_r, to_c = tx_r - r, tx_c - c
    dist = np.hypot(to_r, to_c) + 1e-6
    rad_r, rad_c = to_r / dist, to_c / dist
    v_radial = vr * rad_r + vc * rad_c              # +ve = approaching TX
    v_tangential = vc * rad_r - vr * rad_c          # signed tangential component
    return np.stack(
        [
            t,
            np.concatenate([[0.0], np.diff(t)]),
            r / map_size,
            c / map_size,
            speed,
            np.sin(heading),
            np.cos(heading),
            dist / map_size,
            v_radial,
            v_tangential,
            dbm_to_signed_unit(measured_dbm),
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class Conditioning:
    sparse_rss: np.ndarray          # (H,W) normalised [-1,1], 0 where unobserved
    mask: np.ndarray                # (H,W) {0,1}
    coverage: np.ndarray            # (H,W) [0,1]
    sequences: List[np.ndarray]     # list of (N_k, F) per trajectory
    measured_dbm: List[np.ndarray]  # list of (N_k,) measured RSS per trajectory
    trajectories: List[Trajectory] = field(default_factory=list)
    shadow_field: Optional[np.ndarray] = None

    @property
    def coverage_fraction(self) -> float:
        return float(self.mask.mean())


def make_conditioning(
    building_map: np.ndarray,
    radio_map_dbm: np.ndarray,
    tx_rowcol,
    sampler: TrajectorySampler,
    noise: Optional[MeasurementNoise] = None,
    rng: Optional[np.random.Generator] = None,
    k: int = 3,
    n_points: int = 100,
    coverage_sigma: float = 5.0,
    shadow_field: Optional[np.ndarray] = None,
) -> Conditioning:
    """Build raster + sequence conditioning for one (map, tx), with realistic noise."""
    rng = rng or np.random.default_rng()
    H, W = radio_map_dbm.shape
    trajs = sampler.sample_many(building_map, k=k, n_points=n_points)

    if noise is not None and shadow_field is None:
        shadow_field = noise.new_field((H, W), rng=rng)

    sparse = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), np.float32)
    sequences, measured_list = [], []
    for tr in trajs:
        clean = sample_clean_dbm(radio_map_dbm, tr.rows, tr.cols)
        if noise is not None:
            meas = noise.apply(clean, tr.rows, tr.cols, shadow_field, rng)
        else:
            meas = clean
        measured_list.append(meas)
        sequences.append(trajectory_features(tr, tx_rowcol, meas))
        ri = np.clip(np.round(tr.rows).astype(int), 0, H - 1)
        ci = np.clip(np.round(tr.cols).astype(int), 0, W - 1)
        sparse[ri, ci] = dbm_to_signed_unit(meas)
        mask[ri, ci] = 1.0

    coverage = gaussian_filter(mask, sigma=coverage_sigma).astype(np.float32)
    if coverage.max() > 0:
        coverage = coverage / coverage.max()
    return Conditioning(sparse, mask, coverage, sequences, measured_list, trajs, shadow_field)

"""The active loop must run, reduce error as budget grows, and route on streets."""

from __future__ import annotations

import numpy as np
import pytest

from compass.active import STRATEGIES, run_active_loop, walk_to
from compass.recon import GPReconstructor


def _toy_field(H=64, W=64, seed=0):
    """A smooth dBm-like field on an all-free grid (no buildings) for fast tests."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((H, W)), sigma=10) * 25 - 110
    free = np.ones((H, W), dtype=bool)
    return f.astype(np.float32), free


def test_walk_to_stays_on_free():
    _, free = _toy_field()
    free[20:40, 20:40] = False  # a building block
    tr = walk_to(free, (5, 5), (60, 60), n_points=50)
    assert tr is not None
    ri = np.clip(np.round(tr.rows).astype(int), 0, 63)
    ci = np.clip(np.round(tr.cols).astype(int), 0, 63)
    assert free[ri, ci].mean() >= 0.95


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_loop_runs_all_strategies(strategy):
    gt, free = _toy_field()
    rec = GPReconstructor(length_scale=10.0, noise_std=0.5)
    res = run_active_loop(rec, gt, free, strategy, n_rounds=4, points_per_walk=30, noise_std=0.5, seed=1)
    assert len(res.budgets) == 5  # n_rounds + final
    assert res.budgets[-1] > res.budgets[0]  # budget grows
    assert np.all(np.isfinite(res.rmse_free_unobs))


def test_error_decreases_with_budget():
    gt, free = _toy_field()
    rec = GPReconstructor(length_scale=10.0, noise_std=0.5)
    res = run_active_loop(rec, gt, free, "max_variance", n_rounds=10, points_per_walk=40, noise_std=0.5, seed=2)
    # final error should be below the first-round error on a smooth field
    assert res.rmse_free_unobs[-1] < res.rmse_free_unobs[0]

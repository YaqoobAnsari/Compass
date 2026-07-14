"""Discrete-pool active sensing on real geometry: GP variance + acquisition sanity."""

from __future__ import annotations

import numpy as np

from compass.realdata.active import STRATEGIES, active_cell, curve_at_budgets, gp_predict


def _field(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 40, (n, 2))
    src = np.array([10.0, 30.0])
    rss = -50 - 20 * np.log10(np.linalg.norm(pos - src, axis=1) + 1) + rng.normal(0, 0.5, n)
    return pos, rss


def test_gp_variance_lower_near_observations():
    pos, rss = _field()
    mean, std = gp_predict(pos[:10], rss[:10], pos, length_scale=8.0)
    # std at a held-out point far from all obs should exceed std near an obs point
    near = std[:10].mean()
    assert np.isfinite(std).all() and near < std[10:].mean() + 1e-6


def test_active_curve_decreases_overall():
    pos, rss = _field(n=45)
    curve = dict(active_cell(pos, rss, "max_variance", n_seed=4, seed=0, length_scale=8.0))
    budgets = sorted(curve)
    # more measurements should reduce error on average (compare first third vs last third)
    early = np.mean([curve[b] for b in budgets[:len(budgets) // 3]])
    late = np.mean([curve[b] for b in budgets[-len(budgets) // 3:]])
    assert late < early


def test_all_strategies_run():
    pos, rss = _field()
    for s in STRATEGIES:
        c = curve_at_budgets(pos, rss, s, budgets=[4, 8, 12], n_rep=3, length_scale=8.0)
        assert all(len(c[b]) > 0 for b in [4, 8, 12])

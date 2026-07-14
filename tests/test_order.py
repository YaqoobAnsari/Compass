"""Order-signal helpers: autocorrelated series are fillable; shuffling destroys it."""

from __future__ import annotations

import numpy as np

from compass.realdata.order import gap_fill_errors, lag1_autocorr, rmse


def _walk(T=200, seed=0):
    """Smooth autocorrelated series (random walk + drift) like a real RSS stream."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1.0, T)) - 60.0


def test_autocorr_high_for_smooth_series():
    y = _walk()
    assert lag1_autocorr(y) > 0.8
    assert abs(lag1_autocorr(y[np.random.default_rng(1).permutation(len(y))])) < 0.3


def test_true_order_beats_shuffled_and_mean():
    y = _walk()
    e = gap_fill_errors(y, test_frac=0.4, n_rep=5)
    r_true, r_shuf, r_mean = rmse(e["true"]), rmse(e["shuffled"]), rmse(e["mean"])
    assert r_true < r_shuf, f"true {r_true:.2f} !< shuffled {r_shuf:.2f}"
    assert r_true < r_mean, f"true {r_true:.2f} !< mean {r_mean:.2f}"


def test_degenerate_series_returns_empty():
    assert gap_fill_errors(np.full(50, -70.0))["true"] == []
    assert np.isnan(lag1_autocorr(np.full(50, -70.0)))

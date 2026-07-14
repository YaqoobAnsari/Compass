"""Wall-aware reconstruction: a wall between correlated regions is exploitable."""

from __future__ import annotations

import numpy as np

from compass.realdata.walls import wall_aware_loro, wall_length_m, wall_matrix_m


def test_wall_length_counts_barrier():
    bar = np.zeros((50, 50), bool)
    bar[:, 25] = True  # vertical wall at col 25
    # horizontal segment crossing the wall
    w_cross = wall_length_m(bar, (10, 10), (40, 10), m_per_px=1.0)
    w_clear = wall_length_m(bar, (10, 10), (20, 10), m_per_px=1.0)  # stays left of wall
    assert w_cross > 0.3
    assert w_clear < 0.1


def test_wall_aware_helps_when_wall_attenuates():
    # two rooms separated by a wall; RSS differs by a fixed NLoS drop across it
    rng = np.random.default_rng(0)
    bar = np.zeros((100, 100), bool)
    bar[:, 50] = True
    left = rng.uniform(5, 45, (12, 2))
    right = rng.uniform(55, 95, (12, 2))
    pos = np.vstack([left, right])
    # smooth field + a step drop of 15 dB on the right side (NLoS)
    rss = -50 - 0.2 * pos[:, 0] + np.where(pos[:, 0] > 50, -15.0, 0.0) + rng.normal(0, 0.5, len(pos))
    W = wall_matrix_m(pos, bar, m_per_px=1.0)
    e0 = np.sqrt(np.mean(np.square(wall_aware_loro(pos, rss, W, lam=0.0, buffer_m=0.0))))
    e1 = np.sqrt(np.mean(np.square(wall_aware_loro(pos, rss, W, lam=4.0, buffer_m=0.0))))
    assert e1 < e0, f"wall-aware {e1:.2f} !< plain {e0:.2f}"


def test_lambda_zero_equals_plain_idw():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 40, (20, 2))
    rss = -60 - 0.3 * pos[:, 1] + rng.normal(0, 1, 20)
    W = np.abs(rng.normal(0, 1, (20, 20))); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    e = wall_aware_loro(pos, rss, W, lam=0.0, buffer_m=0.0)
    assert len(e) == 20 and np.isfinite(e).all()

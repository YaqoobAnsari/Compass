"""Source estimation recovers a clean point source and rejects a flat/booster field."""

from __future__ import annotations

import numpy as np

from compass.realdata.source import M_PER_PX, log_distance, reliability


def _point_source(src_px=(200.0, 300.0), n=40, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(50, 500, (n, 2))
    d_m = np.linalg.norm((pos - src_px) * M_PER_PX, axis=1) + 1.0
    rss = -40 - 30 * np.log10(d_m) + rng.normal(0, noise, n)  # n=3 pathloss
    return pos, rss, np.array(src_px)


def test_log_distance_recovers_point_source():
    pos, rss, src = _point_source()
    est, b, r2 = log_distance(pos, rss)
    err_m = float(np.hypot(*(est - src)) * M_PER_PX)
    assert b < 0 and r2 > 0.7
    assert err_m < 8.0, f"source error {err_m:.1f} m"


def test_reliability_trusts_clean_source():
    pos, rss, _ = _point_source(noise=0.5)
    rel = reliability(pos, rss)
    assert rel["trustworthy"]
    assert rel["pathloss_n"] > 1.5


def test_reliability_rejects_flat_field():
    rng = np.random.default_rng(0)
    pos = rng.uniform(50, 500, (40, 2))
    rss = -70 + rng.normal(0, 3, 40)  # no spatial decay -> booster/distributed
    rel = reliability(pos, rss)
    assert not rel["trustworthy"]  # low R2 -> not trusted

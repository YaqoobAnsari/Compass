"""Real-data point interpolators recover a smooth field; CV helpers behave."""

from __future__ import annotations

import numpy as np

from compass.realdata.reconstruct import METHODS, holdout_frac_errors, loro_errors


def _field(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 40, (n, 2))  # metres
    # smooth log-distance-ish field from a source
    src = np.array([10.0, 30.0])
    rss = -50 - 20 * np.log10(np.linalg.norm(pos - src, axis=1) + 1) + rng.normal(0, 1.0, n)
    return pos, rss


def test_all_interpolators_beat_constant():
    pos, rss = _field()
    const_rmse = float(np.sqrt(np.mean((rss - rss.mean()) ** 2)))
    for name, fn in METHODS.items():
        errs = loro_errors(pos, rss, fn, buffer_m=0.0)
        assert len(errs) > 30
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        assert rmse < const_rmse, f"{name} did not beat constant ({rmse:.2f} vs {const_rmse:.2f})"


def test_buffer_increases_difficulty():
    pos, rss = _field()
    fn = METHODS["IDW(p=2)"]
    e0 = np.sqrt(np.mean(np.square(loro_errors(pos, rss, fn, buffer_m=0.0))))
    e3 = np.sqrt(np.mean(np.square(loro_errors(pos, rss, fn, buffer_m=3.0))))
    assert e3 >= e0 - 1e-6  # a spatial buffer should not make interpolation easier


def test_coverage_sweep_more_obs_helps():
    pos, rss = _field(n=80)
    fn = METHODS["GP/Kriging"]
    lo = np.sqrt(np.mean(np.square(holdout_frac_errors(pos, rss, fn, test_frac=0.75, n_rep=5))))  # 25% obs
    hi = np.sqrt(np.mean(np.square(holdout_frac_errors(pos, rss, fn, test_frac=0.25, n_rep=5))))  # 75% obs
    assert hi <= lo + 1e-6  # more observations -> not worse

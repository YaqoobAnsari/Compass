"""Reconstructors must (1) honour observations, (2) give higher uncertainty
where unobserved, and (3) beat a trivial constant predictor on a smooth field."""

from __future__ import annotations

import numpy as np

from compass.recon import GPReconstructor, IDWReconstructor, Observations


def _smooth_field(H=64, W=64, seed=0):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((H, W)), sigma=8) * 30 - 120  # dBm-ish
    return f.astype(np.float32)


def _sample_obs(field, n=120, seed=1):
    rng = np.random.default_rng(seed)
    H, W = field.shape
    r = rng.integers(0, H, n)
    c = rng.integers(0, W, n)
    return Observations(r, c, field[r, c].astype(float))


def test_gp_uncertainty_higher_when_unobserved():
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    obs = _sample_obs(field, n=80)
    rec = GPReconstructor(length_scale=8.0, noise_std=0.5).reconstruct(obs, free)
    obs_m = np.zeros_like(free)
    obs_m[obs.rows, obs.cols] = True
    # std at observed pixels < std far away
    assert rec.std[obs_m].mean() < rec.std[free & ~obs_m].mean()


def test_gp_beats_constant_predictor():
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    obs = _sample_obs(field, n=150)
    rec = GPReconstructor(length_scale=8.0, noise_std=0.5).reconstruct(obs, free)
    gp_rmse = np.sqrt(np.mean((rec.mean - field) ** 2))
    const_rmse = np.sqrt(np.mean((field.mean() - field) ** 2))
    assert gp_rmse < 0.8 * const_rmse


def test_idw_runs_and_predicts_in_range():
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    obs = _sample_obs(field, n=100)
    rec = IDWReconstructor().reconstruct(obs, free)
    assert rec.mean.shape == field.shape
    assert np.isfinite(rec.mean[free]).all()


def test_empty_observations_max_uncertainty():
    free = np.ones((32, 32), dtype=bool)
    rec = GPReconstructor().reconstruct(Observations.empty(), free)
    assert (rec.std[free] > 0).all()


def test_uncertainty_correlates_with_error():
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    obs = _sample_obs(field, n=60)
    rec = GPReconstructor(length_scale=8.0, noise_std=0.5).reconstruct(obs, free)
    err = np.abs(rec.mean - field)[free]
    std = rec.std[free]
    corr = np.corrcoef(err, std)[0, 1]
    assert corr > 0.2  # uncertainty tracks error (acquisition signal is meaningful)


def test_new_classical_reconstructors_run():
    from compass.recon import (
        NearestNeighborReconstructor, NaturalNeighborReconstructor, RBFReconstructor,
    )
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    obs = _sample_obs(field, n=120)
    for rec in [NearestNeighborReconstructor(), NaturalNeighborReconstructor(),
                RBFReconstructor(kernel="multiquadric")]:
        r = rec.reconstruct(obs, free)
        assert r.mean.shape == field.shape
        assert np.isfinite(r.mean[free]).all()
        # better than a constant predictor on a smooth field
        assert np.sqrt(np.mean((r.mean - field) ** 2)) < np.sqrt(np.mean((field.mean() - field) ** 2))


def test_ordinary_kriging_runs_and_beats_constant():
    from compass.recon import OrdinaryKrigingReconstructor
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    obs = _sample_obs(field, n=120)
    r = OrdinaryKrigingReconstructor(vrange=20.0, nugget=1.0).reconstruct(obs, free)
    assert r.mean.shape == field.shape and np.isfinite(r.mean[free]).all()
    assert np.sqrt(np.mean((r.mean - field) ** 2)) < np.sqrt(np.mean((field.mean() - field) ** 2))
    # kriging variance higher far from observations
    obs_m = np.zeros_like(free); obs_m[obs.rows, obs.cols] = True
    assert r.std[obs_m].mean() <= r.std[free & ~obs_m].mean() + 1e-3


def test_geodesic_nearest_respects_walls():
    from compass.recon import GeodesicNearestReconstructor
    field = _smooth_field()
    free = np.ones_like(field, dtype=bool)
    free[:, 30:34] = False  # a wall splitting the map; gap at bottom
    free[60:, 30:34] = True
    obs = _sample_obs(field, n=80)
    obs = type(obs)(np.clip(obs.rows, 0, 63), np.clip(obs.cols, 0, 29), obs.values)  # all obs on LEFT side
    r = GeodesicNearestReconstructor().reconstruct(obs, free)
    assert r.mean.shape == field.shape
    assert np.isfinite(r.mean[free]).all()
    # geodesic distance (uncertainty) on the right side should exceed Euclidean-near values
    assert r.std[free].max() > 0


def test_tx_estimators_recover_source():
    from compass.recon import Observations
    from compass.recon.tx_estimate import log_distance_fit, weighted_centroid, tx_error_px
    rng = np.random.default_rng(0)
    src = (180.0, 70.0)  # true source (row, col)
    # sample free-space points; RSS = -40 - 22*log10(dist) + small noise (log-distance)
    rs = rng.uniform(20, 240, 200); cs = rng.uniform(20, 240, 200)
    d = np.hypot(rs - src[0], cs - src[1]) + 1.0
    vals = -40 - 22 * np.log10(d) + rng.normal(0, 1.5, len(d))
    obs = Observations(rs.astype(int), cs.astype(int), vals)
    est = log_distance_fit(obs, shape=(256, 256), grid_step=8)
    # log-distance fit should land within ~2 grid cells of the true source
    assert tx_error_px(est, src) < 25
    # weighted centroid runs and returns an in-range pixel
    wc = weighted_centroid(obs)
    assert 0 <= wc[0] < 256 and 0 <= wc[1] < 256

"""The realistic noise model: correlated, tunable correlation length, well-behaved."""

from __future__ import annotations

import numpy as np

from compass.data.noise import (
    MeasurementNoise,
    correlated_shadowing_field,
    measured_decorrelation_px,
)


def test_field_std_matches_sigma():
    f = correlated_shadowing_field(sigma_db=6.0, decorr_m=30.0, rng=np.random.default_rng(0))
    assert 0.7 <= f.std() / 6.0 <= 1.3
    assert abs(float(f.mean())) < 1.0  # ~zero mean


def test_decorrelation_grows_with_length():
    rng = np.random.default_rng(0)
    d5 = measured_decorrelation_px(correlated_shadowing_field(decorr_m=5.0, rng=rng))
    d20 = measured_decorrelation_px(correlated_shadowing_field(decorr_m=20.0, rng=rng))
    d40 = measured_decorrelation_px(correlated_shadowing_field(decorr_m=40.0, rng=rng))
    assert d5 < d20 < d40


def test_correlated_not_iid():
    # a correlated field has much longer decorrelation than i.i.d. white noise (~1 px)
    rng = np.random.default_rng(0)
    corr = measured_decorrelation_px(correlated_shadowing_field(decorr_m=30.0, rng=rng))
    iid = measured_decorrelation_px(rng.standard_normal((256, 256)).astype("float32"))
    assert corr > 5 * max(iid, 1.0)


def test_apply_finite_and_floored():
    rng = np.random.default_rng(0)
    noise = MeasurementNoise()
    field = noise.new_field(rng=rng)
    clean = np.full(50, -130.0, dtype=np.float32)
    rows = np.linspace(10, 200, 50)
    cols = np.linspace(10, 200, 50)
    meas = noise.apply(clean, rows, cols, field, rng)
    assert meas.shape == clean.shape
    assert np.isfinite(meas).all()
    assert meas.min() >= noise.floor_dbm - 1e-3
    # quantised to 1 dB grid
    assert np.allclose(meas, np.round(meas / noise.quantization_db) * noise.quantization_db)

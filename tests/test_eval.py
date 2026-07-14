"""Metrics + significance behave correctly on controlled inputs."""

from __future__ import annotations

import numpy as np

from compass.eval import accuracy_metrics, all_metrics, bootstrap_ci, calibration_metrics, paired_wilcoxon


def _scene(H=48, W=48, seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    gt = (gaussian_filter(rng.standard_normal((H, W)), 6) * 30 - 120).astype(np.float32)
    free = np.ones((H, W), bool)
    free[:10, :10] = False
    obs = np.zeros((H, W), bool)
    obs[rng.integers(0, H, 40), rng.integers(0, W, 40)] = True
    obs &= free
    return gt, free, obs


def test_perfect_prediction_zero_error_high_ssim():
    gt, free, obs = _scene()
    m = accuracy_metrics(gt.copy(), gt, free, obs)
    assert m["rmse_free_unobs"] < 1e-4
    assert m["ssim_free"] > 0.99


def test_calibration_corr_when_std_tracks_error():
    gt, free, obs = _scene()
    rng = np.random.default_rng(1)
    err = rng.random(gt.shape).astype(np.float32) * 10
    pred = gt + err
    std = err + rng.random(gt.shape).astype(np.float32) * 0.5  # std ~ error
    c = calibration_metrics(pred, gt, std, free, obs)
    assert c["unc_err_corr"] > 0.7


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).normal(10, 2, 200)
    ci = bootstrap_ci(v)
    assert ci["lo"] < ci["mean"] < ci["hi"]


def test_wilcoxon_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(8, 1, 60)
    b = a + 1.0  # b consistently worse
    res = paired_wilcoxon(a, b)
    assert res["p_value"] < 0.05 and res["median_diff"] < 0


def test_all_metrics_keys_present():
    gt, free, obs = _scene()
    pred = gt + np.random.default_rng(2).normal(0, 5, gt.shape).astype(np.float32)
    std = np.abs(np.random.default_rng(3).normal(0, 5, gt.shape)).astype(np.float32)
    building01 = (~free).astype(np.float32)
    m = all_metrics(pred, gt, std, building01, free, obs, (24, 24))
    for k in ["rmse_free_unobs", "ssim_free", "unc_err_corr", "ece", "rmse_nlos"]:
        assert k in m

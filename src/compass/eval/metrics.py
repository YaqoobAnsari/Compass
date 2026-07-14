"""
Comprehensive evaluation metrics for COMPASS — every publication dimension.

All metrics operate in dBm on dense maps with masks, so classical and learned
methods are scored identically.

  Accuracy:           rmse / mae / psnr / ssim, over free-space and blind-spot.
  Calibration:        uncertainty-error correlation, ECE, PICP (the UQ claim).
  Physical plausibility: behind-building error, ray-monotonicity violation rate.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity


def ray_monotonicity_violation(pred, tx_rc, free, n_rays=72, n_steps=48,
                               max_radius=200.0, thresh_db=0.5):
    """Fraction of outward ray-steps (in free space) where predicted signal INCREASES
    away from the TX — the physical-decay violation the Ray-Consistency Loss targets.
    Lower is more physically plausible."""
    H, W = pred.shape
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    rad = np.linspace(1.0, max_radius, n_steps)
    tr, tc = float(tx_rc[0]), float(tx_rc[1])
    r = tr + np.sin(ang)[:, None] * rad[None, :]
    c = tc + np.cos(ang)[:, None] * rad[None, :]
    p = map_coordinates(pred, [r.ravel(), c.ravel()], order=1, mode="nearest").reshape(n_rays, n_steps)
    f = map_coordinates(free.astype(float), [r.ravel(), c.ravel()], order=0, mode="nearest").reshape(n_rays, n_steps)
    d = p[:, 1:] - p[:, :-1]                       # outward step change
    valid = (f[:, 1:] > 0.5) & (f[:, :-1] > 0.5)   # both endpoints in free space
    inc = (d > thresh_db) & valid                  # increasing outward = violation
    return float(inc.sum() / max(valid.sum(), 1))


def _rmse(err, m):
    return float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")


def _mae(err, m):
    return float(np.mean(np.abs(err[m]))) if m.any() else float("nan")


def accuracy_metrics(pred, gt, free, obs, data_range=139.0):
    """RMSE/MAE over free-all and free-unobserved; PSNR; masked SSIM."""
    err = pred - gt
    unobs = free & ~obs
    out = {
        "rmse_free": _rmse(err, free),
        "rmse_free_unobs": _rmse(err, unobs),
        "rmse_free_obs": _rmse(err, free & obs),
        "mae_free_unobs": _mae(err, unobs),
        "n_free_unobs": int(unobs.sum()),
    }
    # variance-normalised MSE (scale-free; comparable across works, e.g. RMDM's NMSE)
    if unobs.any() and np.var(gt[unobs]) > 1e-9:
        out["nmse_free_unobs"] = float(np.mean(err[unobs] ** 2) / np.var(gt[unobs]))
    mse = np.mean(err[free] ** 2) if free.any() else np.nan
    out["psnr_free"] = float(20 * np.log10(data_range) - 10 * np.log10(mse)) if mse > 0 else float("nan")
    # masked SSIM: SSIM map over the whole image, averaged on free pixels
    try:
        _, ssim_map = structural_similarity(gt, pred, data_range=data_range, full=True)
        out["ssim_free"] = float(ssim_map[free].mean()) if free.any() else float("nan")
    except Exception:
        out["ssim_free"] = float("nan")
    return out


def calibration_metrics(pred, gt, std, free, obs, n_bins=10):
    """Does predicted uncertainty track real error in the blind spot?"""
    unobs = free & ~obs
    if unobs.sum() < 20:
        return {"unc_err_corr": float("nan"), "ece": float("nan"), "picp_1sigma": float("nan")}
    err = np.abs(pred - gt)[unobs]
    s = std[unobs]
    corr = float(np.corrcoef(err, s)[0, 1]) if np.std(s) > 0 else float("nan")
    # PICP at 1 sigma (fraction of true values within +-1 std)
    picp = float(np.mean(np.abs(pred - gt)[unobs] <= std[unobs] + 1e-6))
    # ECE-style: bin by predicted std, compare mean predicted std to RMS error per bin
    order = np.argsort(s)
    ece, n = 0.0, len(s)
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        ece += len(b) / n * abs(np.sqrt(np.mean(err[b] ** 2)) - np.mean(s[b]))
    return {"unc_err_corr": corr, "ece": float(ece), "picp_1sigma": picp}


def _occlusion_count(building01, tx_rc, n_steps=64):
    """For each free pixel, # building pixels on the straight TX->pixel ray (LoS test)."""
    H, W = building01.shape
    yy, xx = np.mgrid[0:H, 0:W]
    tr, tc = tx_rc
    ts = np.linspace(0.0, 1.0, n_steps)[None, None, :]
    ry = (tr + (yy[..., None] - tr) * ts).astype(int).clip(0, H - 1)
    rx = (tc + (xx[..., None] - tc) * ts).astype(int).clip(0, W - 1)
    return building01[ry, rx].sum(axis=-1)  # (H,W)


def plausibility_metrics(pred, gt, building01, free, tx_rc):
    """Behind-building error + does prediction respect occlusion ordering."""
    occ = _occlusion_count(building01, tx_rc)
    nlos = free & (occ >= 2)   # clearly behind >=2 building pixels
    los = free & (occ == 0)
    err = pred - gt
    out = {
        "rmse_nlos": _rmse(err, nlos),
        "rmse_los": _rmse(err, los),
        "n_nlos": int(nlos.sum()),
    }
    # NLoS predicted signal should be <= LoS predicted mean at matched distance band (sanity)
    if los.any() and nlos.any():
        out["nlos_minus_los_pred_db"] = float(pred[nlos].mean() - pred[los].mean())
    else:
        out["nlos_minus_los_pred_db"] = float("nan")
    return out


def all_metrics(pred, gt, std, building01, free, obs, tx_rc):
    m = {}
    m.update(accuracy_metrics(pred, gt, free, obs))
    m.update(calibration_metrics(pred, gt, std, free, obs))
    m.update(plausibility_metrics(pred, gt, building01, free, tx_rc))
    return m

"""
Estimate the transmitter location from sparse RSS observations alone — for the
real-data setting where the cell/TX coordinates are unknown. Three estimators of
increasing principle:

  * argmax_rss        — the strongest sample's position (cheap, biased to coverage).
  * weighted_centroid — linear-power-weighted centroid of the samples.
  * log_distance_fit  — grid-search the source that best explains RSS under a
                        log-distance pathloss model RSS ≈ a + b·log10(d). Principled;
                        recovers a source even outside the observed region.

Used to feed an ESTIMATED TX heatmap to the reconstruction model when the true TX
is unavailable. We de-risk this on synthetic (true TX known) before real data.
"""

from __future__ import annotations

import numpy as np

from .base import Observations


def argmax_rss(obs: Observations) -> tuple[int, int]:
    i = int(np.argmax(obs.values))
    return int(obs.rows[i]), int(obs.cols[i])


def weighted_centroid(obs: Observations) -> tuple[int, int]:
    w = np.power(10.0, obs.values / 10.0)  # dBm -> linear power
    w = w / (w.sum() + 1e-12)
    return int(round(float((obs.rows * w).sum()))), int(round(float((obs.cols * w).sum())))


def log_distance_fit(obs: Observations, shape=(256, 256), grid_step: int = 8) -> tuple[int, int]:
    """Source that best fits RSS = a + b·log10(dist) (least squares per candidate)."""
    H, W = shape
    gy = np.arange(0, H, grid_step)
    gx = np.arange(0, W, grid_step)
    cand = np.array([(r, c) for r in gy for c in gx], dtype=float)  # (M,2)
    P = np.stack([obs.rows, obs.cols], 1).astype(float)            # (N,2)
    v = obs.values.astype(float)                                    # (N,)
    N = len(v)
    # distances candidate->obs : (M,N)
    d = np.sqrt(((cand[:, None, :] - P[None, :, :]) ** 2).sum(-1)) + 1.0
    x = np.log10(d)                                                 # (M,N) regressor
    sx = x.sum(1); sxx = (x * x).sum(1)
    sv = v.sum(); svx = (x * v).sum(1)
    det = N * sxx - sx * sx + 1e-9
    b = (N * svx - sx * sv) / det
    a = (sv - b * sx) / N
    pred = a[:, None] + b[:, None] * x                             # (M,N)
    sse = ((pred - v[None, :]) ** 2).sum(1)
    # prefer physically-sensible decay (b<0); penalise increasing fits
    sse = sse + np.where(b > 0, 1e6, 0.0)
    best = int(np.argmin(sse))
    return int(cand[best, 0]), int(cand[best, 1])


ESTIMATORS = {
    "argmax_rss": argmax_rss,
    "weighted_centroid": weighted_centroid,
    "log_distance_fit": log_distance_fit,
}


def tx_error_px(est, true) -> float:
    return float(np.hypot(est[0] - true[0], est[1] - true[1]))

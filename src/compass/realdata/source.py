"""
Source (transmitter / booster) location approximation on real data — WITH a
reliability gate, because real cells have no ground-truth source and many carry a
booster that shares the transmitter's ID (distributed source, not a point).

Estimators (pixel coords, x=col, y=row):
  * argmax_rss        — RP of strongest mean RSS.
  * power_centroid    — linear-power-weighted centroid of RPs.
  * log_distance      — grid-search the source best explaining RSS ≈ a + b·log10(d).

Reliability (no GT), a source is only trustworthy if ALL hold:
  * FIT: the log-distance model explains the RSS (R² ≥ r2_min) with a physically
    decaying slope (b < 0, i.e. RSS falls with distance).
  * STABILITY: bootstrap-resampling RPs moves the estimate little (p90 < move_max m).
  * AGREEMENT: the three estimators land close together (< agree_max m).
A cell failing any is flagged as a probable booster / distributed source — we do NOT
assign it a point location. This is the "confirm it trends reliably" gate.
"""

from __future__ import annotations

import numpy as np

M_PER_PX = 0.032


def argmax_rss(pos, rss):
    return pos[int(np.argmax(rss))].astype(float)


def power_centroid(pos, rss):
    w = np.power(10.0, rss / 10.0)
    w = w / (w.sum() + 1e-12)
    return (pos * w[:, None]).sum(0)


def log_distance(pos, rss, step_m=1.0, pad_m=8.0, m_per_px=M_PER_PX):
    """Returns (src_px (2,), slope_b, r2). slope_b<0 means RSS decays with distance."""
    P = pos * m_per_px
    v = rss.astype(float)
    lo, hi = P.min(0) - pad_m, P.max(0) + pad_m
    gx = np.arange(lo[0], hi[0] + 1e-6, step_m)
    gy = np.arange(lo[1], hi[1] + 1e-6, step_m)
    cand = np.array([(x, y) for x in gx for y in gy])
    d = np.sqrt(((cand[:, None, :] - P[None, :, :]) ** 2).sum(-1)) + 1.0
    x = np.log10(d)
    N = len(v)
    sx, sxx = x.sum(1), (x * x).sum(1)
    sv, svx = v.sum(), (x * v).sum(1)
    det = N * sxx - sx * sx + 1e-9
    b = (N * svx - sx * sv) / det
    a = (sv - b * sx) / N
    pred = a[:, None] + b[:, None] * x
    sse = ((pred - v[None, :]) ** 2).sum(1)
    sse = sse + np.where(b > 0, 1e12, 0.0)  # forbid increasing fits
    best = int(np.argmin(sse))
    sst = ((v - v.mean()) ** 2).sum() + 1e-9
    r2 = 1.0 - sse[best] / sst
    return cand[best] / m_per_px, float(b[best]), float(r2)


def estimate_all(pos, rss):
    src_ld, b, r2 = log_distance(pos, rss)
    return {
        "argmax_rss": argmax_rss(pos, rss),
        "power_centroid": power_centroid(pos, rss),
        "log_distance": src_ld,
        "slope_b": b, "r2": r2,
    }


def stability_m(pos, rss, n_boot=40, seed=0, m_per_px=M_PER_PX):
    """Bootstrap-RP displacement of the log-distance source (median, p90 in metres)."""
    rng = np.random.default_rng(seed)
    base, _, _ = log_distance(pos, rss)
    disp = []
    n = len(pos)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(idx)) < 4:
            continue
        s, _, _ = log_distance(pos[idx], rss[idx])
        disp.append(float(np.hypot(*(s - base)) * m_per_px))
    if not disp:
        return float("nan"), float("nan")
    return float(np.median(disp)), float(np.percentile(disp, 90))


def method_agreement_m(est, m_per_px=M_PER_PX):
    """Max pairwise distance (m) between the three estimators."""
    pts = [est["argmax_rss"], est["power_centroid"], est["log_distance"]]
    dm = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dm = max(dm, float(np.hypot(*(pts[i] - pts[j])) * m_per_px))
    return dm


def reliability(pos, rss, r2_min=0.3, move_max=8.0, agree_max=10.0):
    """Full reliability verdict for one cell's source estimate."""
    est = estimate_all(pos, rss)
    med_mv, p90_mv = stability_m(pos, rss)
    agree = method_agreement_m(est)
    trustworthy = (est["r2"] >= r2_min) and (est["slope_b"] < 0) and \
                  (p90_mv < move_max) and (agree < agree_max)
    return {
        "src_px": [round(float(est["log_distance"][0]), 1), round(float(est["log_distance"][1]), 1)],
        "slope_b": round(est["slope_b"], 2),
        "pathloss_n": round(-est["slope_b"] / 10.0, 2),  # RSS=P0-10n log10 d -> n=-b/10
        "r2": round(est["r2"], 3),
        "stability_p90_m": round(p90_mv, 2), "stability_med_m": round(med_mv, 2),
        "method_agreement_m": round(agree, 2),
        "trustworthy": bool(trustworthy),
    }

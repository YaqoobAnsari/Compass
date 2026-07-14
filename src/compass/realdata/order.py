"""
Does temporal ORDER carry usable signal in real crowdsensed RSS streams?

Mobile walks give an ordered RSS series per (transmitter, phone) with NO position
labels. Consecutive scans are spatially adjacent, so if the walk is informative the
series is temporally autocorrelated and gaps can be filled from temporal neighbours.

We quantify this two ways, each with a shuffled control that destroys adjacency:
  * lag-1 autocorrelation of the ordered series (true vs shuffled).
  * gap-filling: hold out a fraction of scans, impute from temporal neighbours
    (linear in scan-index) on the TRUE order vs a SHUFFLED order vs the series mean.

If true-order imputation << shuffled ~ mean, order is informative on real data —
the empirical basis for COMPASS's order-aware conditioning (innovation #3).
"""

from __future__ import annotations

from typing import List

import numpy as np


def lag1_autocorr(y: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation of a 1-D series (nan if degenerate)."""
    y = np.asarray(y, float)
    if len(y) < 3 or y.std() < 1e-9:
        return float("nan")
    a, b = y[:-1], y[1:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _interp_holdout(idx: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Linear interpolation (in scan-index space) at held-out positions from the rest."""
    mask = np.ones(len(y), bool)
    mask[test] = False
    xt, yt = idx[mask], y[mask]
    order = np.argsort(xt)
    return np.interp(idx[test], xt[order], yt[order])


def gap_fill_errors(y: np.ndarray, test_frac: float = 0.4, n_rep: int = 5,
                    seed: int = 0) -> dict:
    """Abs gap-filling errors for TRUE order, SHUFFLED order, and MEAN baseline.

    idx is the scan position (0..T-1); a held-out point is imputed by linear
    interpolation from its temporal neighbours. The shuffled control permutes the
    series before imputing (same held-out count) so only adjacency is destroyed.
    Returns {'true': [...], 'shuffled': [...], 'mean': [...]}.
    """
    y = np.asarray(y, float)
    T = len(y)
    idx = np.arange(T, dtype=float)
    rng = np.random.default_rng(seed)
    out = {"true": [], "shuffled": [], "mean": []}
    if T < 8 or y.std() < 1e-9:
        return out
    ntest = max(1, int(round(T * test_frac)))
    for _ in range(n_rep):
        # keep endpoints observed so every held-out point is bracketed
        test = rng.choice(np.arange(1, T - 1), size=min(ntest, T - 2), replace=False)
        yhat = _interp_holdout(idx, y, test)
        out["true"] += list(np.abs(yhat - y[test]))
        # mean baseline (order-blind): predict mean of observed
        mask = np.ones(T, bool); mask[test] = False
        out["mean"] += list(np.abs(y[test] - y[mask].mean()))
        # shuffled control: permute the series, same test positions
        ys = y[rng.permutation(T)]
        yhat_s = _interp_holdout(idx, ys, test)
        out["shuffled"] += list(np.abs(yhat_s - ys[test]))
    return out


def rmse(errs: List[float]) -> float:
    e = np.asarray(errs, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")

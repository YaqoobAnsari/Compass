"""Significance testing: bootstrap CIs + paired Wilcoxon (publication-grade)."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for the MEAN of a per-sample metric array."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"mean": float(np.mean(v)) if len(v) else float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)])
    return {
        "mean": float(v.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(len(v)),
    }


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank between two per-sample metric arrays (lower=better)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.allclose(a[m], b[m]):
        return {"p_value": float("nan"), "median_diff": float(np.median((a - b)[m])) if m.any() else float("nan")}
    try:
        stat, p = wilcoxon(a[m], b[m])
    except ValueError:
        p = float("nan")
    return {"p_value": float(p), "median_diff": float(np.median((a - b)[m])), "n": int(m.sum())}

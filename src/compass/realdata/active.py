"""
Active crowdsensing on REAL floor geometry (innovation #5), discrete-pool version.

On real data the candidate measurement locations are a cell's reference points (RPs),
not a dense grid. Starting from a small seed of measured RPs, an acquisition strategy
picks the next RP to measure; we reconstruct at the still-unmeasured RPs and track
held-out RMSE vs budget. Uncertainty-guided (max-variance) acquisition uses the GP
posterior variance — the quantity a geometry-blind method cannot get right near walls.

Strategies: max_variance (GP posterior std), space_filling (max-min distance to the
observed set), coverage (farthest from the observed centroid), random.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def gp_predict(Xtr, ytr, Q, length_scale=5.0, noise=3.0):
    """GP posterior mean + std at Q (metres). Returns (mean, std)."""
    def K(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / length_scale ** 2)

    m = ytr.mean()
    y = ytr - m
    sig2 = max(float(ytr.var()), 1e-3)
    Kxx = sig2 * K(Xtr, Xtr) + noise ** 2 * np.eye(len(Xtr))
    Kinv = np.linalg.inv(Kxx + 1e-6 * np.eye(len(Xtr)))
    Kqx = sig2 * K(Q, Xtr)
    mean = Kqx @ (Kinv @ y) + m
    var = sig2 - np.einsum("ij,jk,ik->i", Kqx, Kinv, Kqx)
    return mean, np.sqrt(np.clip(var, 1e-6, None))


def _pick(strategy, unobs, obs, pos_m, std, rng):
    if strategy == "max_variance":
        return unobs[int(np.argmax(std))]
    if strategy == "random":
        return int(rng.choice(unobs))
    if strategy == "space_filling":
        d = np.array([min(np.hypot(*(pos_m[u] - pos_m[o])) for o in obs) for u in unobs])
        return unobs[int(np.argmax(d))]
    if strategy == "coverage":
        c = pos_m[obs].mean(0)
        d = np.array([np.hypot(*(pos_m[u] - c)) for u in unobs])
        return unobs[int(np.argmax(d))]
    raise ValueError(strategy)


STRATEGIES = ["max_variance", "space_filling", "coverage", "random"]


def active_cell(pos_m: np.ndarray, rss: np.ndarray, strategy: str, n_seed: int = 4,
                seed: int = 0, length_scale: float = 5.0, noise: float = 3.0) -> List[tuple]:
    """One cell: returns [(budget, heldout_rmse), ...] as RPs are acquired."""
    rng = np.random.default_rng(seed)
    n = len(pos_m)
    obs = list(rng.choice(n, min(n_seed, n - 1), replace=False))
    curve = []
    while len(obs) < n - 1:
        unobs = [i for i in range(n) if i not in obs]
        mean, std = gp_predict(pos_m[obs], rss[obs], pos_m[unobs], length_scale, noise)
        rmse = float(np.sqrt(np.mean((mean - rss[unobs]) ** 2)))
        curve.append((len(obs), rmse))
        nxt = _pick(strategy, unobs, obs, pos_m, std, rng)
        obs.append(nxt)
    return curve


def curve_at_budgets(pos_m, rss, strategy, budgets, n_rep=5, **kw) -> Dict[int, List[float]]:
    """Held-out RMSE at each budget, over n_rep random seeds (returns {budget:[rmse]})."""
    out = {b: [] for b in budgets}
    for rep in range(n_rep):
        curve = dict(active_cell(pos_m, rss, strategy, seed=rep, **kw))
        for b in budgets:
            # nearest available budget <= b (curve is at integer budgets)
            avail = [k for k in curve if k <= b]
            if avail:
                out[b].append(curve[max(avail)])
    return out

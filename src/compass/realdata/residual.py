"""
Geometry-conditioned learned residual on top of the classical interpolation field.

Motivation. On sparse real RPs a tuned classical interpolant (RBF multiquadric) is
the strongest reconstructor; a dense learned map-predictor loses to it because it
needs dense supervision, a matching spatial scale, and a reliable transmitter that
real crowdsensing lacks. This module keeps the classical field as the backbone and
asks a small learned model only to predict the RESIDUAL the kernel misses, driven by
wall geometry (the NLoS shadowing a geometry-blind kernel cannot represent):

    prediction(q) = RBF(q | observed) + f_theta( features(q, observed) )

f_theta is a compact tabular regressor. Its inputs are several classical estimates at
the query, distance/density statistics, and wall-crossing statistics between the query
and its supporting observations. Where there are no walls (open sites) the geometry
features vanish and f_theta falls back toward zero, recovering plain RBF; where walls
are present it applies a learned correction. Evaluated leave-one-CELL-out so the learner
never sees the held-out cell, with the same buffered leave-one-RP-out protocol used for
the classical panel, making the numbers directly comparable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .reconstruct import M_PER_PX, gp, idw, natural, rbf
from .walls import wall_aware_predict

# feature order kept explicit for interpretability and the no-geometry ablation
FEATURE_NAMES: List[str] = [
    "e_rbf", "e_idw2", "e_idw3", "e_gp", "e_natural", "e_wall",     # classical estimates
    "d1_m", "dmean_k_m", "nbr_count", "rss_std_db",                 # distance / density
    "wall1_m", "wall_mean_k_m", "wall_max_k_m", "wall_min_k_m",     # wall geometry
    "los_count", "disagree_wall", "disagree_idw", "disagree_gp",    # geometry / kernel disagreement
]
GEOM_FEATURES = {"e_wall", "wall1_m", "wall_mean_k_m", "wall_max_k_m",
                 "wall_min_k_m", "los_count", "disagree_wall"}
GEOM_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n in GEOM_FEATURES]
NONGEOM_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n not in GEOM_FEATURES]


def _estimate(fn, pos_m, rss, O, i):
    try:
        v = float(fn(pos_m[O], rss[O], pos_m[i:i + 1])[0])
        return v if np.isfinite(v) else np.nan
    except Exception:  # noqa: BLE001
        return np.nan


def cell_samples(pos_px: np.ndarray, rss: np.ndarray, W: np.ndarray,
                 buffer_m: float = 1.0, min_train: int = 4, k: int = 5, radius_m: float = 8.0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buffered leave-one-RP-out feature rows for one cell.

    Returns (Phi (n,F), base (n,), true (n,)): the feature matrix, the classical RBF
    base estimate at each held-out RP, and the true RSS at that RP. The learning target
    is the residual  true - base.
    """
    pos_m = pos_px * M_PER_PX
    n = len(pos_px)
    rows, base, true = [], [], []
    for i in range(n):
        d = np.linalg.norm(pos_m - pos_m[i], axis=1)
        keep = d > buffer_m
        keep[i] = False
        O = np.where(keep)[0]
        if len(O) < min_train:
            continue
        e_rbf = _estimate(rbf, pos_m, rss, O, i)
        if not np.isfinite(e_rbf):
            continue
        e_idw2 = _estimate(lambda a, b, q: idw(a, b, q, 2.0), pos_m, rss, O, i)
        e_idw3 = _estimate(lambda a, b, q: idw(a, b, q, 3.0), pos_m, rss, O, i)
        e_gp = _estimate(gp, pos_m, rss, O, i)
        e_nat = _estimate(natural, pos_m, rss, O, i)
        try:
            e_wall = float(wall_aware_predict(pos_px, rss, W, O, [i], lam=8.0)[0])
        except Exception:  # noqa: BLE001
            e_wall = e_idw2
        e_idw2 = e_idw2 if np.isfinite(e_idw2) else e_rbf
        e_idw3 = e_idw3 if np.isfinite(e_idw3) else e_rbf
        e_gp = e_gp if np.isfinite(e_gp) else e_idw2
        e_nat = e_nat if np.isfinite(e_nat) else e_idw2
        e_wall = e_wall if np.isfinite(e_wall) else e_idw2

        dO = d[O]
        wO = W[i, O]
        order = np.argsort(dO)
        sel = order[:min(k, len(O))]
        within = dO < radius_m
        d1 = float(dO[order[0]])
        wall1 = float(wO[order[0]])
        dmean_k = float(dO[sel].mean())
        wall_mean_k = float(wO[sel].mean())
        wall_max_k = float(wO[sel].max())
        wall_min_k = float(wO[sel].min())
        los_count = float(np.sum((wO < 0.1) & within))
        nbr_count = float(np.sum(within))
        rss_std = float(np.std(rss[O]))

        rows.append([
            e_rbf, e_idw2, e_idw3, e_gp, e_nat, e_wall,
            d1, dmean_k, nbr_count, rss_std,
            wall1, wall_mean_k, wall_max_k, wall_min_k,
            los_count, e_wall - e_rbf, e_idw2 - e_rbf, e_gp - e_rbf,
        ])
        base.append(e_rbf)
        true.append(float(rss[i]))
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), np.zeros(0)
    return np.asarray(rows, float), np.asarray(base, float), np.asarray(true, float)


def _fit_predict(model_name, Xtr, rtr, Xte, cols):
    """Train a residual regressor on training folds and predict held-out residuals."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    if model_name == "ridge":
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=10.0).fit(sc.transform(Xtr), rtr)
        return m.predict(sc.transform(Xte))
    m = HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0)
    m.fit(Xtr, rtr)
    return m.predict(Xte)


def run_residual_cv(cells: List[Dict], buffer_m: float = 1.0) -> Dict:
    """Leave-one-cell-out residual CV. `cells` is a list of dicts with keys
    site, floor, cell, pos_px, rss, W. Returns pooled abs-error arrays per method
    plus per-cell RMSE for paired significance tests."""
    # precompute per-cell samples
    per_cell = []
    for c in cells:
        Phi, base, true = cell_samples(c["pos_px"], c["rss"], c["W"], buffer_m=buffer_m)
        if len(Phi):
            per_cell.append({**{k: c[k] for k in ("site", "floor", "cell")},
                             "Phi": Phi, "base": base, "true": true})

    configs = {
        "RBF (base)": None,
        "residual-GBT": ("gbt", list(range(len(FEATURE_NAMES)))),
        "residual-GBT-nogeom": ("gbt", NONGEOM_COLS),
        "residual-Ridge": ("ridge", list(range(len(FEATURE_NAMES)))),
    }
    abs_err = {name: [] for name in configs}                     # pooled |pred-true|
    per_cell_rmse = {name: [] for name in configs}               # one RMSE per cell
    site_abs = {name: {} for name in configs}                    # site -> pooled |err|

    for h, held in enumerate(per_cell):
        train = [pc for j, pc in enumerate(per_cell) if j != h]
        Xtr = np.vstack([pc["Phi"] for pc in train])
        rtr = np.concatenate([pc["true"] - pc["base"] for pc in train])
        site = held["site"]
        for name, cfg in configs.items():
            if cfg is None:
                pred = held["base"]
            else:
                mname, cols = cfg
                resid = _fit_predict(mname, Xtr, rtr, held["Phi"], cols)
                pred = held["base"] + resid
            e = np.abs(pred - held["true"])
            abs_err[name].extend(e.tolist())
            per_cell_rmse[name].append(float(np.sqrt(np.mean(e ** 2))))
            site_abs[name].setdefault(site, []).extend(e.tolist())

    return {"n_cells": len(per_cell), "abs_err": abs_err,
            "per_cell_rmse": per_cell_rmse, "site_abs": site_abs}

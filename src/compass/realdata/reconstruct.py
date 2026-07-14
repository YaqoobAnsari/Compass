"""
Real-data (UniCellular) TX-agnostic reconstruction machinery.

No dense ground truth exists, so reconstruction is validated by held-out
reference-point (RP) cross-validation: observe a cell's RSS at a subset of RPs,
predict at held-out RPs, spatially buffered so we test true interpolation (not
near-duplicates). Positions are in METRES (pixel x 0.032 m/px).

Point interpolators (train on RP points, predict at query points) mirror the
classical baseline families used on synthetic — a fair, comprehensive panel for
the sparse-RP indoor regime where a dense U-Net is not applicable.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

from .unicellular import RECON_COLS, DATASET_ROOT, load_floor, load_rp_coords, valid_measurements

M_PER_PX = 0.032


def cell_rp_tables(site: str, floor: int, min_rp: int = 8, root=DATASET_ROOT) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """{cell_id: (positions_m (N,2), mean_rss_dbm (N,))} for cells seen by >= min_rp RPs."""
    df = load_floor(site, "stationary", floor, root, usecols=RECON_COLS)
    v = valid_measurements(df)
    v = v[v["rpNumber"] >= 1]
    coords = load_rp_coords(site, floor, root)
    if not coords:
        return {}
    out = {}
    for cell, g in v.groupby("transmitter_id"):
        rp_rss = g.groupby("rpNumber")["transmitter_rss"].mean()
        rps = [int(r) for r in rp_rss.index if int(r) in coords]
        if len(rps) < min_rp:
            continue
        pos = np.array([coords[r] for r in rps], dtype=float) * M_PER_PX
        rss = np.array([float(rp_rss[r]) for r in rps], dtype=float)
        out[int(cell)] = (pos, rss)
    return out


# --- point interpolators: (Xtrain (n,2), ytrain (n,), Xquery (m,2)) -> yhat (m,) ---
def idw(Xtr, ytr, Q, power=2.0, k=12):
    tree = cKDTree(Xtr)
    kk = min(k, len(ytr))
    d, idx = tree.query(Q, k=kk)
    d = np.atleast_2d(d.T).T if kk > 1 else d[:, None]
    idx = np.atleast_2d(idx.T).T if kk > 1 else idx[:, None]
    w = 1.0 / np.clip(d, 1e-6, None) ** power
    return np.sum(w * ytr[idx], axis=1) / np.sum(w, axis=1)


def nearest(Xtr, ytr, Q):
    return NearestNDInterpolator(Xtr, ytr)(Q)


def natural(Xtr, ytr, Q):
    if len(Xtr) < 4:
        return nearest(Xtr, ytr, Q)
    lin = LinearNDInterpolator(Xtr, ytr)
    y = lin(Q)
    nan = ~np.isfinite(y)
    if nan.any():
        y[nan] = NearestNDInterpolator(Xtr, ytr)(Q[nan])
    return y


def rbf(Xtr, ytr, Q, kernel="multiquadric", epsilon=5.0, smoothing=1.0):
    kw = {"kernel": kernel, "smoothing": smoothing}
    if kernel in ("multiquadric", "gaussian", "inverse_multiquadric"):
        kw["epsilon"] = epsilon
    return RBFInterpolator(Xtr, ytr, **kw)(Q)


def gp(Xtr, ytr, Q, length_scale=5.0, noise=3.0):
    def K(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / length_scale ** 2)
    m = ytr.mean()
    y = ytr - m
    sig2 = max(float(ytr.var()), 1e-3)
    A = sig2 * K(Xtr, Xtr) + noise ** 2 * np.eye(len(Xtr))
    alpha = np.linalg.solve(A + 1e-6 * np.eye(len(A)), y)
    return sig2 * K(Q, Xtr) @ alpha + m


METHODS: Dict[str, Callable] = {
    "IDW(p=2)": lambda a, b, q: idw(a, b, q, 2.0),
    "IDW(p=3)": lambda a, b, q: idw(a, b, q, 3.0),
    "NearestNeighbor": nearest,
    "NaturalNeighbor": natural,
    "RBF(multiquadric)": lambda a, b, q: rbf(a, b, q, "multiquadric", 5.0),
    "RBF(thin_plate)": lambda a, b, q: rbf(a, b, q, "thin_plate_spline"),
    "GP/Kriging": lambda a, b, q: gp(a, b, q, 5.0, 3.0),
}


def loro_errors(pos: np.ndarray, rss: np.ndarray, method: Callable,
                buffer_m: float = 0.0, min_train: int = 4) -> List[float]:
    """Buffered leave-one-RP-out abs errors: for each RP, train on RPs farther than
    buffer_m, predict at the held-out RP."""
    errs = []
    for i in range(len(pos)):
        d = np.linalg.norm(pos - pos[i], axis=1)
        keep = d > buffer_m
        keep[i] = False
        if keep.sum() < min_train:
            continue
        try:
            yhat = float(method(pos[keep], rss[keep], pos[i:i + 1])[0])
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(yhat):
            errs.append(abs(yhat - rss[i]))
    return errs


def holdout_frac_errors(pos: np.ndarray, rss: np.ndarray, method: Callable,
                        test_frac: float = 0.5, n_rep: int = 3, seed: int = 0,
                        min_train: int = 4) -> List[float]:
    """Random train/test RP splits (coverage sweep): observe (1-test_frac) of RPs,
    predict the held-out test_frac; averaged over n_rep random splits."""
    rng = np.random.default_rng(seed)
    n = len(pos)
    errs = []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        ntest = max(1, int(round(n * test_frac)))
        test, train = idx[:ntest], idx[ntest:]
        if len(train) < min_train:
            continue
        try:
            yhat = method(pos[train], rss[train], pos[test])
        except Exception:  # noqa: BLE001
            continue
        for e in np.abs(np.asarray(yhat) - rss[test]):
            if np.isfinite(e):
                errs.append(float(e))
    return errs

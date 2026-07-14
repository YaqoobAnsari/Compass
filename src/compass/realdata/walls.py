"""
Wall-aware (NLoS-attenuation) reconstruction on real indoor data — the real-data
test of innovation #1 (geometry-aware reconstruction).

Indoor walls attenuate signal but do NOT force path detours at these open sites
(geodesic ≈ Euclidean, ratio ≈ 1.05; see the A2b geodesic probe). The informative
geometry signal is therefore the amount of WALL the line-of-sight segment crosses,
not path length. We form an effective distance

    d_eff(i, j) = d_euclidean(i, j) + lambda * wall_length_crossed(i, j)

and use it inside IDW. lambda > 0 down-weights across-wall neighbours, encoding the
NLoS attenuation that a geometry-blind interpolator ignores. lambda = 0 recovers
plain IDW, so any gain is attributable purely to geometry awareness.

Wall mask: floor-plan PNG thresholded dark, with the red reference-point markers
removed (they are drawn on the plan and must not count as walls).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .unicellular import DATASET_ROOT, RECON_COLS, RSS_COL, TX_COL, load_floor, valid_measurements

M_PER_PX = 0.032


def load_wall_mask(site: str, floor: int, root: Path = DATASET_ROOT, thresh: int = 100) -> Optional[np.ndarray]:
    """Boolean barrier mask from the floor-plan PNG (dark ink = wall/structure),
    excluding the red RP markers. None if no usable plan exists."""
    p = Path(root) / "floor_plans" / site / f"floor{floor}.png"
    if not p.exists() or p.stat().st_size < 1000:
        return None
    rgb = np.asarray(Image.open(p).convert("RGB")).astype(int)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    red = (R > 120) & (G < 100) & (B < 100)  # RP markers, not walls
    return (gray < thresh) & (~red)


def wall_length_m(bar: np.ndarray, p, q, m_per_px: float = M_PER_PX, n: int = 96) -> float:
    """Approximate metres of barrier crossed by the segment p->q (pixel coords)."""
    xs = np.linspace(p[0], q[0], n)
    ys = np.linspace(p[1], q[1], n)
    c = np.clip(xs.astype(int), 0, bar.shape[1] - 1)
    r = np.clip(ys.astype(int), 0, bar.shape[0] - 1)
    seg_px = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    return float(bar[r, c].mean()) * seg_px * m_per_px


def wall_matrix_m(pos_px: np.ndarray, bar: np.ndarray, m_per_px: float = M_PER_PX) -> np.ndarray:
    """Symmetric NxN matrix of wall-length (m) between all RP pairs."""
    n = len(pos_px)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = wall_length_m(bar, pos_px[i], pos_px[j], m_per_px)
    return W


def cell_rp_tables_px(site: str, floor: int, min_rp: int = 8, root: Path = DATASET_ROOT
                      ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """{cell_id: (positions_PIXELS (N,2), mean_rss_dbm (N,))} for wall tracing."""
    df = load_floor(site, "stationary", floor, root, usecols=RECON_COLS)
    v = valid_measurements(df)
    v = v[v["rpNumber"] >= 1]
    p = Path(root) / "coordinates" / site / f"floor{floor}.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())

    def xy(val):
        return (val["x"], val["y"]) if isinstance(val, dict) else (val[0], val[1])

    coords = {int(k): xy(val) for k, val in raw.items()}
    out = {}
    for cell, g in v.groupby(TX_COL):
        rp_rss = g.groupby("rpNumber")[RSS_COL].mean()
        rps = [int(r) for r in rp_rss.index if int(r) in coords]
        if len(rps) < min_rp:
            continue
        pos = np.array([coords[r] for r in rps], dtype=float)
        rss = np.array([float(rp_rss[r]) for r in rps], dtype=float)
        out[int(cell)] = (pos, rss)
    return out


def wall_aware_predict(pos_px, rss, W, obs_idx, test_idx, lam=8.0, power=2.0,
                       m_per_px=M_PER_PX, k=12):
    """Predict RSS at test_idx from obs_idx via wall-aware IDW (d_eff = d_euc + lam*wall)."""
    pos_m = pos_px * m_per_px
    obs_idx = np.asarray(obs_idx)
    yhat = []
    for t in test_idx:
        de = np.linalg.norm(pos_m[obs_idx] - pos_m[t], axis=1)
        deff = de + lam * W[t, obs_idx]
        kk = min(k, len(obs_idx))
        sel = np.argsort(deff)[:kk]
        w = 1.0 / np.clip(deff[sel], 1e-6, None) ** power
        yhat.append(float(np.sum(w * rss[obs_idx][sel]) / np.sum(w)))
    return np.array(yhat)


def wall_isolation(pos_px, W, obs_idx, cand_idx, lam=8.0, m_per_px=M_PER_PX):
    """Wall-aware isolation of each candidate = min effective distance to any observed
    RP (d_euc + lam*wall). High = poorly covered accounting for walls -> acquire next."""
    pos_m = pos_px * m_per_px
    obs_idx = np.asarray(obs_idx)
    out = []
    for c in cand_idx:
        de = np.linalg.norm(pos_m[obs_idx] - pos_m[c], axis=1)
        out.append(float(np.min(de + lam * W[c, obs_idx])))
    return np.array(out)


def wall_aware_loro(pos_px: np.ndarray, rss: np.ndarray, W: np.ndarray, lam: float = 0.0,
                    power: float = 2.0, m_per_px: float = M_PER_PX, k: int = 12,
                    buffer_m: float = 1.0, min_train: int = 4):
    """Buffered leave-one-RP-out abs errors for wall-aware IDW with a precomputed
    wall-length matrix W (m). lam=0 -> plain IDW."""
    pos_m = pos_px * m_per_px
    errs = []
    n = len(pos_px)
    for i in range(n):
        de = np.linalg.norm(pos_m - pos_m[i], axis=1)
        keep = de > buffer_m
        keep[i] = False
        idx = np.where(keep)[0]
        if len(idx) < min_train:
            continue
        deff = de[idx] + lam * W[i, idx]
        kk = min(k, len(idx))
        sel = np.argsort(deff)[:kk]
        w = 1.0 / np.clip(deff[sel], 1e-6, None) ** power
        yhat = float(np.sum(w * rss[idx][sel]) / np.sum(w))
        if np.isfinite(yhat):
            errs.append(abs(yhat - rss[i]))
    return errs

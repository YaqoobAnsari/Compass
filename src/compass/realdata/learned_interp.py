"""
Learned geometry-aware interpolator for REAL data — closes the gap that all real
reconstruction so far is classical. A permutation-invariant DeepSet predicts a query
RP's RSS from its observed neighbours, using per-neighbour features that INCLUDE the
wall-length between neighbour and query (geometry-aware). Trained across cells with a
held-out-cell split (no leakage); gives MC-dropout uncertainty (real-data UQ).

Feature per (query q, observed neighbour o):
    [ (o.x-q.x)/S, (o.y-q.y)/S, dist_m/S, wall_len_m/S, rss_o - ctx_mean ]
The 'no-walls' variant zeros the wall feature — an ablation testing #1 with a LEARNED
model on real data. Prediction = context-mean + learned residual.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

M_PER_PX = 0.032
FEAT_SCALE = 20.0  # metres, feature normaliser
K = 12             # neighbours per query


def _mlp(i, h, o, p):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Dropout(p),
                         nn.Linear(h, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, o))


class SetInterpolator(nn.Module):
    def __init__(self, hidden: int = 96, p_drop: float = 0.12):
        super().__init__()
        self.phi = _mlp(5, hidden, hidden, p_drop)     # per-neighbour encoder
        self.rho = _mlp(2 * hidden, hidden, 1, p_drop)  # aggregate -> residual

    def forward(self, feats, mask, ctx_mean):
        """feats (B,K,5), mask (B,K) 1=valid, ctx_mean (B,) -> pred rss (B,)."""
        h = self.phi(feats) * mask[..., None]          # (B,K,hidden)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        mean = h.sum(1) / denom
        neg = h.masked_fill(mask[..., None] < 0.5, -1e9)
        mx = neg.max(1).values
        resid = self.rho(torch.cat([mean, mx], dim=-1)).squeeze(-1)
        return ctx_mean + resid


def build_features(pos_px, rss, W, obs_idx, query_idx, use_walls=True, m_per_px=M_PER_PX):
    """Return (feats (K,5), mask (K,), ctx_mean) for one query from obs_idx neighbours.
    m_per_px lets the SAME feature space serve real (0.032) and synthetic (1.0) data."""
    q = pos_px[query_idx] * m_per_px
    op = pos_px[obs_idx] * m_per_px
    d = np.linalg.norm(op - q, axis=1)
    order = np.argsort(d)[:K]
    sel = np.asarray(obs_idx)[order]
    ctx_mean = float(rss[sel].mean())
    feats = np.zeros((K, 5), np.float32)
    mask = np.zeros(K, np.float32)
    for j, (oi, dd) in enumerate(zip(sel, d[order])):
        wl = float(W[query_idx, oi]) if use_walls else 0.0
        dx, dy = (pos_px[oi] - pos_px[query_idx]) * m_per_px
        feats[j] = [dx / FEAT_SCALE, dy / FEAT_SCALE, dd / FEAT_SCALE, wl / FEAT_SCALE,
                    (rss[oi] - ctx_mean)]
        mask[j] = 1.0
    return feats, mask, ctx_mean


def make_samples(cells, use_walls=True, buffer_m=1.0, min_train=4, m_per_px=M_PER_PX):
    """cells: list of (pos_px, rss, W). Yield (feats,mask,ctx_mean,target) leave-one-RP-out."""
    F, Mk, Cm, Y = [], [], [], []
    for pos, rss, W in cells:
        pm = pos * m_per_px
        for i in range(len(pos)):
            de = np.linalg.norm(pm - pm[i], axis=1)
            keep = np.where((de > buffer_m) & (np.arange(len(pos)) != i))[0]
            if len(keep) < min_train:
                continue
            f, m, c = build_features(pos, rss, W, keep, i, use_walls, m_per_px)
            F.append(f); Mk.append(m); Cm.append(c); Y.append(rss[i])
    if not F:
        return None
    return (torch.tensor(np.stack(F)), torch.tensor(np.stack(Mk)),
            torch.tensor(np.array(Cm, np.float32)), torch.tensor(np.array(Y, np.float32)))

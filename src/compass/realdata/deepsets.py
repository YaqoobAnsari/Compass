"""
TX-agnostic DEEP baseline for real crowdsensed reconstruction: a Conditional Neural
Process (CNP) over scattered RSS measurements.

Why this model. Every deep radio-map network in our synthetic panel (RadioUNet, PMNet,
RadioTransformer, RadioGAN, RadioMamba, URAM, RadioDiff, and COMPASS itself) consumes a
transmitter location and a dense pixel grid. Real indoor cellular crowdsensing supplies
neither: serving cells are largely unlocatable and measurements are scattered reference
points, not a raster. A comparison on real data therefore needs a deep model that is
transmitter-agnostic BY CONSTRUCTION and learns directly from scattered points. The CNP
family (Garnelo et al., 2018; attentive variant Kim et al., 2019) is exactly that, and
is the standard deep counterpart to Kriging for scattered spatial regression.

Formulation. For a query RP the model sees its supporting observations in RELATIVE
coordinates, so it is translation invariant and never references an absolute origin or
a transmitter. Each observation contributes (dx, dy, distance, log-distance, metres of
wall between it and the query, centred RSS). A shared encoder embeds every observation,
a learned attention pools them into one representation, and a decoder emits a mean and
a variance. RSS is centred on the context mean and the mean is added back, which absorbs
the per-cell power offset that would otherwise dominate across heterogeneous cells.

Protocol. Identical to the classical and residual panels: buffered leave-one-RP-out
targets inside each cell, wrapped in leave-one-CELL-out cross validation, so the network
never sees the held-out cell. Two variants are trained, with and without the wall
feature, which isolates whether geometry helps a deep model the same way it helps the
learned residual.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .reconstruct import M_PER_PX, rbf

SCALE_M = 10.0          # metre scale for relative coordinates
FEAT_DIM = 6            # dx, dy, d, logd, wall, y_centred


def cell_context_samples(pos_px: np.ndarray, rss: np.ndarray, W: np.ndarray,
                         buffer_m: float = 1.0, min_train: int = 4) -> Dict:
    """Buffered leave-one-RP-out context/target pairs for one cell.

    Mirrors `residual.cell_samples` exactly (same buffer, same min_train, same skip
    rules) so the evaluated target set is identical to the classical panel's.
    """
    pos_m = pos_px * M_PER_PX
    n = len(pos_px)
    ctx, ctx_len, true, base, offs = [], [], [], [], []
    for i in range(n):
        d = np.linalg.norm(pos_m - pos_m[i], axis=1)
        keep = d > buffer_m
        keep[i] = False
        O = np.where(keep)[0]
        if len(O) < min_train:
            continue
        try:
            e_rbf = float(rbf(pos_m[O], rss[O], pos_m[i:i + 1])[0])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(e_rbf):
            continue
        mu = float(np.mean(rss[O]))
        rel = (pos_m[O] - pos_m[i]) / SCALE_M
        dd = d[O] / SCALE_M
        f = np.stack([rel[:, 0], rel[:, 1], dd, np.log1p(dd),
                      W[i, O], rss[O] - mu], axis=1)
        ctx.append(f.astype(np.float32))
        ctx_len.append(len(O))
        true.append(float(rss[i]))
        base.append(e_rbf)
        offs.append(mu)                      # context mean, added back at prediction time
    return {"ctx": ctx, "ctx_len": ctx_len,
            "true": np.asarray(true, np.float32), "base": np.asarray(base, np.float32),
            "off": np.asarray(offs, np.float32)}


def _pack(samples: List[Dict], use_geom: bool):
    """Pad variable-length context sets into one tensor plus a mask."""
    ctx = [c for s in samples for c in s["ctx"]]
    true = np.concatenate([s["true"] for s in samples]) if samples else np.zeros(0, np.float32)
    base = np.concatenate([s["base"] for s in samples]) if samples else np.zeros(0, np.float32)
    off = np.concatenate([s["off"] for s in samples]) if samples else np.zeros(0, np.float32)
    if not ctx:
        return None
    m = max(len(c) for c in ctx)
    X = np.zeros((len(ctx), m, FEAT_DIM), np.float32)
    M = np.zeros((len(ctx), m), np.float32)
    for k, c in enumerate(ctx):
        X[k, :len(c)] = c
        M[k, :len(c)] = 1.0
    if not use_geom:
        X[:, :, 4] = 0.0                     # ablate the wall feature
    return (torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(true),
            torch.from_numpy(base), torch.from_numpy(off))


class CNP(nn.Module):
    """Conditional Neural Process with attentive pooling over the observation set."""

    def __init__(self, hid: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(FEAT_DIM, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU())
        self.att = nn.Linear(hid, 1)
        self.dec = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, X, M):
        h = self.enc(X)                                   # (B, m, hid)
        logit = self.att(h).squeeze(-1)                    # (B, m)
        logit = logit.masked_fill(M < 0.5, float("-inf"))
        a = torch.softmax(logit, dim=1).unsqueeze(-1)      # (B, m, 1)
        r_att = (a * h).sum(1)
        denom = M.sum(1, keepdim=True).clamp(min=1.0)
        r_mean = (h * M.unsqueeze(-1)).sum(1) / denom
        out = self.dec(torch.cat([r_att, r_mean], dim=-1))
        mu = out[:, 0]
        sigma = 0.1 + torch.nn.functional.softplus(out[:, 1])
        return mu, sigma


def _train_predict(tr_pack, te_pack, epochs: int = 400, lr: float = 3e-3, seed: int = 0):
    """Fit a CNP on the training folds and predict the held-out cell."""
    torch.manual_seed(seed)
    Xtr, Mtr, ytr, btr, offtr = tr_pack
    Xte, Mte, yte, bte, offte = te_pack
    net = CNP()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # the context is centred on its own mean, so the model predicts the centred level
    # and the SAME per-sample offset is added back at prediction time
    tgt = ytr - offtr
    net.train()
    for _ in range(epochs):
        mu, sigma = net(Xtr, Mtr)
        nll = (torch.log(sigma) + 0.5 * ((tgt - mu) / sigma) ** 2).mean()
        mse = ((tgt - mu) ** 2).mean()
        loss = nll + 0.1 * mse
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        mu, sigma = net(Xte, Mte)
    return (mu + offte).numpy(), sigma.numpy()


def run_deep_cv(cells: List[Dict], buffer_m: float = 1.0, epochs: int = 400) -> Dict:
    """Leave-one-CELL-out CV for the TX-agnostic deep baselines."""
    per_cell = []
    for c in cells:
        s = cell_context_samples(c["pos_px"], c["rss"], c["W"], buffer_m=buffer_m)
        if len(s["true"]):
            per_cell.append({**{k: c[k] for k in ("site", "floor", "cell")}, **s})

    configs = {"RBF (base)": None, "CNP-deep (geometry)": True, "CNP-deep (no geometry)": False}
    abs_err = {k: [] for k in configs}
    per_cell_rmse = {k: [] for k in configs}
    site_abs = {k: {} for k in configs}
    unc = {k: [] for k in configs if k != "RBF (base)"}

    for h, held in enumerate(per_cell):
        train = [pc for j, pc in enumerate(per_cell) if j != h]
        site = held["site"]
        for name, use_geom in configs.items():
            if use_geom is None:
                pred = held["base"]
            else:
                tr = _pack(train, use_geom)
                te = _pack([held], use_geom)
                if tr is None or te is None:
                    continue
                pred, sigma = _train_predict(tr, te, epochs=epochs, seed=h)
                unc[name].extend(sigma.tolist())
            e = np.abs(np.asarray(pred, float) - held["true"])
            abs_err[name].extend(e.tolist())
            per_cell_rmse[name].append(float(np.sqrt(np.mean(e ** 2))))
            site_abs[name].setdefault(site, []).extend(e.tolist())

    return {"n_cells": len(per_cell), "abs_err": abs_err,
            "per_cell_rmse": per_cell_rmse, "site_abs": site_abs, "uncertainty": unc}

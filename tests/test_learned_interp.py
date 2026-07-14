"""Learned DeepSet interpolator: correct shapes, masking, and it can fit a field."""

from __future__ import annotations

import numpy as np
import torch

from compass.realdata.learned_interp import SetInterpolator, build_features, make_samples


def _cell(n=20, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 400, (n, 2))  # pixels
    src = np.array([100.0, 300.0])
    rss = -50 - 20 * np.log10(np.linalg.norm((pos - src) * 0.032, axis=1) + 1) + rng.normal(0, 0.5, n)
    W = np.zeros((n, n))
    return pos, rss, W


def test_forward_shape_and_mask():
    m = SetInterpolator()
    feats = torch.randn(4, 12, 5)
    mask = torch.ones(4, 12); mask[:, 6:] = 0
    cm = torch.zeros(4)
    out = m(feats, mask, cm)
    assert out.shape == (4,) and torch.isfinite(out).all()


def test_build_features_uses_walls():
    pos, rss, W = _cell()
    W[0, 5] = 3.0  # a wall between RP0 and RP5
    f_w, _, _ = build_features(pos, rss, W, [5, 3, 2, 1], 0, use_walls=True)
    f_nw, _, _ = build_features(pos, rss, W, [5, 3, 2, 1], 0, use_walls=False)
    assert (f_w[:, 3] != 0).any()          # wall feature present
    assert (f_nw[:, 3] == 0).all()          # ablation zeros it


def test_make_samples_and_overfit():
    cells = [_cell(seed=i) for i in range(3)]
    data = make_samples(cells, use_walls=True)
    assert data is not None
    F, Mk, Cm, Y = data
    assert F.shape[1:] == (12, 5) and len(Y) > 30
    m = SetInterpolator(p_drop=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(200):
        pred = m(F, Mk, Cm)
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < float(((Y - Y.mean()) ** 2).mean())  # beats constant

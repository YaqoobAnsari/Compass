"""COMPASS-WNet: correct shape, coarse aux for deep supervision, capacity in range."""

from __future__ import annotations

import torch

from compass.models.compass_net import CompassConfig
from compass.models.compass_wnet import CompassWNet


def _batch(b=2, h=64, w=64, seq_dim=11):
    return {
        "sparse_rss": torch.randn(b, 1, h, w),
        "mask": (torch.rand(b, 1, h, w) > 0.9).float(),
        "coverage": torch.rand(b, 1, h, w),
        "building": (torch.rand(b, 1, h, w) > 0.5).float(),
        "free_mask": (torch.rand(b, 1, h, w) > 0.3).float(),
        "tx_rowcol": torch.randint(0, h, (b, 2)),
        "sequence": torch.randn(b, 20, seq_dim),
        "target": torch.randn(b, 1, h, w),
    }


def test_wnet_shape_and_aux():
    m = CompassWNet(CompassConfig())
    b = _batch()
    out = m(b)
    assert out.shape == (2, 1, 64, 64)
    assert m.aux is not None and m.aux.shape == (2, 1, 64, 64)  # coarse for deep supervision


def test_wnet_trains_one_step():
    m = CompassWNet(CompassConfig())
    b = _batch()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    out = m(b)
    loss = ((out - b["target"]) ** 2).mean() + 0.5 * ((m.aux - b["target"]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss)


def test_wnet_capacity_comparable():
    n = sum(p.numel() for p in CompassWNet(CompassConfig()).parameters())
    assert 4e6 < n < 1.2e7, f"WNet has {n/1e6:.1f}M params"

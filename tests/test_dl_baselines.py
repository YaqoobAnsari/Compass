"""External DL baselines: correct output shape, trainable, RadioUNet cascades."""

from __future__ import annotations

import torch

from compass.models.baselines import BASELINES, PMNet, RadioUNet, SparseUNet
from compass.models.baselines.rmdm import RMDM


def _batch(b=2, h=64, w=64):
    return {
        "sparse_rss": torch.randn(b, 1, h, w),
        "mask": (torch.rand(b, 1, h, w) > 0.9).float(),
        "coverage": torch.rand(b, 1, h, w),
        "building": (torch.rand(b, 1, h, w) > 0.5).float(),
        "free_mask": (torch.rand(b, 1, h, w) > 0.3).float(),
        "tx_rowcol": torch.randint(0, h, (b, 2)),
        "target": torch.randn(b, 1, h, w),
    }


def test_all_baselines_output_shape():
    b = _batch()
    for name, cls in BASELINES.items():
        model = cls()
        out = model(b)
        assert out.shape == (2, 1, 64, 64), f"{name} bad shape {out.shape}"


def test_rmdm_training_losses_and_sampling():
    model = RMDM(base=16, sample_steps=4)  # tiny for speed
    b = _batch()
    loss, comp = model.training_losses(b)
    assert torch.isfinite(loss) and "diff" in comp and "coarse" in comp
    loss.backward()
    out = model(b)  # DDIM sampling
    assert out.shape == (2, 1, 64, 64) and torch.isfinite(out).all()


def test_radiounet_sets_coarse_aux():
    model = RadioUNet()
    b = _batch()
    _ = model(b)
    assert model.aux is not None and model.aux.shape == (2, 1, 64, 64)


def test_baselines_take_one_train_step():
    b = _batch()
    for cls in (SparseUNet, RadioUNet, PMNet):
        model = cls()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        out = model(b)
        loss = ((out - b["target"]) ** 2).mean()
        if getattr(model, "aux", None) is not None:
            loss = loss + ((model.aux - b["target"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss)


def test_param_counts_reasonable():
    for cls in (SparseUNet, RadioUNet, PMNet):
        n = sum(p.numel() for p in cls().parameters())
        assert 1e5 < n < 5e7, f"{cls.__name__} has {n} params"

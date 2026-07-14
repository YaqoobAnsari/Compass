"""Model wiring: full model + every ablation produce correct shapes; loss + one
training step run; order-aware model is actually sensitive to sequence order."""

from __future__ import annotations

import torch

from compass.models import CompassConfig, CompassNet, tx_heatmap
from compass.training.losses import compass_loss, ray_consistency_loss


def _fake_batch(B=2, H=64, W=64, seq_len=40, F=11):
    g = torch.Generator().manual_seed(0)
    free = torch.ones(B, 1, H, W)
    free[:, :, :20, :20] = 0
    return {
        "sparse_rss": torch.randn(B, 1, H, W, generator=g) * 0.3,
        "mask": (torch.rand(B, 1, H, W, generator=g) > 0.97).float(),
        "coverage": torch.rand(B, 1, H, W, generator=g),
        "building": 1 - free,
        "free_mask": free,
        "tx_rowcol": torch.tensor([[30, 30], [40, 20]]),
        "sequence": torch.randn(B, seq_len, F, generator=g),
        "target": torch.randn(B, 1, H, W, generator=g) * 0.3,
    }


def test_tx_heatmap_peaks_at_tx():
    hm = tx_heatmap(torch.tensor([[10, 20]]), 64, 64, 4)
    assert hm.shape == (1, 4, 64, 64)
    # narrowest scale peaks at (10,20)
    r, c = divmod(int(hm[0, 0].argmax()), 64)
    assert abs(r - 10) <= 1 and abs(c - 20) <= 1


def test_full_and_ablations_forward():
    batch = _fake_batch()
    for flags in [
        {}, {"use_building": False}, {"use_tx": False},
        {"use_order": False}, {"use_device": False},
    ]:
        model = CompassNet(CompassConfig(base=16, depth=3, **flags))
        out = model(batch)
        assert out.shape == (2, 1, 64, 64)
        assert torch.isfinite(out).all()


def test_loss_and_train_step():
    batch = _fake_batch()
    model = CompassNet(CompassConfig(base=16, depth=3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(batch)
    loss, comp = compass_loss(pred, batch, {"recon": 1.0, "traj": 0.5, "rcl": 0.1})
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    opt.step()
    for key in ("recon", "traj", "rcl", "total"):
        assert key in comp


def test_rcl_penalises_outward_increase():
    # a field that INCREASES away from TX should incur > RCL than one that decays
    B, H, W = 1, 64, 64
    tx = torch.tensor([[32, 32]])
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = torch.sqrt(((yy - 32) ** 2 + (xx - 32) ** 2).float())[None, None]
    building = torch.zeros(B, 1, H, W)
    increasing = (dist / dist.max())          # bright far away (unphysical)
    decaying = (1 - dist / dist.max())        # bright near TX (physical)
    assert ray_consistency_loss(increasing, building, tx) > ray_consistency_loss(decaying, building, tx)


def test_order_aware_sensitive_to_shuffle():
    batch = _fake_batch()
    model = CompassNet(CompassConfig(base=16, depth=3, use_order=True))
    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        b2 = dict(batch)
        b2["sequence"] = batch["sequence"][:, torch.randperm(batch["sequence"].size(1))]
        out2 = model(b2)
    assert (out1 - out2).abs().mean() > 1e-5  # order changes the prediction

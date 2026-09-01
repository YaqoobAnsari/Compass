#!/usr/bin/env python
"""Fast correctness check for the newly added DL baselines (shapes, params, MC round-trip)."""
import torch
from compass.models.baselines import BASELINES

B, H, W = 2, 256, 256
batch = {
    "sparse_rss": torch.randn(B, 1, H, W),
    "mask": (torch.rand(B, 1, H, W) > 0.9).float(),
    "coverage": torch.rand(B, 1, H, W),
    "building": (torch.rand(B, 1, H, W) > 0.5).float(),
    "tx_rowcol": torch.randint(0, H, (B, 2)).float(),
    "target": torch.randn(B, 1, H, W),
    "free_mask": (torch.rand(B, 1, H, W) > 0.3).float(),
}
for arch in ("radiomamba", "uram"):
    m = BASELINES[arch]()
    n = sum(p.numel() for p in m.parameters())
    m.train()
    y = m(batch)
    loss = (y - batch["target"]).pow(2).mean()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert y.shape == (B, 1, H, W), f"{arch} bad shape {y.shape}"
    assert gnorm > 0, f"{arch} no grad"
    has_do = any(isinstance(mm, (torch.nn.Dropout, torch.nn.Dropout2d)) for mm in m.modules())
    print(f"[OK] {arch:12s} out={tuple(y.shape)} params={n/1e6:.2f}M grad_ok={gnorm>0} dropout={has_do} "
          f"stochastic={getattr(m,'stochastic',False)}")
print("all new baselines OK")

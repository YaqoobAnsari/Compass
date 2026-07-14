"""
COMPASS learned reconstruction model (config-driven, ablation-ready).

Inputs (each toggleable so a single class realises the full model AND every
ablation, keeping comparisons fair):
  * sparse_rss, mask, coverage           — always on (the measurements)
  * building_map                         — innovation #1 (geometry)
  * tx multi-scale heatmap               — innovation #1 (source prior)
  * order-aware sequence embedding (FiLM) — innovation #3
  * device-offset head                   — innovation #4

Output: dense radio map (normalised). MC-dropout at inference -> uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .sequence import GRUDenoiser  # reuse the validated GRU as a sequence encoder
from .unet import ConditioningUNet


@dataclass
class CompassConfig:
    use_building: bool = True
    use_tx: bool = True
    use_order: bool = True       # sequence-aware (FiLM); False -> order-blind
    use_device: bool = True      # per-trajectory offset head
    tx_scales: int = 4
    base: int = 48
    depth: int = 4
    seq_in_dim: int = 11         # SEQ_FEATURES length
    seq_hidden: int = 64
    p_drop: float = 0.15


def tx_heatmap(tx_rowcol: torch.Tensor, H: int, W: int, scales: int) -> torch.Tensor:
    """Multi-scale Gaussian heatmap at the TX pixel. (B,2)->(B,scales,H,W)."""
    B = tx_rowcol.shape[0]
    device = tx_rowcol.device
    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    r = tx_rowcol[:, 0].view(B, 1, 1).float()
    c = tx_rowcol[:, 1].view(B, 1, 1).float()
    d2 = (ys - r) ** 2 + (xs - c) ** 2
    maps = []
    for i in range(scales):
        sigma = 4.0 * (2 ** i)
        maps.append(torch.exp(-0.5 * d2 / sigma ** 2))
    return torch.stack(maps, dim=1)  # (B, scales, H, W)


class SequenceEncoder(nn.Module):
    """Encode ordered trajectory features -> a global vector for FiLM (order-aware)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.gru(seq)                  # (B, N, 2H)
        pooled = y.mean(dim=1)                # permutation-SENSITIVE via GRU recurrence
        return self.proj(pooled)              # (B, hidden)


class DeviceHead(nn.Module):
    """Predict a per-sample additive offset (dB, normalised) from the measured RSS stats."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, sparse_rss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # simple, permutation-invariant device summary stats of the observed RSS
        B = sparse_rss.shape[0]
        obs = [sparse_rss[i][mask[i] > 0.5] for i in range(B)]
        stats = torch.stack([
            torch.stack([o.mean(), o.std() if o.numel() > 1 else torch.zeros((), device=o.device),
                         o.min(), o.max()]) if o.numel() > 0
            else torch.zeros(4, device=sparse_rss.device)
            for o in obs
        ])
        return self.net(stats)  # (B,1)


class CompassNet(nn.Module):
    def __init__(self, cfg: CompassConfig):
        super().__init__()
        self.cfg = cfg
        cin = 3  # sparse_rss, mask, coverage
        if cfg.use_building:
            cin += 1
        if cfg.use_tx:
            cin += cfg.tx_scales
        film_dim = cfg.seq_hidden if cfg.use_order else 0
        self.unet = ConditioningUNet(cin, base=cfg.base, depth=cfg.depth,
                                     film_dim=film_dim, p_drop=cfg.p_drop)
        if cfg.use_order:
            self.seq_enc = SequenceEncoder(cfg.seq_in_dim, cfg.seq_hidden)
        if cfg.use_device:
            self.device_head = DeviceHead()

    def forward(self, batch: dict) -> torch.Tensor:
        x = [batch["sparse_rss"], batch["mask"], batch["coverage"]]
        if self.cfg.use_building:
            x.append(batch["building"])
        if self.cfg.use_tx:
            H, W = batch["sparse_rss"].shape[-2:]
            x.append(tx_heatmap(batch["tx_rowcol"], H, W, self.cfg.tx_scales))
        x = torch.cat(x, dim=1)

        film_vec = None
        if self.cfg.use_order:
            film_vec = self.seq_enc(batch["sequence"])
        out = self.unet(x, film_vec)  # (B,1,H,W) normalised residual/map

        if self.cfg.use_device:
            off = self.device_head(batch["sparse_rss"], batch["mask"]).view(-1, 1, 1, 1)
            out = out + off
        return out

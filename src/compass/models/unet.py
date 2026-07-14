"""
Conditioning U-Net backbone for COMPASS learned reconstruction.

A compact, configurable encoder-decoder that maps a stack of conditioning
channels -> a dense radio map. Dropout in the bottleneck gives Monte-Carlo
uncertainty (the acquisition surface for active sensing) without a separate
ensemble. Kept deliberately standard so ablations differ only in their INPUTS /
losses, isolating each innovation's contribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int, p_drop: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    ]
    if p_drop > 0:
        layers.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*layers)


class ConditioningUNet(nn.Module):
    """U-Net: (B, Cin, H, W) -> (B, 1, H, W). FiLM hook for sequence conditioning."""

    def __init__(self, in_channels: int, base: int = 48, depth: int = 4,
                 film_dim: int = 0, p_drop: float = 0.15):
        super().__init__()
        self.depth = depth
        self.film_dim = film_dim
        chs = [base * (2 ** i) for i in range(depth)]

        self.inc = _block(in_channels, chs[0])
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(_block(chs[i], chs[i + 1], p_drop if i == depth - 2 else 0.0))
            self.pools.append(nn.MaxPool2d(2))

        self.mid = _block(chs[-1], chs[-1], p_drop)

        # FiLM: project a sequence/global vector to per-channel (scale, shift) at the bottleneck
        if film_dim > 0:
            self.film = nn.Linear(film_dim, 2 * chs[-1])

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.up_blocks.append(_block(chs[i - 1] * 2, chs[i - 1]))

        self.outc = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor, film_vec: torch.Tensor | None = None) -> torch.Tensor:
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down, pool in zip(self.downs, self.pools):
            h = down(pool(h))
            skips.append(h)
        h = self.mid(h)
        if self.film_dim > 0 and film_vec is not None:
            gamma, beta = self.film(film_vec).chunk(2, dim=-1)
            h = h * (1 + gamma[..., None, None]) + beta[..., None, None]
        for i, (up, blk) in enumerate(zip(self.ups, self.up_blocks)):
            h = up(h)
            skip = skips[-(i + 2)]
            h = blk(torch.cat([h, skip], dim=1))
        return self.outc(h)

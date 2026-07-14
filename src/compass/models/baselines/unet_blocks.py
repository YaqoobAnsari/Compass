"""Shared conv building blocks + input assembly for the DL radio-map baselines."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..compass_net import tx_heatmap


def assemble_input(batch: dict, use_building: bool, use_tx: bool, tx_scales: int = 4,
                   use_coverage: bool = True) -> torch.Tensor:
    """Build the input tensor from the shared batch dict — same channels COMPASS uses,
    so no baseline is starved of information it was designed to consume."""
    x = [batch["sparse_rss"], batch["mask"]]
    if use_coverage:
        x.append(batch["coverage"])
    if use_building:
        x.append(batch["building"])
    if use_tx:
        H, W = batch["sparse_rss"].shape[-2:]
        x.append(tx_heatmap(batch["tx_rowcol"], H, W, tx_scales))
    return torch.cat(x, dim=1)


def in_channels(use_building: bool, use_tx: bool, tx_scales: int = 4, use_coverage: bool = True) -> int:
    c = 2  # sparse_rss, mask
    c += 1 if use_coverage else 0
    c += 1 if use_building else 0
    c += tx_scales if use_tx else 0
    return c


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        p = dilation
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=p, dilation=dilation), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Plain UNet (encoder-decoder with skips). cin channels -> cout channels."""

    def __init__(self, cin: int, cout: int = 1, base: int = 48, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = cin
        chs = []
        for i in range(depth):
            oc = base * (2 ** i)
            self.downs.append(DoubleConv(c, oc))
            self.pools.append(nn.MaxPool2d(2))
            chs.append(oc)
            c = oc
        self.bottleneck = DoubleConv(c, c * 2)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        c = c * 2
        for i in reversed(range(depth)):
            oc = chs[i]
            self.upconvs.append(nn.ConvTranspose2d(c, oc, 2, stride=2))
            self.ups.append(DoubleConv(oc * 2, oc))
            c = oc
        self.head = nn.Conv2d(c, cout, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            x = up(torch.cat([x, skip], dim=1))
        return self.head(x)

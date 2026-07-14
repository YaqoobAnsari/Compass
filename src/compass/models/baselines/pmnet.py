"""
PMNet (Lee et al., 2023, arXiv:2211.10527) — winner of the 1st Pathloss Radio Map
Prediction Challenge.

ResNet-style encoder built from bottleneck residual blocks with dilated convolutions,
an ASPP (Atrous Spatial Pyramid Pooling) bottleneck aggregating multi-scale context
over several atrous rates, and a decoder that restores resolution via ConvTranspose
upsampling with skip connections. Inputs: building + TX (+ sparse measurements for our
reconstruction task, matching RadioUNet's 3-channel adaptation).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet_blocks import assemble_input, in_channels


class Bottleneck(nn.Module):
    """ResNet bottleneck block with an optional dilated 3x3 conv."""

    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        mid = cout // 4
        self.conv = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                    nn.BatchNorm2d(cout)) if (cin != cout or stride != 1) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.short(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling over several dilation rates + image pooling."""

    def __init__(self, cin, cout, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 3 if r > 1 else 1, padding=r if r > 1 else 0,
                                    dilation=r, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
            for r in rates
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(cout * (len(rates) + 1), cout, 1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        p = self.pool(x)
        p = nn.functional.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(p)
        return self.project(torch.cat(feats, dim=1))


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.fuse = nn.Sequential(nn.Conv2d(cout + cskip, cout, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class PMNet(nn.Module):
    use_building = True
    use_tx = True

    def __init__(self, base: int = 48, tx_scales: int = 4):
        super().__init__()
        self.tx_scales = tx_scales
        cin = in_channels(self.use_building, self.use_tx, tx_scales)
        self.stem = nn.Sequential(nn.Conv2d(cin, base, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.enc1 = Bottleneck(base, base * 2, stride=2)          # /2
        self.enc2 = Bottleneck(base * 2, base * 4, stride=2)      # /4
        self.enc3 = Bottleneck(base * 4, base * 8, stride=2, dilation=2)  # /8, dilated
        self.aspp = ASPP(base * 8, base * 8)
        self.up3 = _Up(base * 8, base * 4, base * 4)
        self.up2 = _Up(base * 4, base * 2, base * 2)
        self.up1 = _Up(base * 2, base, base)
        self.head = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(base, 1, 1))
        self.aux = None

    def forward(self, batch: dict):
        x = assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.aspp(s3)
        d = self.up3(b, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        return self.head(d)

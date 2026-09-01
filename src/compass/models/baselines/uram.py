"""
URAM — an Uncertainty-aware Radio-map baseline (Bayesian U-Net), representing the
recent line of work that couples radio-map reconstruction with calibrated predictive
uncertainty on sparse / crowdsensed measurements (URAM, GeoUQ-style nets). A U-Net
inpainter consumes the sparse RSS field and its observation mask (plus building + TX,
matching the other baselines) and is made Bayesian with Monte-Carlo dropout in both
encoder and decoder, so repeated stochastic forward passes yield an epistemic
uncertainty map. An auxiliary log-variance head models aleatoric uncertainty; only the
mean prediction is scored for RMSE so the comparison stays on the same footing as the
deterministic baselines.

The distinguishing feature over the plain CNNs is not the point predictor but the
uncertainty mechanism, so this baseline tests whether an uncertainty-first design costs
point accuracy on the dense benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet_blocks import assemble_input, in_channels


class DropConv(nn.Module):
    """DoubleConv with spatial dropout after each activation (kept active at inference
    for MC sampling)."""

    def __init__(self, cin: int, cout: int, p: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(p),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(p),
        )

    def forward(self, x):
        return self.net(x)


class URAM(nn.Module):
    """Bayesian U-Net with MC-dropout and an aleatoric log-variance head."""

    use_building = True
    use_tx = True
    stochastic = True   # tell the eval wrapper to run multiple MC passes

    def __init__(self, base: int = 32, depth: int = 4, tx_scales: int = 4, p: float = 0.15):
        super().__init__()
        self.tx_scales = tx_scales
        cin = in_channels(self.use_building, self.use_tx, tx_scales)
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = cin
        chs = []
        for i in range(depth):
            oc = base * (2 ** i)
            self.downs.append(DropConv(c, oc, p))
            self.pools.append(nn.MaxPool2d(2))
            chs.append(oc)
            c = oc
        self.bottleneck = DropConv(c, c * 2, p)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        c = c * 2
        for i in reversed(range(depth)):
            oc = chs[i]
            self.upconvs.append(nn.ConvTranspose2d(c, oc, 2, stride=2))
            self.ups.append(DropConv(oc * 2, oc, p))
            c = oc
        self.head = nn.Conv2d(c, 1, 1)      # mean
        self.logvar = nn.Conv2d(c, 1, 1)    # aleatoric log-variance (aux, not RMSE-scored)
        self.aux = None
        # keep a handle so training could add a heteroscedastic term later if desired
        self.last_logvar = None

    def forward(self, batch: dict):
        x = assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips)):
            x = upconv(x)
            x = up(torch.cat([x, skip], dim=1))
        self.last_logvar = self.logvar(x)
        return self.head(x)

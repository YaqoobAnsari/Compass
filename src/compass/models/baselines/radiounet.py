"""
RadioUNet (Levie, Yapar, Kutyniok, Caire — IEEE TWC 2021, arXiv:1911.09002).

WNet: two cascaded UNets (U+U = W). The first UNet predicts a coarse radio map from
the inputs; the second refines it, taking the same inputs PLUS the first UNet's
output as an extra channel. The paper's 3-channel input is exactly our setting:
building occupancy + TX location + sparse pathloss measurements.

Faithful details:
  * 3-channel input (building + TX + samples) — here TX is COMPASS's multi-scale
    heatmap (a smoother superset of the paper's 1-hot TX mask) for a fair comparison.
  * Cascade refinement (WNet). We train end-to-end with DEEP SUPERVISION (loss on
    both the coarse and refined outputs), a standard alternative to the paper's
    freeze-stage-1 curriculum that trains in one pass; noted in the findings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet_blocks import UNet, assemble_input, in_channels


class RadioUNet(nn.Module):
    use_building = True
    use_tx = True

    def __init__(self, base: int = 24, depth: int = 4, tx_scales: int = 4):
        super().__init__()
        self.tx_scales = tx_scales
        cin = in_channels(self.use_building, self.use_tx, tx_scales)
        self.unet1 = UNet(cin, cout=1, base=base, depth=depth)          # coarse
        self.unet2 = UNet(cin + 1, cout=1, base=base, depth=depth)      # refine (+ coarse)
        self.aux = None  # coarse output, for deep supervision in the trainer

    def forward(self, batch: dict):
        x = assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)
        coarse = self.unet1(x)
        refined = self.unet2(torch.cat([x, coarse], dim=1))
        self.aux = coarse  # trainer adds 0.5 * MSE(coarse, target)
        return refined

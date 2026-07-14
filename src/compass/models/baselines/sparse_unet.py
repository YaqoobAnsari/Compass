"""
SparseUNet — generic deep sparse-to-dense interpolation baseline.

The plainest DL reference: a single UNet that sees ONLY the measurements
(sparse_rss, mask, coverage) — no building, no TX. Isolates "does a generic deep
interpolator beat classical interpolation, and how far below geometry-aware
COMPASS does it sit?" Any gap COMPASS opens over this is attributable to geometry.
"""

from __future__ import annotations

import torch.nn as nn

from .unet_blocks import UNet, assemble_input, in_channels


class SparseUNet(nn.Module):
    use_building = False
    use_tx = False

    def __init__(self, base: int = 32, depth: int = 4, p_drop: float = 0.15):
        super().__init__()
        cin = in_channels(self.use_building, self.use_tx)
        self.unet = UNet(cin, cout=1, base=base, depth=depth)
        self.drop = nn.Dropout2d(p_drop)
        self.aux = None

    def forward(self, batch: dict):
        x = assemble_input(batch, self.use_building, self.use_tx)
        # dropout on the input-side feature via a light stem for MC-dropout uncertainty
        return self.unet(self.drop(x) if self.training else x)

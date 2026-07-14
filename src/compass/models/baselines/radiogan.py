"""
RadioGAN — conditional GAN radio-map baseline (pix2pix-style; the generative-
adversarial family, e.g. RME-GAN). A UNet generator maps building+TX+samples → radio
map; a PatchGAN discriminator judges (condition, map) patches. Trained with LSGAN +
L1 reconstruction. At inference the generator IS the reconstruction (forward(batch)),
so it plugs into the standard benchmark like the other baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet_blocks import UNet, assemble_input, in_channels


class PatchDiscriminator(nn.Module):
    def __init__(self, cin, base=48):
        super().__init__()
        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 4, stride=s, padding=1),
                                 nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(
            nn.Conv2d(cin, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            blk(base, base * 2), blk(base * 2, base * 4), blk(base * 4, base * 4, s=1),
            nn.Conv2d(base * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, cond, y):
        return self.net(torch.cat([cond, y], dim=1))


class RadioGAN(nn.Module):
    use_building = True
    use_tx = True

    def __init__(self, base: int = 30, depth: int = 4, tx_scales: int = 4):
        super().__init__()
        self.tx_scales = tx_scales
        self.cond_c = in_channels(self.use_building, self.use_tx, tx_scales)
        self.gen = UNet(self.cond_c, cout=1, base=base, depth=depth)
        self.disc = PatchDiscriminator(self.cond_c + 1)
        self.aux = None

    def cond(self, batch):
        return assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)

    def forward(self, batch: dict):
        return self.gen(self.cond(batch))  # generator = reconstruction at inference

"""
COMPASS-WNet — COMPASS's conditioning (building/TX/order/device + MC-dropout
uncertainty) on a WNet cascade backbone (two ConditioningUNets, deep-supervised),
motivated by Phase G: RadioUNet's WNet backbone out-reconstructs a single UNet on
synthetic point accuracy, so we graft our orthogonal innovations onto the stronger
backbone to reclaim accuracy AND keep uncertainty + the real-data innovations.

Stage 1: conditioned inputs -> coarse map. Stage 2: [inputs + coarse] -> refined map.
Both stages share the order-FiLM vector; the device offset is applied to the refined
output. ``self.aux`` exposes the coarse map for deep supervision in the trainer.
Drop-in: same batch interface and CompassConfig as CompassNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .compass_net import (CompassConfig, DeviceHead, SequenceEncoder, occlusion_field,
                          measurement_occlusion_field, tx_heatmap)
from .unet import ConditioningUNet


class CompassWNet(nn.Module):
    def __init__(self, cfg: CompassConfig):
        super().__init__()
        self.cfg = cfg
        cin = 3  # sparse_rss, mask, coverage
        if cfg.use_building:
            cin += 1
        if cfg.use_tx:
            cin += cfg.tx_scales
        if cfg.use_occlusion:
            cin += 1
        film_dim = cfg.seq_hidden if cfg.use_order else 0
        # base scaled down so two UNets ~ match the single-UNet capacity budget
        # (multiple of 8 for the UNet's GroupNorm)
        b = max(16, (int(round(cfg.base * 0.72)) // 8) * 8)
        self.unet1 = ConditioningUNet(cin, base=b, depth=cfg.depth, film_dim=film_dim, p_drop=cfg.p_drop)
        self.unet2 = ConditioningUNet(cin + 1, base=b, depth=cfg.depth, film_dim=film_dim, p_drop=cfg.p_drop)
        if cfg.use_order:
            self.seq_enc = SequenceEncoder(cfg.seq_in_dim, cfg.seq_hidden)
        if cfg.use_device:
            self.device_head = DeviceHead()
        self.aux = None

    def _inputs(self, batch: dict) -> torch.Tensor:
        x = [batch["sparse_rss"], batch["mask"], batch["coverage"]]
        if self.cfg.use_building:
            x.append(batch["building"])
        if self.cfg.use_tx:
            H, W = batch["sparse_rss"].shape[-2:]
            x.append(tx_heatmap(batch["tx_rowcol"], H, W, self.cfg.tx_scales))
        if self.cfg.use_occlusion:
            if self.cfg.occlusion_anchor == "measurement":
                x.append(measurement_occlusion_field(
                    batch["building"], batch["mask"], n_anchors=self.cfg.occlusion_anchors))
            else:
                x.append(occlusion_field(batch["building"], batch["tx_rowcol"]))
        return torch.cat(x, dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._inputs(batch)
        film_vec = self.seq_enc(batch["sequence"]) if self.cfg.use_order else None
        coarse = self.unet1(x, film_vec)
        refined = self.unet2(torch.cat([x, coarse], dim=1), film_vec)
        if self.cfg.use_device:
            off = self.device_head(batch["sparse_rss"], batch["mask"]).view(-1, 1, 1, 1)
            refined = refined + off
        self.aux = coarse  # deep supervision (0.5 * recon on coarse)
        return refined

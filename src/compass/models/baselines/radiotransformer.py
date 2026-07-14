"""
RadioTransformer — a hybrid CNN-Transformer radio-map baseline (TransUNet-style),
representing the transformer family now prominent for radio-map / pathloss prediction
(e.g. ViT-based indoor pathloss nets, RMTransformer). A CNN encoder downsamples to a
low-resolution token grid, transformer encoder blocks model global context via
self-attention, and a CNN decoder with skip connections restores full resolution.
Inputs: building + TX + sparse measurements (same 3-channel setting as the CNN
baselines), so the comparison isolates the attention-based backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet_blocks import DoubleConv, assemble_input, in_channels


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp=4, p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(dim * mlp, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class RadioTransformer(nn.Module):
    use_building = True
    use_tx = True

    def __init__(self, base: int = 24, depth: int = 4, tx_scales: int = 4,
                 dim: int = 192, layers: int = 4, heads: int = 8):
        super().__init__()
        self.tx_scales = tx_scales
        cin = in_channels(self.use_building, self.use_tx, tx_scales)
        # CNN encoder (256 -> 16), keep skips
        self.inc = DoubleConv(cin, base)
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        chs = [base]
        c = base
        for i in range(depth):
            oc = base * (2 ** (i + 1))
            self.downs.append(DoubleConv(c, oc))
            self.pools.append(nn.MaxPool2d(2))
            chs.append(oc)
            c = oc
        # transformer over the 16x16 token grid
        self.to_tok = nn.Conv2d(c, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, (256 // (2 ** depth)) ** 2, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.from_tok = nn.Conv2d(dim, c, 1)
        # CNN decoder with skips
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            oc = chs[i]
            self.ups.append(nn.ConvTranspose2d(c, oc, 2, stride=2))
            self.up_blocks.append(DoubleConv(oc * 2, oc))
            c = oc
        self.head = nn.Conv2d(c, 1, 1)
        self.aux = None

    def forward(self, batch: dict):
        x = assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)
        x = self.inc(x)                       # (B, base, 256, 256)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            skips.append(x)                   # skip at current resolution (pre-downsample)
            x = down(pool(x))                 # downsample then conv -> next channel width
        # x is the bottleneck at 256/2^depth
        B, C, H, W = x.shape
        t = self.to_tok(x).flatten(2).transpose(1, 2)   # (B, HW, dim)
        t = t + self.pos[:, : t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        y = self.from_tok(t.transpose(1, 2).reshape(B, -1, H, W))  # (B, C, H, W)
        for up, ub, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            y = ub(torch.cat([up(y), skip], dim=1))
        return self.head(y)

"""
RadioMamba — a hybrid Mamba-UNet radio-map baseline, representing the state-space
(SSM) family that has recently been applied to radio-map / pathloss prediction
(RadioMamba, RadaMamba, and the broader Vision-Mamba line). A CNN encoder downsamples
to a low-resolution token grid, bidirectional selective-scan (S6 / Mamba) blocks model
global context in linear time, and a CNN decoder with skip connections restores full
resolution. This is the SSM analogue of the RadioTransformer baseline (self-attention
replaced by input-dependent state-space scanning), so the comparison isolates the
backbone. Inputs: building + TX + sparse measurements (same setting as the CNN
baselines).

The selective scan is implemented in pure PyTorch (the ``mamba_ssm`` CUDA kernel is not
required), keeping the input-dependent (B, C, delta) selectivity, the diagonal state
matrix A, and the sequential recurrence that define the Mamba mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import DoubleConv, assemble_input, in_channels


class SelectiveScan(nn.Module):
    """Pure-PyTorch S6 (Mamba) selective scan over a 1D token sequence.

    Faithful to the Mamba core: input-dependent B, C and step size delta (the
    "selective" part), a diagonal negative state matrix A, zero-order-hold
    discretisation, and a sequential state recurrence h_t = dA_t h_{t-1} + dB_t x_t.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, self.d_inner // 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)          # -> x, z (gate)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)  # -> dt, B, C
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))                       # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _scan(self, x):
        """x: (B, L, d_inner) -> (B, L, d_inner) via the selective recurrence."""
        Bsz, L, _ = x.shape
        A = -torch.exp(self.A_log)                                    # (d_inner, d_state)
        dbc = self.x_proj(x)                                          # (B, L, dt_rank+2N)
        dt, Bmat, Cmat = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                          # (B, L, d_inner)
        # discretise: dA = exp(delta * A), dB_x = delta * B * x
        dA = torch.exp(delta.unsqueeze(-1) * A)                       # (B, L, d_inner, N)
        dBx = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, d_inner, N)
        h = x.new_zeros(Bsz, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]                              # (B, d_inner, N)
            ys.append(torch.einsum("bdn,bn->bd", h, Cmat[:, t]))     # (B, d_inner)
        y = torch.stack(ys, dim=1)                                    # (B, L, d_inner)
        return y + x * self.D

    def forward(self, x):
        # x: (B, L, d_model)
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)                                   # each (B, L, d_inner)
        # depthwise causal conv over the sequence
        xc = self.conv1d(xi.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        xc = F.silu(xc)
        # bidirectional scan (vision): forward + reversed, summed
        y = self._scan(xc) + self._scan(xc.flip(1)).flip(1)
        y = y * F.silu(z)                                             # gating
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, mlp=2, p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = SelectiveScan(dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(dim * mlp, dim))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class RadioMamba(nn.Module):
    """Hybrid Mamba-UNet: CNN encoder/decoder with bidirectional SSM blocks at the
    bottleneck. Mirrors the RadioTransformer skeleton so the only difference from the
    attention baseline is the global-context mixer (state-space vs self-attention)."""

    use_building = True
    use_tx = True

    def __init__(self, base: int = 24, depth: int = 4, tx_scales: int = 4,
                 dim: int = 192, layers: int = 4, d_state: int = 16):
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
        # Mamba over the 16x16 token grid (linear-time global context)
        self.to_tok = nn.Conv2d(c, dim, 1)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state=d_state) for _ in range(layers)])
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
        x = self.inc(x)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            skips.append(x)
            x = down(pool(x))
        B, C, H, W = x.shape
        t = self.to_tok(x).flatten(2).transpose(1, 2)                # (B, HW, dim)
        for blk in self.blocks:
            t = blk(t)
        y = self.from_tok(t.transpose(1, 2).reshape(B, -1, H, W))
        for up, ub, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            y = ub(torch.cat([up(y), skip], dim=1))
        return self.head(y)

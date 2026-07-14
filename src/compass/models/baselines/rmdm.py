"""
RMDM — Radio Map Diffusion Model (Jia et al., 2025, arXiv:2501.19160).

Faithful two-stage design: (1) a coarse U-Net that produces a physics-consistent
estimate from building + TX + sparse measurements, (2) a conditional DDPM that
refines it by denoising, conditioned on the inputs + coarse map. We keep the
architecture (dual-UNet, coarse + diffusion refinement) and train it on the same
data/budget as every other method. We approximate the paper's PINN/Helmholtz stage-1
constraint with the coarse reconstruction objective (a full Helmholtz PINN is out of
scope for a baseline; noted honestly in the findings).

eps-prediction DDPM, cosine schedule; DDIM sampling at inference. Stochastic sampling
gives a natural ensemble → mean + uncertainty (like COMPASS's MC-dropout).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .unet_blocks import UNet, assemble_input, in_channels


def _cosine_alpha_bar(T: int) -> torch.Tensor:
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(1e-5, 1.0)  # (T+1,)


class SinTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class TimeFiLMUNet(nn.Module):
    """Compact UNet with time-embedding FiLM for eps prediction. cin -> 1 (noise)."""

    def __init__(self, cin: int, base: int = 48, tdim: int = 128):
        super().__init__()
        self.temb = SinTimeEmb(tdim)
        self.enc1 = nn.Sequential(nn.Conv2d(cin, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU())
        self.film2 = nn.Linear(tdim, base * 2)
        self.film3 = nn.Linear(tdim, base * 4)
        self.mid = nn.Sequential(nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU())
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU())
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x, t):
        te = self.temb(t)
        e1 = self.enc1(x)
        e2 = self.enc2(e1) + self.film2(te)[:, :, None, None]
        e3 = self.enc3(e2) + self.film3(te)[:, :, None, None]
        m = self.mid(e3)
        d3 = self.up3(m)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class RMDM(nn.Module):
    use_building = True
    use_tx = True
    stochastic = True  # signals the wrapper to draw an ensemble via sampling

    def __init__(self, base: int = 32, tx_scales: int = 4, T: int = 1000, sample_steps: int = 25):
        super().__init__()
        self.tx_scales = tx_scales
        self.T = T
        self.sample_steps = sample_steps
        cond_c = in_channels(self.use_building, self.use_tx, tx_scales)
        self.coarse_net = UNet(cond_c, cout=1, base=base, depth=4)
        self.eps_net = TimeFiLMUNet(cond_c + 2, base=base)  # cond + coarse + noisy_x
        self.register_buffer("alpha_bar", _cosine_alpha_bar(T))
        self.aux = None  # unused (RMDM has its own training path)

    def _cond(self, batch):
        return assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)

    def training_losses(self, batch):
        cond = self._cond(batch)
        target = batch["target"]
        coarse = self.coarse_net(cond)
        B = target.size(0)
        t = torch.randint(1, self.T + 1, (B,), device=target.device)
        ab = self.alpha_bar[t][:, None, None, None]
        eps = torch.randn_like(target)
        x_t = ab.sqrt() * target + (1 - ab).sqrt() * eps
        eps_hat = self.eps_net(torch.cat([x_t, cond, coarse], dim=1), t)
        loss_diff = ((eps_hat - eps) ** 2).mean()
        loss_coarse = ((coarse - target) ** 2).mean()
        return loss_diff + loss_coarse, {"diff": float(loss_diff), "coarse": float(loss_coarse)}

    @torch.no_grad()
    def forward(self, batch):
        """DDIM sampling conditioned on inputs + coarse → refined map (B,1,H,W)."""
        cond = self._cond(batch)
        coarse = self.coarse_net(cond)
        B, _, H, W = coarse.shape
        x = torch.randn(B, 1, H, W, device=cond.device)
        steps = torch.linspace(self.T, 1, self.sample_steps, device=cond.device).long()
        for i, t in enumerate(steps):
            tt = t.expand(B)
            ab_t = self.alpha_bar[t]
            eps_hat = self.eps_net(torch.cat([x, cond, coarse], dim=1), tt)
            x0 = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0 = x0.clamp(-1.2, 1.2)
            if i < len(steps) - 1:
                ab_next = self.alpha_bar[steps[i + 1]]
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps_hat
            else:
                x = x0
        return x

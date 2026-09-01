"""
RadioDiff (Wang et al., IEEE TCCN 2024, arXiv:2408.08593) — a decoupled,
sampling-free generative diffusion model for radio-map construction, and the current
reference point for the diffusion family.

Two mechanisms define it, and both are implemented here:

  1. Decoupled diffusion (DDM). The forward process is split into an explicit
     data-attenuation term and a noise term, x_t = (1 - t) x_0 + sqrt(t) eps for a
     continuous t in [0, 1]. The network regresses BOTH the clean map and the noise,
     so the reverse process has a closed form, x_s = (1 - s) x0_hat + sqrt(s) eps_hat,
     and setting s = 0 recovers the map in a SINGLE step. This is the "sampling-free"
     property, and it is why RadioDiff is stable where an iterative eps-only pixel
     DDPM (our RMDM baseline) has to hallucinate the whole field from noise.
  2. Adaptive FFT filtering (AFT). Feature maps are filtered in the frequency domain
     by a filter predicted from global context, a per-channel complex gain plus a
     radial band modulation, which sharpens the high-frequency structure (shadow
     edges at building boundaries) that plain convolutional decoders blur.

Scope note, stated honestly in the write-up: the published model runs the diffusion in
the latent space of a pre-trained VAE. We run it in pixel space, because introducing an
externally pre-trained autoencoder would break the "same data, same budget" fairness
guarantee that every other baseline in this panel is held to. The decoupled diffusion
and the adaptive frequency filter, which are the contributions that distinguish
RadioDiff, are kept intact. This mirrors how we approximated RMDM's Helmholtz PINN
stage with a coarse reconstruction objective.

Inputs are building + TX + sparse measurements, the same channels every other
conditioned baseline consumes. Trained via the shared diffusion trainer
(`compass.training.train_rmdm`), which optimises `training_losses`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import assemble_input, in_channels


class SinTimeEmb(nn.Module):
    """Sinusoidal embedding for the continuous diffusion time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000.0)[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class AdaptiveFFTFilter(nn.Module):
    """RadioDiff's adaptive frequency filter.

    The feature map is taken to the frequency domain, multiplied by a filter that is
    PREDICTED from global context (a per-channel complex gain and a set of radial
    frequency-band weights), and taken back. The band weights are defined on the
    normalised frequency radius, so the module is resolution independent. The output
    projection is zero-initialised, so the block starts as an identity and learns the
    filtering it needs.
    """

    def __init__(self, ch: int, bands: int = 8):
        super().__init__()
        self.bands = bands
        self.ctx = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(ch, ch), nn.SiLU())
        self.gain = nn.Linear(ch, ch * 2)      # per-channel complex gain
        self.band = nn.Linear(ch, bands)       # radial band modulation
        self.norm = nn.GroupNorm(8, ch)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        c = self.ctx(x)
        g = self.gain(c).view(B, C, 2)
        bw = torch.softmax(self.band(c), dim=-1)                     # (B, bands)

        f = torch.fft.rfft2(x.float(), norm="ortho")                 # (B, C, H, W//2+1)
        fy = torch.fft.fftfreq(H, device=x.device).abs()[:, None]
        fx = torch.fft.rfftfreq(W, device=x.device).abs()[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2)
        r = r / (r.max() + 1e-8)
        idx = (r * self.bands).clamp(0, self.bands - 1e-3).long()     # (H, W//2+1)
        bmap = bw[:, idx]                                            # (B, H, W//2+1)

        gain = torch.complex(g[..., 0], g[..., 1])[:, :, None, None]  # (B, C, 1, 1)
        f = f * gain * bmap[:, None].to(f.dtype)
        y = torch.fft.irfft2(f, s=(H, W), norm="ortho").to(x.dtype)
        return x + self.proj(self.norm(y))


class ResBlock(nn.Module):
    """Residual block with time-embedding conditioning."""

    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(t)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CondBranch(nn.Module):
    """Condition encoder: building + TX + sparse measurements to a bottleneck feature,
    finished with an adaptive frequency filter."""

    def __init__(self, cond_c: int, base: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_c, base, 3, stride=2, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU(),
        )
        self.aft = AdaptiveFFTFilter(base * 4)

    def forward(self, c):
        return self.aft(self.net(c))


class RadioDiff(nn.Module):
    """Decoupled sampling-free diffusion radio-map model."""

    use_building = True
    use_tx = True
    stochastic = False   # sampling-free: the map is produced in one deterministic jump

    def __init__(self, base: int = 64, tx_scales: int = 4, tdim: int = 128,
                 sample_steps: int = 1, lambda_noise: float = 1.0):
        super().__init__()
        self.tx_scales = tx_scales
        self.sample_steps = sample_steps
        self.lambda_noise = lambda_noise
        cond_c = in_channels(self.use_building, self.use_tx, tx_scales)
        self.temb = SinTimeEmb(tdim)
        self.cond_branch = CondBranch(cond_c, base)

        self.inc = nn.Conv2d(1 + cond_c, base, 3, padding=1)
        self.d1 = ResBlock(base, base, tdim)
        self.p1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.d2 = ResBlock(base, base * 2, tdim)
        self.p2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.d3 = ResBlock(base * 2, base * 4, tdim)
        self.p3 = nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1)

        self.m1 = ResBlock(base * 4, base * 4, tdim)
        self.aft = AdaptiveFFTFilter(base * 4)
        self.m2 = ResBlock(base * 4, base * 4, tdim)

        self.u3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.b3 = ResBlock(base * 8, base * 2, tdim)
        self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.b2 = ResBlock(base * 4, base, tdim)
        self.u1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.b1 = ResBlock(base * 2, base, tdim)

        self.out_norm = nn.GroupNorm(8, base)
        self.x0_head = nn.Conv2d(base, 1, 1)    # clean-map branch
        self.eps_head = nn.Conv2d(base, 1, 1)   # noise branch
        self.aux = None

    def _cond(self, batch: dict) -> torch.Tensor:
        return assemble_input(batch, self.use_building, self.use_tx, self.tx_scales)

    def predict(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """Return (x0_hat, eps_hat), the two decoupled components."""
        te = self.temb(t)
        cf = self.cond_branch(cond)
        h = self.inc(torch.cat([x_t, cond], dim=1))
        s1 = self.d1(h, te)
        s2 = self.d2(self.p1(s1), te)
        s3 = self.d3(self.p2(s2), te)
        m = self.m1(self.p3(s3) + cf, te)       # condition injected at the bottleneck
        m = self.aft(m)
        m = self.m2(m, te)
        y = self.b3(torch.cat([self.u3(m), s3], dim=1), te)
        y = self.b2(torch.cat([self.u2(y), s2], dim=1), te)
        y = self.b1(torch.cat([self.u1(y), s1], dim=1), te)
        y = F.silu(self.out_norm(y))
        return self.x0_head(y), self.eps_head(y)

    def training_losses(self, batch: dict):
        """Decoupled objective: regress the clean map AND the noise at a random t."""
        cond = self._cond(batch)
        x0 = batch["target"]
        B = x0.size(0)
        t = torch.rand(B, device=x0.device).clamp(1e-3, 1.0)
        eps = torch.randn_like(x0)
        tt = t[:, None, None, None]
        x_t = (1.0 - tt) * x0 + tt.sqrt() * eps
        x0_hat, eps_hat = self.predict(x_t, cond, t)
        l_data = ((x0_hat - x0) ** 2).mean()
        l_noise = ((eps_hat - eps) ** 2).mean()
        loss = l_data + self.lambda_noise * l_noise
        return loss, {"data": float(l_data), "noise": float(l_noise)}

    @torch.no_grad()
    def forward(self, batch: dict) -> torch.Tensor:
        """Sampling-free generation: analytic jumps from t=1 to t=0 (one step by default)."""
        cond = self._cond(batch)
        B, _, H, W = batch["sparse_rss"].shape
        x = torch.randn(B, 1, H, W, device=cond.device, dtype=cond.dtype)
        ts = torch.linspace(1.0, 0.0, self.sample_steps + 1, device=cond.device)
        for i in range(self.sample_steps):
            t = ts[i].expand(B)
            x0_hat, eps_hat = self.predict(x, cond, t)
            s = ts[i + 1]
            x = (1.0 - s) * x0_hat + s.sqrt() * eps_hat   # s=0 on the last step -> x0_hat
        return x

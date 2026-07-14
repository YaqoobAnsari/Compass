"""
COMPASS training losses.

  * masked_reconstruction_loss — accuracy where it matters (free space, weighted
    toward observed regions but learning everywhere).
  * trajectory_consistency_loss — prediction must match the measurement at
    observed pixels (don't denoise the conditioning away).
  * ray_consistency_loss (RCL, innovation #2) — PHYSICS prior using real geometry:
    along rays cast from the true TX, predicted signal must not increase outward
    (decay) and must drop further behind buildings (occlusion/shadowing). This is
    a per-ray, geometry-grounded constraint — unlike TrajectoryDiff's single
    global mean inequality, and unlike Helmholtz (global PDE, dense-only).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_reconstruction_loss(pred, target, free_mask, obs_mask, w_obs=2.0, w_free=1.0, w_bldg=0.1):
    """Weighted MSE: emphasise free space, with extra weight at observed pixels."""
    w = torch.full_like(target, w_bldg)
    w = torch.where(free_mask > 0.5, torch.full_like(w, w_free), w)
    w = torch.where(obs_mask > 0.5, torch.full_like(w, w_obs), w)
    return (w * (pred - target) ** 2).sum() / w.sum().clamp_min(1.0)


def trajectory_consistency_loss(pred, sparse_rss, mask):
    """MSE between prediction and measured RSS at observed pixels only."""
    m = mask > 0.5
    if m.sum() == 0:
        return pred.new_zeros(())
    return ((pred - sparse_rss)[m] ** 2).mean()


def _sample_along_rays(field, tx_rowcol, n_rays, n_steps, max_radius):
    """Bilinear-sample `field` (B,1,H,W) along n_rays bearings from TX -> (B, n_rays, n_steps)."""
    B, _, H, W = field.shape
    device = field.device
    ang = torch.linspace(0, 2 * torch.pi, n_rays + 1, device=device)[:-1]      # (R,)
    rad = torch.linspace(1.0, max_radius, n_steps, device=device)              # (S,)
    dr = (torch.sin(ang)[:, None] * rad[None, :])                              # (R,S)
    dc = (torch.cos(ang)[:, None] * rad[None, :])
    r = tx_rowcol[:, 0].view(B, 1, 1) + dr[None]                               # (B,R,S)
    c = tx_rowcol[:, 1].view(B, 1, 1) + dc[None]
    gy = (r / (H - 1)) * 2 - 1
    gx = (c / (W - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)                                       # (B,R,S,2)
    samp = F.grid_sample(field, grid, align_corners=True, padding_mode="border")  # (B,1,R,S)
    return samp[:, 0]                                                          # (B,R,S)


def ray_consistency_loss(pred, building, tx_rowcol, n_rays=64, n_steps=48,
                         max_radius=200.0, shadow_weight=3.0):
    """Decay + occlusion prior along TX-rays. `building` is 1=building, 0=street."""
    p = _sample_along_rays(pred, tx_rowcol, n_rays, n_steps, max_radius)        # (B,R,S)
    b = _sample_along_rays(building, tx_rowcol, n_rays, n_steps, max_radius)    # (B,R,S) ~[0,1]
    d_pred = p[:, :, 1:] - p[:, :, :-1]                                         # step-to-step change
    occ = b[:, :, 1:]                                                           # building-ness of the outer step
    # signal must not increase outward; penalise harder when stepping into a building (shadow)
    weight = 1.0 + shadow_weight * occ
    return (weight * F.relu(d_pred) ** 2).mean()


def compass_loss(pred, batch, weights, use_rcl=True):
    """Total loss + component dict. `pred`, target in normalised [-1,1]."""
    comp = {}
    comp["recon"] = masked_reconstruction_loss(
        pred, batch["target"], batch["free_mask"], batch["mask"])
    comp["traj"] = trajectory_consistency_loss(pred, batch["sparse_rss"], batch["mask"])
    if use_rcl:
        comp["rcl"] = ray_consistency_loss(pred, batch["building"], batch["tx_rowcol"])
    else:
        comp["rcl"] = pred.new_zeros(())
    total = (weights["recon"] * comp["recon"]
             + weights["traj"] * comp["traj"]
             + weights["rcl"] * comp["rcl"])
    comp["total"] = total
    return total, comp

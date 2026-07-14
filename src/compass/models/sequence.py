"""
Small sequence models for the Phase-2.1 make-or-break experiment.

Both take per-point trajectory features (B, N, F) and output a per-point
prediction (B, N) — here, the denoised RSS. The contrast isolates the value of
ORDER:

  * ``GRUDenoiser``  — a bidirectional GRU: order-AWARE (uses adjacency).
  * ``SetDenoiser``  — per-point MLP + permutation-invariant mean-pool context:
    order-BLIND control (can estimate set-level things like a per-device offset,
    but cannot use the sequence ordering).

If the GRU beats the Set model (and beats itself under shuffled order) only when
the noise is spatially correlated, then order carries usable signal — the
Pillar-1 hypothesis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDenoiser(nn.Module):
    """Order-aware per-point denoiser (bidirectional GRU)."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, F) -> (B, N)
        h = F.silu(self.inp(x))
        y, _ = self.gru(h)
        return self.out(y).squeeze(-1)


class SetDenoiser(nn.Module):
    """Order-blind control: per-point MLP + permutation-invariant global context."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, F) -> (B, N)
        h = self.enc(x)
        ctx = h.mean(dim=1, keepdim=True).expand_as(h)  # permutation-invariant
        return self.out(torch.cat([h, ctx], dim=-1)).squeeze(-1)

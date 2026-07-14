"""
Reconstructor interface for COMPASS.

A Reconstructor turns sparse trajectory observations into a dense radio-map
estimate over free space, plus a per-pixel UNCERTAINTY map that the active-
sensing loop consumes as its acquisition surface. Keeping this an interface lets
the classical GP (now) and the learned/diffusion reconstructor (later) plug into
the SAME active loop and the SAME evaluation harness, and lets the data source
swap from RadioMapSeer to UniCellular without touching the loop.

Conventions (shared with compass.data):
  * coordinates are (row, col) integer pixels on a HxW grid;
  * values are RSS / gain in dBm;
  * predictions are returned ONLY meaningfully over free space; building pixels
    are filled with `fill_value` and flagged via the free mask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Observations:
    """Sparse measurements collected so far (the accumulating crowdsensed set)."""

    rows: np.ndarray   # (N,) int
    cols: np.ndarray   # (N,) int
    values: np.ndarray  # (N,) dBm

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def empty() -> "Observations":
        z = np.zeros(0)
        return Observations(z.astype(int), z.astype(int), z)

    def add(self, rows, cols, values) -> "Observations":
        return Observations(
            np.concatenate([self.rows, np.asarray(rows, int)]),
            np.concatenate([self.cols, np.asarray(cols, int)]),
            np.concatenate([self.values, np.asarray(values, float)]),
        )


@dataclass
class Reconstruction:
    """Dense estimate + uncertainty over the grid."""

    mean: np.ndarray   # (H,W) dBm
    std: np.ndarray    # (H,W) dBm, >=0 ; the acquisition surface
    free_mask: np.ndarray  # (H,W) bool, True = free space (where estimates are valid)

    def masked_rmse(self, gt: np.ndarray, observed_mask: np.ndarray | None = None) -> dict:
        """Blind-spot + free-space RMSE vs a dense ground truth (dBm)."""
        free = self.free_mask
        unobs = free & (~observed_mask if observed_mask is not None else np.ones_like(free))
        err = self.mean - gt

        def _rmse(m):
            return float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")

        return {
            "rmse_free": _rmse(free),
            "rmse_free_unobs": _rmse(unobs),
            "n_free": int(free.sum()),
            "n_free_unobs": int(unobs.sum()),
        }


class Reconstructor(ABC):
    """Fit sparse observations, predict a dense mean + uncertainty over free space."""

    def __init__(self, fill_value: float = -186.0):
        self.fill_value = fill_value

    @abstractmethod
    def fit(self, obs: Observations) -> "Reconstructor":
        ...

    @abstractmethod
    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        """Return a Reconstruction over the grid implied by ``free_mask`` (H,W)."""
        ...

    def reconstruct(self, obs: Observations, free_mask: np.ndarray) -> Reconstruction:
        return self.fit(obs).predict(free_mask)

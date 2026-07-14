"""
Inverse-distance-weighting reconstructor — classical baseline.

Mean is IDW over observations. IDW has no principled posterior variance, so we
expose distance-to-nearest-observation as a pseudo-uncertainty (monotonic proxy)
purely so the active loop has a comparable acquisition surface for the baseline.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .base import Observations, Reconstruction, Reconstructor


class IDWReconstructor(Reconstructor):
    def __init__(self, power: float = 2.0, k: int = 12, fill_value: float = -186.0):
        super().__init__(fill_value)
        self.power = power
        self.k = k
        self._fitted = False

    def fit(self, obs: Observations) -> "IDWReconstructor":
        if len(obs) == 0:
            self._fitted = False
            return self
        self._pts = np.stack([obs.rows, obs.cols], axis=1).astype(float)
        self._vals = obs.values.astype(float)
        self._tree = cKDTree(self._pts)
        self._fitted = True
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        mean = np.full((H, W), self.fill_value, dtype=np.float32)
        std = np.zeros((H, W), dtype=np.float32)
        if not self._fitted:
            std[free_mask] = 1.0
            return Reconstruction(mean, std, free_mask)
        qr, qc = np.where(free_mask)
        Q = np.stack([qr, qc], axis=1).astype(float)
        k = min(self.k, len(self._vals))
        dist, idx = self._tree.query(Q, k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = 1.0 / np.clip(dist, 1e-6, None) ** self.power
        m = np.sum(w * self._vals[idx], axis=1) / np.sum(w, axis=1)
        mean[qr, qc] = m.astype(np.float32)
        std[qr, qc] = dist[:, 0].astype(np.float32)  # distance-to-nearest as pseudo-uncertainty
        np.clip(mean, -186.0, -20.0, out=mean)
        return Reconstruction(mean, std, free_mask)

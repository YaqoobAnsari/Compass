"""
Gaussian-Process (Kriging) reconstructor — training-free, principled uncertainty.

Exact GP regression with an RBF kernel over pixel coordinates. The posterior
mean is the radio-map estimate; the posterior STANDARD DEVIATION is the
acquisition surface for active sensing (high where we have not measured — the
classic optimal-experimental-design signal, Krause/Guestrin JMLR 2008).

Tractable because the trajectory observation count is small (~hundreds): the
N×N kernel solve is trivial; dense prediction over ~40k free pixels is done in
chunks. "Building-aware" here means we only fit/evaluate over free space and can
optionally inflate distance across walls (geodesic option left as a hook).
"""

from __future__ import annotations

import numpy as np

from .base import Observations, Reconstruction, Reconstructor


def _rbf(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    """RBF kernel between (Na,2) and (Nb,2) pixel coords."""
    d2 = (
        (a[:, 0:1] - b[None, :, 0]) ** 2
        + (a[:, 1:2] - b[None, :, 1]) ** 2
    )
    return np.exp(-0.5 * d2 / (length_scale ** 2))


class GPReconstructor(Reconstructor):
    def __init__(
        self,
        length_scale: float = 25.0,
        signal_std: float | None = None,   # None -> estimate from data
        noise_std: float = 2.0,
        fill_value: float = -186.0,
        chunk: int = 4096,
    ):
        super().__init__(fill_value)
        self.length_scale = length_scale
        self.signal_std = signal_std
        self.noise_std = noise_std
        self.chunk = chunk
        self._fitted = False

    def fit(self, obs: Observations) -> "GPReconstructor":
        if len(obs) == 0:
            self._fitted = False
            return self
        self._X = np.stack([obs.rows.astype(float), obs.cols.astype(float)], axis=1)
        self._mean = float(np.mean(obs.values))
        y = obs.values - self._mean
        sig2 = (float(np.std(obs.values)) ** 2 if self.signal_std is None else self.signal_std ** 2)
        self._sig2 = max(sig2, 1e-3)
        K = self._sig2 * _rbf(self._X, self._X, self.length_scale)
        K[np.diag_indices_from(K)] += self.noise_std ** 2
        # Cholesky solve for stability
        self._L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._fitted = True
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        mean = np.full((H, W), self.fill_value, dtype=np.float32)
        std = np.zeros((H, W), dtype=np.float32)
        if not self._fitted:
            # no data yet: flat prior, max uncertainty everywhere free
            std[free_mask] = float(np.sqrt(self._sig2)) if hasattr(self, "_sig2") else 1.0
            return Reconstruction(mean, std, free_mask)

        qr, qc = np.where(free_mask)
        Q = np.stack([qr.astype(float), qc.astype(float)], axis=1)
        prior_std = float(np.sqrt(self._sig2))
        m_out = np.empty(len(Q), np.float32)
        s_out = np.empty(len(Q), np.float32)
        for i in range(0, len(Q), self.chunk):
            q = Q[i : i + self.chunk]
            Ks = self._sig2 * _rbf(q, self._X, self.length_scale)   # (c, N)
            m_out[i : i + self.chunk] = Ks @ self._alpha + self._mean
            v = np.linalg.solve(self._L, Ks.T)                       # (N, c)
            var = self._sig2 - np.sum(v ** 2, axis=0)
            s_out[i : i + self.chunk] = np.sqrt(np.clip(var, 0.0, None))
        mean[qr, qc] = m_out
        std[qr, qc] = s_out
        # clamp predictions to a sane dBm window
        np.clip(mean, -186.0, -20.0, out=mean)
        _ = prior_std
        return Reconstruction(mean, std, free_mask)

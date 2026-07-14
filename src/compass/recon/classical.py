"""
Additional classical reconstructors so the baseline panel is COMPREHENSIVE and
fairly tuned — not a strawman. Together with GP (gp.py) and IDW (idw.py) this
covers the standard spatial-interpolation families used in radio-map / spectrum-
cartography papers: IDW, RBF (multiple kernels), nearest-neighbour, natural-
neighbour (linear ND), and Kriging/GP.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)
from scipy.spatial import cKDTree

from .base import Observations, Reconstruction, Reconstructor


class NearestNeighborReconstructor(Reconstructor):
    def fit(self, obs: Observations) -> "NearestNeighborReconstructor":
        self._ok = len(obs) > 0
        if self._ok:
            self._interp = NearestNDInterpolator(
                np.stack([obs.rows, obs.cols], 1).astype(float), obs.values.astype(float))
            self._tree = cKDTree(np.stack([obs.rows, obs.cols], 1).astype(float))
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        mean = np.full((H, W), self.fill_value, np.float32)
        std = np.zeros((H, W), np.float32)
        if not self._ok:
            std[free_mask] = 1.0
            return Reconstruction(mean, std, free_mask)
        qr, qc = np.where(free_mask)
        Q = np.stack([qr, qc], 1).astype(float)
        mean[qr, qc] = self._interp(Q).astype(np.float32)
        std[qr, qc] = self._tree.query(Q, k=1)[0].astype(np.float32)
        np.clip(mean, -186.0, -20.0, out=mean)
        return Reconstruction(mean, std, free_mask)


class NaturalNeighborReconstructor(Reconstructor):
    """Linear ND interpolation (Delaunay) with nearest-neighbour fallback outside hull."""

    def fit(self, obs: Observations) -> "NaturalNeighborReconstructor":
        self._ok = len(obs) >= 4
        if self._ok:
            P = np.stack([obs.rows, obs.cols], 1).astype(float)
            self._lin = LinearNDInterpolator(P, obs.values.astype(float))
            self._near = NearestNDInterpolator(P, obs.values.astype(float))
            self._tree = cKDTree(P)
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        mean = np.full((H, W), self.fill_value, np.float32)
        std = np.zeros((H, W), np.float32)
        if not self._ok:
            std[free_mask] = 1.0
            return Reconstruction(mean, std, free_mask)
        qr, qc = np.where(free_mask)
        Q = np.stack([qr, qc], 1).astype(float)
        v = self._lin(Q)
        nan = ~np.isfinite(v)
        if nan.any():
            v[nan] = self._near(Q[nan])
        mean[qr, qc] = v.astype(np.float32)
        std[qr, qc] = self._tree.query(Q, k=1)[0].astype(np.float32)
        np.clip(mean, -186.0, -20.0, out=mean)
        return Reconstruction(mean, std, free_mask)


class RBFReconstructor(Reconstructor):
    # scale-invariant kernels do not take epsilon; others (multiquadric, gaussian) do.
    _NEEDS_EPSILON = {"multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"}

    def __init__(self, kernel: str = "multiquadric", smoothing: float = 1.0,
                 epsilon: float = 20.0, max_obs: int = 800, fill_value: float = -186.0):
        super().__init__(fill_value)
        self.kernel = kernel
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.max_obs = max_obs

    def fit(self, obs: Observations) -> "RBFReconstructor":
        # dedupe duplicate pixels (trajectory points repeat) -> avoids singular RBF system
        coords = np.stack([obs.rows, obs.cols], 1).astype(np.int64)
        uniq, inv = np.unique(coords, axis=0, return_inverse=True)
        v = np.zeros(len(uniq))
        np.add.at(v, inv, obs.values.astype(float))
        v /= np.bincount(inv, minlength=len(uniq))
        P = uniq.astype(float)
        self._ok = len(P) >= 3
        if self._ok:
            if len(P) > self.max_obs:  # RBF is O(N^3); subsample for tractability
                idx = np.random.default_rng(0).choice(len(P), self.max_obs, replace=False)
                P, v = P[idx], v[idx]
            kw = {"kernel": self.kernel, "smoothing": self.smoothing}
            if self.kernel in self._NEEDS_EPSILON:
                kw["epsilon"] = self.epsilon
            try:
                self._rbf = RBFInterpolator(P, v, **kw)
            except np.linalg.LinAlgError:  # fall back to a tiny ridge for stability
                kw["smoothing"] = max(self.smoothing, 1e-3)
                self._rbf = RBFInterpolator(P, v, **kw)
            self._tree = cKDTree(P)
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        mean = np.full((H, W), self.fill_value, np.float32)
        std = np.zeros((H, W), np.float32)
        if not self._ok:
            std[free_mask] = 1.0
            return Reconstruction(mean, std, free_mask)
        qr, qc = np.where(free_mask)
        Q = np.stack([qr, qc], 1).astype(float)
        mean[qr, qc] = self._rbf(Q).astype(np.float32)
        std[qr, qc] = self._tree.query(Q, k=1)[0].astype(np.float32)
        np.clip(mean, -186.0, -20.0, out=mean)
        return Reconstruction(mean, std, free_mask)


class OrdinaryKrigingReconstructor(Reconstructor):
    """Ordinary Kriging with a spherical variogram and the unbiasedness (sum-to-1)
    constraint via a Lagrange multiplier — the geostatistics-standard estimator,
    distinct from the GP/Simple-Kriging variant in gp.py (which fixes the mean).
    Gives the principled kriging variance as uncertainty.
    """

    def __init__(self, vrange: float = 40.0, nugget: float = 4.0,
                 max_obs: int = 400, chunk: int = 4096, fill_value: float = -186.0):
        super().__init__(fill_value)
        self.vrange = vrange
        self.nugget = nugget
        self.max_obs = max_obs
        self.chunk = chunk

    def _cov(self, h):
        # spherical covariance C(h) = sill*(1 - (1.5 h/a - 0.5 (h/a)^3)) for h<a, else 0
        a = self.vrange
        c = np.where(h < a, self._sill * (1 - (1.5 * h / a - 0.5 * (h / a) ** 3)), 0.0)
        return c

    def fit(self, obs: Observations) -> "OrdinaryKrigingReconstructor":
        self._ok = len(obs) >= 3
        if not self._ok:
            return self
        P = np.stack([obs.rows, obs.cols], 1).astype(float)
        v = obs.values.astype(float)
        if len(P) > self.max_obs:
            idx = np.random.default_rng(0).choice(len(P), self.max_obs, replace=False)
            P, v = P[idx], v[idx]
        self._P, self._v = P, v
        self._sill = max(float(np.var(v)), 1e-3)
        n = len(P)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        C = self._cov(D)
        C[np.diag_indices_from(C)] += self.nugget
        A = np.ones((n + 1, n + 1))
        A[:n, :n] = C
        A[n, n] = 0.0
        self._Ainv = np.linalg.pinv(A)
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        mean = np.full((H, W), self.fill_value, np.float32)
        std = np.zeros((H, W), np.float32)
        if not self._ok:
            std[free_mask] = 1.0
            return Reconstruction(mean, std, free_mask)
        qr, qc = np.where(free_mask)
        Q = np.stack([qr, qc], 1).astype(float)
        n = len(self._P)
        m_out = np.empty(len(Q), np.float32)
        s_out = np.empty(len(Q), np.float32)
        for i in range(0, len(Q), self.chunk):
            q = Q[i:i + self.chunk]
            d = np.sqrt(((q[:, None, :] - self._P[None, :, :]) ** 2).sum(-1))  # (c,n)
            b = np.empty((len(q), n + 1))
            b[:, :n] = self._cov(d)
            b[:, n] = 1.0
            w = b @ self._Ainv.T                       # (c, n+1)
            m_out[i:i + self.chunk] = w[:, :n] @ self._v
            var = self._sill - np.sum(w * b, axis=1)
            s_out[i:i + self.chunk] = np.sqrt(np.clip(var, 0.0, None))
        mean[qr, qc] = m_out
        std[qr, qc] = s_out
        np.clip(mean, -186.0, -20.0, out=mean)
        return Reconstruction(mean, std, free_mask)


class GeodesicNearestReconstructor(Reconstructor):
    """Building-AWARE nearest-neighbour: distance to observations is measured as a
    geodesic through free space (around buildings), not Euclidean through walls.
    Multi-source Dijkstra over the free-space grid propagates the nearest seed's
    value; geodesic distance is the uncertainty. This is the fair building-aware
    classical baseline (no learning, but respects geometry)."""

    def __init__(self, fill_value: float = -186.0):
        super().__init__(fill_value)

    def fit(self, obs: Observations) -> "GeodesicNearestReconstructor":
        self._obs = obs
        return self

    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        import heapq
        H, W = free_mask.shape
        INF = np.inf
        dist = np.full((H, W), INF, np.float32)
        val = np.full((H, W), self.fill_value, np.float32)
        mean = np.full((H, W), self.fill_value, np.float32)
        std = np.zeros((H, W), np.float32)
        if len(self._obs) == 0:
            std[free_mask] = 1.0
            return Reconstruction(mean, std, free_mask)
        heap = []
        for r, c, v in zip(self._obs.rows, self._obs.cols, self._obs.values):
            r, c = int(r), int(c)
            if 0 <= r < H and 0 <= c < W and dist[r, c] > 0:
                dist[r, c] = 0.0
                val[r, c] = v
                heapq.heappush(heap, (0.0, r, c, float(v)))
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356)]
        while heap:
            d, r, c, v = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w in nbrs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and free_mask[nr, nc]:
                    nd = d + w
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        val[nr, nc] = v
                        heapq.heappush(heap, (nd, nr, nc, v))
        m = free_mask & np.isfinite(dist)
        mean[m] = val[m]
        std[m] = np.where(np.isfinite(dist[m]), dist[m], 0.0).astype(np.float32)
        np.clip(mean, -186.0, -20.0, out=mean)
        return Reconstruction(mean, std, free_mask)

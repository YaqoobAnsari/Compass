"""
Adapter so a trained CompassNet drives the active-sensing loop through the
standard Reconstructor interface. Unlike the GP (geometric uncertainty), the
learned model's MC-dropout uncertainty is PHYSICS-AWARE (depends on building +
TX), so max-variance acquisition can target informative-but-unvisited regions
(e.g. behind buildings near the TX) rather than merely unvisited ones — the
hypothesis that learned uncertainty makes max-variance beat space-filling.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from ..data.conditioning import dbm_to_signed_unit, trajectory_features
from ..data.conventions import signed_unit_to_dbm
from ..data.trajectory import Trajectory
from .base import Observations, Reconstruction, Reconstructor
from .learned import _enable_dropout


class LearnedActiveReconstructor(Reconstructor):
    def __init__(self, model, building01, free, tx_rowcol, device="cpu", n_mc=8,
                 coverage_sigma=5.0, seq_len=256, fill_value=-186.0):
        super().__init__(fill_value)
        self.model = model.to(device).eval()
        self.device = device
        self.n_mc = n_mc
        self.building01 = building01.astype(np.float32)   # 1=building
        self.free = free
        self.tx_rowcol = torch.tensor(tx_rowcol, dtype=torch.long)
        self.coverage_sigma = coverage_sigma
        self.seq_len = seq_len
        self._batch = None

    def fit(self, obs: Observations) -> "LearnedActiveReconstructor":
        H, W = self.building01.shape
        sparse = np.zeros((H, W), np.float32)
        mask = np.zeros((H, W), np.float32)
        if len(obs):
            sparse[obs.rows, obs.cols] = dbm_to_signed_unit(obs.values)
            mask[obs.rows, obs.cols] = 1.0
            tr = Trajectory(obs.rows.astype(float), obs.cols.astype(float),
                            np.arange(len(obs), dtype=float))
            seq = trajectory_features(tr, (int(self.tx_rowcol[0]), int(self.tx_rowcol[1])), obs.values)
        else:
            seq = np.zeros((1, 11), np.float32)
        cov = gaussian_filter(mask, sigma=self.coverage_sigma)
        if cov.max() > 0:
            cov = cov / cov.max()
        if len(seq) >= self.seq_len:
            seq = seq[: self.seq_len]
        else:
            seq = np.concatenate([seq, np.zeros((self.seq_len - len(seq), seq.shape[1]), np.float32)])
        t = torch.from_numpy
        self._batch = {
            "sparse_rss": t(sparse[None, None]).float().to(self.device),
            "mask": t(mask[None, None]).float().to(self.device),
            "coverage": t(cov[None, None]).float().to(self.device),
            "building": t(self.building01[None, None]).float().to(self.device),
            "free_mask": t(self.free[None, None].astype(np.float32)).float().to(self.device),
            "tx_rowcol": self.tx_rowcol[None].to(self.device),
            "sequence": t(seq[None].astype(np.float32)).float().to(self.device),
        }
        return self

    @torch.no_grad()
    def predict(self, free_mask: np.ndarray) -> Reconstruction:
        H, W = free_mask.shape
        if self._batch is None:
            std = np.zeros((H, W), np.float32)
            std[free_mask] = 1.0
            return Reconstruction(np.full((H, W), self.fill_value, np.float32), std, free_mask)
        _enable_dropout(self.model)
        preds = np.stack([self.model(self._batch).cpu().numpy()[0, 0] for _ in range(self.n_mc)], 0)
        mean = signed_unit_to_dbm(preds.mean(0)).astype(np.float32)
        std = (preds.std(0) * 69.5).astype(np.float32)
        mean[~free_mask] = self.fill_value
        std[~free_mask] = 0.0
        return Reconstruction(mean, std, free_mask)

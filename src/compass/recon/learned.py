"""
Wrap a trained CompassNet as a reconstructor with Monte-Carlo-dropout uncertainty,
so learned variants go through the SAME evaluation (and active loop) as classical
baselines. Uncertainty = std over n_mc stochastic forward passes (dropout on) —
crucially this uncertainty is PHYSICS-AWARE (it depends on building + TX), unlike
the purely geometric GP variance.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.conventions import signed_unit_to_dbm
from ..models.compass_net import CompassConfig, CompassNet


def load_compass(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CompassConfig(**ck["cfg"])
    if ck.get("arch") == "compass_wnet":
        from ..models.compass_wnet import CompassWNet
        model = CompassWNet(cfg)
    else:
        model = CompassNet(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def _enable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


class LearnedReconstructor:
    def __init__(self, model: CompassNet, device: str = "cpu", n_mc: int = 8):
        self.model = model
        self.device = device
        self.n_mc = n_mc

    @torch.no_grad()
    def predict_batch(self, batch: dict):
        """Return (mean_dbm, std_dbm) maps, each (B,H,W) numpy, via MC-dropout."""
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.model.eval()
        _enable_dropout(self.model)            # dropout ON for MC sampling
        preds = []
        for _ in range(self.n_mc):
            preds.append(self.model(b).cpu().numpy()[:, 0])   # (B,H,W) normalised
        preds = np.stack(preds, axis=0)                       # (mc,B,H,W)
        mean_norm = preds.mean(0)
        std_norm = preds.std(0)
        mean_dbm = signed_unit_to_dbm(mean_norm)
        # convert normalised std -> dB scale (1 unit == 69.5 dB)
        std_dbm = std_norm * 69.5
        return mean_dbm, std_dbm

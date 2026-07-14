"""
Wrap a trained DL baseline as a benchmark reconstructor with the same
`predict_batch(batch) -> (mean_dbm, std_dbm)` interface as `LearnedReconstructor`,
so baselines run through the IDENTICAL evaluation as COMPASS and classical methods.

Uncertainty: baselines with dropout get MC-dropout std; those without emit a constant
small std (they were not designed for calibrated UQ — reported honestly, and their
calibration columns reflect that). Point-accuracy (the headline RMSE) is unaffected.
"""

from __future__ import annotations

import numpy as np
import torch

from ...data.conventions import signed_unit_to_dbm
from . import BASELINES


def load_baseline(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BASELINES[ck["arch"]](**ck.get("kwargs", {}))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def _has_dropout(model):
    return any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)) for m in model.modules())


class BaselineReconstructor:
    def __init__(self, model, device: str = "cpu", n_mc: int = 8):
        self.model = model
        self.device = device
        # ensemble if the model has dropout OR is a stochastic sampler (e.g. RMDM diffusion)
        self.n_mc = n_mc if (_has_dropout(model) or getattr(model, "stochastic", False)) else 1

    @torch.no_grad()
    def predict_batch(self, batch: dict):
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.model.eval()
        if self.n_mc > 1:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                    m.train()
        preds = [self.model(b).cpu().numpy()[:, 0] for _ in range(self.n_mc)]
        preds = np.stack(preds, axis=0)
        mean_dbm = signed_unit_to_dbm(preds.mean(0))
        std_dbm = preds.std(0) * 69.5 if self.n_mc > 1 else np.full_like(mean_dbm, 1e-3)
        return mean_dbm, std_dbm

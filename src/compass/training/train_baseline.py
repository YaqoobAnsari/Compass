"""
Train an external DL radio-map baseline (RadioUNet / PMNet / SparseUNet / RMDM) on
the SAME data pipeline, trajectory sampling, noise, split, optimiser, and budget as
COMPASS — so any difference is architecture, not protocol. This is the fairness
guarantee for the DL-baseline comparison (Phase G).

Loss = masked reconstruction (the objective these methods are trained on). Models
that expose a coarse output (`model.aux`, e.g. RadioUNet's WNet stage-1) get deep
supervision (0.5 * recon on the coarse map).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.baselines import BASELINES
from .data import CompassTrainDataset
from .losses import masked_reconstruction_loss


@dataclass
class BaselineTrainConfig:
    arch: str                      # key in BASELINES
    root: str
    out_dir: str
    train_maps: List[int]
    val_maps: List[int]
    variant: str = "IRT2"
    tx_per_map: int = 20
    correlated_noise: bool = True
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-4
    num_workers: int = 8
    seed: int = 0
    kwargs: Dict = field(default_factory=dict)  # model constructor kwargs


def _val_rmse(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(b)
            free = b["free_mask"] > 0.5
            unobs = free & (b["mask"] < 0.5)
            err = (pred - b["target"]) * 69.5
            for i in range(pred.size(0)):
                m = unobs[i, 0]
                if m.any():
                    out.append(float(torch.sqrt((err[i, 0][m] ** 2).mean())))
    return float(np.mean(out)) if out else float("nan")


def train_baseline(cfg: BaselineTrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_ds = CompassTrainDataset(cfg.root, cfg.train_maps, cfg.variant, cfg.tx_per_map,
                                correlated_noise=cfg.correlated_noise, seed=cfg.seed)
    va_ds = CompassTrainDataset(cfg.root, cfg.val_maps, cfg.variant, cfg.tx_per_map,
                                correlated_noise=cfg.correlated_noise, seed=cfg.seed + 1)
    tr = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, persistent_workers=cfg.num_workers > 0)
    va = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0)

    model = BASELINES[cfg.arch](**cfg.kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = {"config": asdict(cfg), "arch": cfg.arch, "n_params": n_params, "epochs": []}
    best = float("inf")
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(b)
            loss = masked_reconstruction_loss(pred, b["target"], b["free_mask"], b["mask"])
            if getattr(model, "aux", None) is not None:  # deep supervision (e.g. WNet coarse)
                loss = loss + 0.5 * masked_reconstruction_loss(model.aux, b["target"], b["free_mask"], b["mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        val_rmse = _val_rmse(model, va, device)
        rec = {"epoch": ep, "train_loss": ep_loss / max(nb, 1), "val_rmse_free_unobs_db": val_rmse,
               "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        print(f"  ep {ep:3d} train {rec['train_loss']:.4f} val_rmse {val_rmse:.2f} dB ({rec['elapsed_s']:.0f}s)")
        if val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(), "arch": cfg.arch, "kwargs": cfg.kwargs,
                        "epoch": ep, "val_rmse": val_rmse}, out / "best.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    history["best_val_rmse_free_unobs_db"] = best
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train-baseline:{cfg.arch}] done. best val free-unobs RMSE = {best:.2f} dB | {n_params/1e6:.2f}M params")
    return history

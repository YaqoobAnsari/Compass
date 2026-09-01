"""
Dedicated trainer for the RMDM diffusion baseline (its objective is eps-prediction +
coarse reconstruction, not a single masked-MSE, so it can't use the shared loop).
Same data/split/budget as every other method. Full-sampling val RMSE (the real metric)
is costly, so it is computed every ``val_sample_every`` epochs for checkpoint
selection; a cheap diffusion-loss is logged each epoch.
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


@dataclass
class RMDMTrainConfig:
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
    val_sample_every: int = 8
    arch: str = "rmdm"          # key in BASELINES (rmdm | radiodiff)
    kwargs: Dict = field(default_factory=dict)


@torch.no_grad()
def _val_rmse_sampled(model, loader, device, max_batches=6):
    model.eval()
    out = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model(b)  # DDIM sampling
        free = b["free_mask"] > 0.5
        unobs = free & (b["mask"] < 0.5)
        err = (pred - b["target"]) * 69.5
        for i in range(pred.size(0)):
            m = unobs[i, 0]
            if m.any():
                out.append(float(torch.sqrt((err[i, 0][m] ** 2).mean())))
    return float(np.mean(out)) if out else float("nan")


def train_rmdm(cfg: RMDMTrainConfig) -> dict:
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
            loss, comp = model.training_losses(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        do_sample = (ep % cfg.val_sample_every == 0) or (ep == cfg.epochs - 1)
        val_rmse = _val_rmse_sampled(model, va, device) if do_sample else None
        rec = {"epoch": ep, "train_loss": ep_loss / max(nb, 1),
               "val_rmse_free_unobs_db": val_rmse, "lr": opt.param_groups[0]["lr"],
               "elapsed_s": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        vstr = f"{val_rmse:.2f}" if val_rmse is not None else "  -- "
        print(f"  ep {ep:3d} train {rec['train_loss']:.4f} val_rmse {vstr} dB ({rec['elapsed_s']:.0f}s)")
        torch.save({"model": model.state_dict(), "arch": cfg.arch, "kwargs": cfg.kwargs, "epoch": ep},
                   out / "last.ckpt")
        if val_rmse is not None and val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(), "arch": cfg.arch, "kwargs": cfg.kwargs,
                        "epoch": ep, "val_rmse": val_rmse}, out / "best.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    history["best_val_rmse_free_unobs_db"] = best
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train-{cfg.arch}] done. best val free-unobs RMSE = {best:.2f} dB | {n_params/1e6:.2f}M params")
    return history

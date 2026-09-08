"""
Config-driven training loop for the COMPASS reconstructor (and every ablation).

Plain PyTorch (no Lightning dependency) for transparency. Logs per-epoch train/val
metrics to a JSON history, checkpoints the best val model, and is fully driven by a
TrainConfig so each ablation differs only by its flags.
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

from ..models.compass_net import CompassConfig, CompassNet
from ..models.compass_wnet import CompassWNet
from .data import CompassTrainDataset
from .losses import compass_loss, masked_reconstruction_loss


@dataclass
class TrainConfig:
    root: str
    out_dir: str
    train_maps: List[int]
    val_maps: List[int]
    variant: str = "IRT2"
    tx_per_map: int = 20
    arch: str = "compass"  # "compass" | "compass_wnet"
    # model / ablation flags
    use_building: bool = True
    use_tx: bool = True
    use_order: bool = True
    use_device: bool = True
    use_occlusion: bool = False
    occlusion_anchor: str = "tx"  # "tx" | "measurement" (transmitter-free)
    use_rcl: bool = True
    order_shuffle: bool = False
    correlated_noise: bool = True
    tx_jitter_px: float = 0.0
    # optim
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-4
    base: int = 48
    depth: int = 4
    num_workers: int = 8
    seed: int = 0
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"recon": 1.0, "traj": 0.5, "rcl": 0.1})


def _val_metrics(model, loader, weights, use_rcl, device):
    model.eval()
    tot, n = 0.0, 0
    rmse_free_unobs = []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(b)
            loss, _ = compass_loss(pred, b, weights, use_rcl)
            tot += loss.item() * pred.size(0)
            n += pred.size(0)
            free = b["free_mask"] > 0.5
            unobs = free & (b["mask"] < 0.5)
            err = (pred - b["target"]) * 69.5  # normalised -> dB
            for i in range(pred.size(0)):
                m = unobs[i, 0]
                if m.any():
                    rmse_free_unobs.append(float(torch.sqrt((err[i, 0][m] ** 2).mean())))
    return tot / max(n, 1), float(np.mean(rmse_free_unobs)) if rmse_free_unobs else float("nan")


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_ds = CompassTrainDataset(cfg.root, cfg.train_maps, cfg.variant, cfg.tx_per_map,
                                order_shuffle=cfg.order_shuffle, correlated_noise=cfg.correlated_noise,
                                tx_jitter_px=cfg.tx_jitter_px, seed=cfg.seed)
    va_ds = CompassTrainDataset(cfg.root, cfg.val_maps, cfg.variant, cfg.tx_per_map,
                                order_shuffle=cfg.order_shuffle, correlated_noise=cfg.correlated_noise,
                                tx_jitter_px=cfg.tx_jitter_px, seed=cfg.seed + 1)
    tr = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, persistent_workers=cfg.num_workers > 0)
    va = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0)

    mcfg = CompassConfig(use_building=cfg.use_building, use_tx=cfg.use_tx,
                         use_order=cfg.use_order, use_device=cfg.use_device,
                         use_occlusion=cfg.use_occlusion, base=cfg.base, depth=cfg.depth,
                         occlusion_anchor=cfg.occlusion_anchor)
    model = (CompassWNet(mcfg) if cfg.arch == "compass_wnet" else CompassNet(mcfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = {"config": asdict(cfg), "n_params": n_params, "epochs": []}
    best = float("inf")
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(b)
            loss, comp = compass_loss(pred, b, cfg.loss_weights, cfg.use_rcl)
            if getattr(model, "aux", None) is not None:  # WNet coarse-stage deep supervision
                loss = loss + 0.5 * masked_reconstruction_loss(
                    model.aux, b["target"], b["free_mask"], b["mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        val_loss, val_rmse = _val_metrics(model, va, cfg.loss_weights, cfg.use_rcl, device)
        rec = {"epoch": ep, "train_loss": ep_loss / max(nb, 1),
               "val_loss": val_loss, "val_rmse_free_unobs_db": val_rmse,
               "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        print(f"  ep {ep:3d} train {rec['train_loss']:.4f} val {val_loss:.4f} "
              f"val_rmse {val_rmse:.2f} dB ({rec['elapsed_s']:.0f}s)")
        if val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(), "cfg": asdict(mcfg), "arch": cfg.arch,
                        "epoch": ep, "val_rmse": val_rmse}, out / "best.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    history["best_val_rmse_free_unobs_db"] = best
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done. best val free-unobs RMSE = {best:.2f} dB | params {n_params/1e6:.2f}M")
    return history

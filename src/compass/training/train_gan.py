"""
Adversarial trainer for the RadioGAN baseline (LSGAN + L1), same data/split/budget as
every other method. Generator loss = LSGAN(D(fake)=1) + λ·masked-recon; discriminator
loss = LSGAN(real=1, fake=0). Best-checkpoint by generator val free-unobs RMSE.
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

from ..models.baselines.radiogan import RadioGAN
from .data import CompassTrainDataset
from .losses import masked_reconstruction_loss


@dataclass
class GANTrainConfig:
    root: str
    out_dir: str
    train_maps: List[int]
    val_maps: List[int]
    variant: str = "IRT2"
    tx_per_map: int = 20
    correlated_noise: bool = True
    epochs: int = 150
    batch_size: int = 16
    lr: float = 2e-4
    l1_weight: float = 50.0
    num_workers: int = 8
    seed: int = 0
    kwargs: Dict = field(default_factory=dict)


@torch.no_grad()
def _val_rmse(model, loader, device):
    model.eval()
    out = []
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


def train_gan(cfg: GANTrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    tr_ds = CompassTrainDataset(cfg.root, cfg.train_maps, cfg.variant, cfg.tx_per_map,
                                correlated_noise=cfg.correlated_noise, seed=cfg.seed)
    va_ds = CompassTrainDataset(cfg.root, cfg.val_maps, cfg.variant, cfg.tx_per_map,
                                correlated_noise=cfg.correlated_noise, seed=cfg.seed + 1)
    tr = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, persistent_workers=cfg.num_workers > 0)
    va = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0)

    model = RadioGAN(**cfg.kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt_g = torch.optim.AdamW(model.gen.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(model.disc.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, cfg.epochs)

    history = {"config": asdict(cfg), "arch": "radiogan", "n_params": n_params, "epochs": []}
    best = float("inf"); t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        gl, dl, nb = 0.0, 0.0, 0
        for batch in tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            cond = model.cond(b); real = b["target"]
            fake = model.gen(cond)
            # discriminator
            d_real = model.disc(cond, real)
            d_fake = model.disc(cond, fake.detach())
            loss_d = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            # generator
            d_fake2 = model.disc(cond, fake)
            loss_g = ((d_fake2 - 1) ** 2).mean() + cfg.l1_weight * masked_reconstruction_loss(
                fake, real, b["free_mask"], b["mask"])
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            gl += loss_g.item(); dl += loss_d.item(); nb += 1
        sch_g.step()
        val_rmse = _val_rmse(model, va, device)
        rec = {"epoch": ep, "g_loss": gl / max(nb, 1), "d_loss": dl / max(nb, 1),
               "val_rmse_free_unobs_db": val_rmse, "elapsed_s": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        print(f"  ep {ep:3d} G {rec['g_loss']:.3f} D {rec['d_loss']:.3f} val_rmse {val_rmse:.2f} ({rec['elapsed_s']:.0f}s)")
        if val_rmse < best:
            best = val_rmse
            torch.save({"model": model.state_dict(), "arch": "radiogan", "kwargs": cfg.kwargs,
                        "epoch": ep, "val_rmse": val_rmse}, out / "best.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    history["best_val_rmse_free_unobs_db"] = best
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train-gan] done. best val free-unobs RMSE = {best:.2f} dB | {n_params/1e6:.2f}M params")
    return history

#!/usr/bin/env python
"""
NLoS-gap test: does an explicit diffraction/occlusion conditioning channel close the
behind-building reconstruction wall that COMPASS-WNet and RadioUNet BOTH plateau at
(identical NLoS RMSE 10.89)?

Same backbone (ConditioningUNet, no order/device heads, to isolate the effect), trained
with vs without a per-pixel shadow-depth channel = fraction of the TX->pixel ray that
lies inside buildings. Reports free-unobserved RMSE split into LoS / NLoS.

  bash scripts/submit.sh 2g.35gb python scripts/exp_nlos_gap.py --occlusion --out .../occ
  bash scripts/submit.sh 2g.35gb python scripts/exp_nlos_gap.py            --out .../base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compass.data import list_map_ids, split_map_ids
from compass.eval.metrics import _occlusion_count
from compass.models.compass_net import tx_heatmap
from compass.models.unet import ConditioningUNet
from compass.training.data import CompassTrainDataset
from compass.training.losses import masked_reconstruction_loss

REPO = Path(__file__).resolve().parents[1]
DB = 69.5  # normalised [-1,1] -> dB (PATHLOSS_RANGE_DB / 2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def occlusion_field(building, tx_rc, n_steps=32):
    """(B,1,H,W) shadow-depth = fraction of the TX->pixel ray inside buildings."""
    B, _, H, W = building.shape
    ys = torch.arange(H, device=building.device).view(1, H, 1).expand(B, H, W).float()
    xs = torch.arange(W, device=building.device).view(1, 1, W).expand(B, H, W).float()
    tr = tx_rc[:, 0].view(B, 1, 1).float()
    tc = tx_rc[:, 1].view(B, 1, 1).float()
    acc = torch.zeros(B, H, W, device=building.device)
    for i in range(1, n_steps):
        t = i / n_steps
        gy = (tr + t * (ys - tr)) / max(H - 1, 1) * 2 - 1
        gx = (tc + t * (xs - tc)) / max(W - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        acc += F.grid_sample(building, grid, mode="nearest", align_corners=True)[:, 0]
    return (acc / n_steps).unsqueeze(1)


def build_input(b, use_occ):
    H, W = b["sparse_rss"].shape[-2:]
    x = [b["sparse_rss"], b["mask"], b["coverage"], b["building"],
         tx_heatmap(b["tx_rowcol"], H, W, 4)]
    if use_occ:
        x.append(occlusion_field(b["building"], b["tx_rowcol"]))
    return torch.cat(x, dim=1)


def evaluate(model, loader, use_occ):
    model.eval()
    se = {"unobs": 0.0, "los": 0.0, "nlos": 0.0}
    n = {"unobs": 0, "los": 0, "nlos": 0}
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(build_input(b, use_occ), None)
            err = ((pred - b["target"]) * DB).cpu().numpy()[:, 0]
            free = (b["free_mask"] > 0.5).cpu().numpy()[:, 0]
            obs = (b["mask"] > 0.5).cpu().numpy()[:, 0]
            bld = (b["building"] > 0.5).cpu().numpy()[:, 0].astype(float)
            txrc = b["tx_rowcol"].cpu().numpy()
            for i in range(err.shape[0]):
                unobs = free[i] & ~obs[i]
                occ = _occlusion_count(bld[i], (float(txrc[i][0]), float(txrc[i][1])))
                for key, m in (("unobs", unobs), ("los", unobs & (occ == 0)),
                               ("nlos", unobs & (occ >= 2))):
                    e = err[i][m]
                    se[key] += float((e ** 2).sum()); n[key] += int(m.sum())
    return {k: round(float(np.sqrt(se[k] / max(n[k], 1))), 3) for k in se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occlusion", action="store_true")
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--train-maps", type=int, default=120)
    ap.add_argument("--test-maps", type=int, default=30)
    ap.add_argument("--tx-per-map", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "results" / "nlos_gap.json"))
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"[nlos-gap] occlusion={args.occlusion} device={DEV}")

    sp = split_map_ids(list_map_ids(args.root))
    tr_ds = CompassTrainDataset(args.root, sp["train"][: args.train_maps], "IRT2", args.tx_per_map, seed=0)
    te_ds = CompassTrainDataset(args.root, sp["test"][: args.test_maps], "IRT2", args.tx_per_map, seed=7)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)

    cin = 8 + (1 if args.occlusion else 0)
    model = ConditioningUNet(cin, base=48, depth=4, film_dim=0, p_drop=0.15).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_metrics = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for batch in tr:
            b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(build_input(b, args.occlusion), None)
            loss = masked_reconstruction_loss(pred, b["target"], b["free_mask"], b["mask"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        sched.step()
        if ep >= 5 and (ep % 5 == 0 or ep == args.epochs - 1):
            m = evaluate(model, te, args.occlusion)
            print(f"  ep {ep:3d} loss {tot/nb:.4f} | unobs {m['unobs']} LoS {m['los']} NLoS {m['nlos']}")
            if m["unobs"] < best:
                best, best_metrics = m["unobs"], m
        else:
            print(f"  ep {ep:3d} loss {tot/nb:.4f}")

    out = {"occlusion": args.occlusion, "n_params": n_params, "epochs": args.epochs,
           "best": best_metrics, "final": evaluate(model, te, args.occlusion)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[nlos-gap] occlusion={args.occlusion}  best={best_metrics}  -> {args.out}")


if __name__ == "__main__":
    main()

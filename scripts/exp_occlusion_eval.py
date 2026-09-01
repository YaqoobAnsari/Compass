#!/usr/bin/env python
"""
Benchmark-protocol test eval of the occlusion confirmation runs.

Loads the trained WNet checkpoints (occlusion vs matched baseline), evaluates on the
held-out test split with the benchmark metrics (free-unobserved RMSE + LoS/NLoS split +
SSIM), and reports the head-to-head vs the published COMPASS-WNet (9.97) / RadioUNet (10.00).

  bash scripts/submit.sh 2g.35gb python scripts/exp_occlusion_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from compass.data import list_map_ids, split_map_ids
from compass.eval.metrics import accuracy_metrics, plausibility_metrics
from compass.recon.learned import load_compass
from compass.training.data import CompassTrainDataset

REPO = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_dbm(v):
    return (v + 1.0) * 69.5 - 186.0


def eval_ckpt(ckpt, loader):
    model = load_compass(ckpt, DEV)
    acc = {"rmse_free_unobs": [], "ssim_free": [], "rmse_free": []}
    plaus = {"rmse_nlos": [], "rmse_los": []}
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(b)
            pd = to_dbm(pred).cpu().numpy()[:, 0]
            gd = to_dbm(b["target"]).cpu().numpy()[:, 0]
            free = (b["free_mask"] > 0.5).cpu().numpy()[:, 0]
            obs = (b["mask"] > 0.5).cpu().numpy()[:, 0]
            bld = (b["building"] > 0.5).cpu().numpy()[:, 0]
            txrc = b["tx_rowcol"].cpu().numpy()
            for i in range(pd.shape[0]):
                a = accuracy_metrics(pd[i], gd[i], free[i], obs[i])
                p = plausibility_metrics(pd[i], gd[i], bld[i], free[i],
                                         (float(txrc[i][0]), float(txrc[i][1])))
                for k in acc:
                    if np.isfinite(a.get(k, np.nan)):
                        acc[k].append(a[k])
                for k in plaus:
                    if np.isfinite(p.get(k, np.nan)):
                        plaus[k].append(p[k])
    out = {k: round(float(np.mean(v)), 3) for k, v in {**acc, **plaus}.items() if v}
    return out


def main():
    root = str(REPO / "data" / "raw")
    sp = split_map_ids(list_map_ids(root))
    te = CompassTrainDataset(root, sp["test"][:40], "IRT2", 20, seed=123)
    loader = DataLoader(te, batch_size=16, shuffle=False, num_workers=6)
    print(f"[occ-eval] device={DEV}  test samples={len(te)}")

    res = {}
    for name, d in [("occlusion", "wnet_occ"), ("baseline", "wnet_base")]:
        ck = REPO / "experiments" / "reconstructors_long" / d / "best.ckpt"
        if not ck.exists():
            print(f"  [skip] {d}: no checkpoint"); continue
        res[name] = eval_ckpt(str(ck), loader)
        print(f"  {name:10s}: {res[name]}")

    (REPO / "results" / "occlusion_eval.json").write_text(json.dumps(res, indent=2))
    print("\n===== TEST-set head-to-head (free-unobs RMSE, dB) =====")
    if "occlusion" in res and "baseline" in res:
        o, base = res["occlusion"], res["baseline"]
        print(f"  baseline WNet (ours, re-run): {base['rmse_free_unobs']}  "
              f"| LoS {base.get('rmse_los')} NLoS {base.get('rmse_nlos')} SSIM {base.get('ssim_free')}")
        print(f"  + occlusion (ours)          : {o['rmse_free_unobs']}  "
              f"| LoS {o.get('rmse_los')} NLoS {o.get('rmse_nlos')} SSIM {o.get('ssim_free')}")
        print(f"  delta (occ - base)          : {round(o['rmse_free_unobs'] - base['rmse_free_unobs'], 3)} dB")
        print("  published: COMPASS-WNet 9.97 / RadioUNet 10.00 (test)")
    print("[occ-eval] -> results/occlusion_eval.json")


if __name__ == "__main__":
    main()

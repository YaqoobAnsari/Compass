#!/usr/bin/env python
"""
Sim-to-real fine-tuning — the decisive fix for W1 (learned recon loses to classical on
sparse real because it is data-starved). Pretrain the geometry-aware DeepSet interpolator
on sparse points SAMPLED from dense synthetic maps (same feature space, m/px-aware), then
fine-tune on real cells. Compare on real held-out RPs (cell-CV):
  scratch (real only) · zero-shot (pretrained only) · FINE-TUNED · classical RBF.

If fine-tuned beats classical, W1 flips from the paper's weakness to its punchline.

  bash scripts/submit.sh 2g.35gb python scripts/exp_sim2real_finetune.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids
from compass.realdata.learned_interp import SetInterpolator, build_features, make_samples
from compass.realdata.reconstruct import METHODS, M_PER_PX
from compass.realdata.walls import cell_rp_tables_px, load_wall_mask, wall_matrix_m

REPO = Path(__file__).resolve().parents[1]
SITES = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}


def synth_cells(n_maps=40, n_tx=4, m_pts=40, seed=0):
    """Sparse 'RP' cells sampled from dense RadioMapSeer maps (synthetic pretraining data)."""
    root = str(REPO / "data" / "raw")
    maps = split_map_ids(list_map_ids(root))["train"][:n_maps]
    base = RadioMapSeerDataset(root, map_ids=maps, variant="IRT2")
    rng = np.random.default_rng(seed)
    cells = []
    for map_id in maps:
        for tx in range(n_tx):
            try:
                s = base[base.samples.index((map_id, tx))]
            except ValueError:
                continue
            dbm = s["radio_map_dbm"].numpy()[0]
            bld = s["building_map"].numpy()[0]
            free = bld < 0.5
            fr = np.argwhere(free)
            if len(fr) < m_pts + 5:
                continue
            pts = fr[rng.choice(len(fr), m_pts, replace=False)]  # (M,2) row,col
            pos_px = pts[:, ::-1].astype(float)                   # (x=col, y=row)
            rss = dbm[pts[:, 0], pts[:, 1]].astype(float)
            W = wall_matrix_m(pos_px, bld > 0.5, m_per_px=1.0)   # synthetic 1 m/px
            cells.append((pos_px, rss, W))
    return cells


def real_cells():
    cells = []
    for site, floors in SITES.items():
        for fl in floors:
            bar = load_wall_mask(site, fl)
            if bar is None:
                continue
            for _c, (pos, rss) in cell_rp_tables_px(site, fl, min_rp=10).items():
                cells.append((pos, rss, wall_matrix_m(pos, bar)))
    return cells


def fit(model, cells, device, epochs, lr, m_per_px):
    data = make_samples(cells, use_walls=True, m_per_px=m_per_px)
    if data is None:
        return model
    F, Mk, Cm, Y = (t.to(device) for t in data)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Y), device=device)
        for b in range(0, len(Y), 256):
            i = perm[b:b + 256]
            loss = ((model(F[i], Mk[i], Cm[i]) - Y[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_real(model, cells, device):
    model.eval(); errs = []
    for pos, rss, W in cells:
        pm = pos * M_PER_PX
        for i in range(len(pos)):
            de = np.linalg.norm(pm - pm[i], axis=1)
            keep = np.where((de > 1.0) & (np.arange(len(pos)) != i))[0]
            if len(keep) < 4:
                continue
            f, m, c = build_features(pos, rss, W, keep, i, True, M_PER_PX)
            p = model(torch.tensor(f[None]).to(device), torch.tensor(m[None]).to(device),
                      torch.tensor([c], dtype=torch.float32).to(device))
            errs.append(abs(float(p) - rss[i]))
    return errs


def classical_real(cells, key="RBF(multiquadric)"):
    errs = []
    for pos, rss, _ in cells:
        pm = pos * M_PER_PX
        for i in range(len(pos)):
            de = np.linalg.norm(pm - pm[i], axis=1)
            keep = np.where((de > 1.0) & (np.arange(len(pos)) != i))[0]
            if len(keep) < 4:
                continue
            try:
                yh = float(METHODS[key](pm[keep], rss[keep], pm[i:i + 1])[0])
                if np.isfinite(yh):
                    errs.append(abs(yh - rss[i]))
            except Exception:  # noqa: BLE001
                pass
    return errs


def rmse(e):
    return round(float(np.sqrt(np.mean(np.square(e)))), 3) if e else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "sim2real_finetune.json"))
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print("[s2r] building synthetic pretraining cells...")
    sc = synth_cells()
    print(f"[s2r] synthetic cells={len(sc)}; pretraining...")
    base_model = SetInterpolator().to(device)
    fit(base_model, sc, device, epochs=60, lr=2e-3, m_per_px=1.0)

    rc = real_cells()
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(rc)), args.folds)
    agg = {"scratch": [], "zeroshot": [], "finetuned": [], "classical_RBF": []}
    for fi in range(args.folds):
        test = [rc[i] for i in folds[fi]]
        train = [rc[i] for i in range(len(rc)) if i not in set(folds[fi].tolist())]
        # scratch
        ms = fit(SetInterpolator().to(device), train, device, 120, 2e-3, M_PER_PX)
        agg["scratch"] += eval_real(ms, test, device)
        # zero-shot (pretrained only)
        agg["zeroshot"] += eval_real(copy.deepcopy(base_model), test, device)
        # fine-tuned (pretrained -> fine-tune on real train)
        mf = fit(copy.deepcopy(base_model), train, device, 60, 5e-4, M_PER_PX)
        agg["finetuned"] += eval_real(mf, test, device)
        agg["classical_RBF"] += classical_real(test)
        print(f"  fold {fi}: scratch={rmse(agg['scratch'])} zeroshot={rmse(agg['zeroshot'])} "
              f"finetuned={rmse(agg['finetuned'])} RBF={rmse(agg['classical_RBF'])}")

    out = {"n_synth_cells": len(sc), "n_real_cells": len(rc), "folds": args.folds,
           "rmse": {k: rmse(v) for k, v in agg.items()}}
    beat = out["rmse"]["finetuned"] is not None and out["rmse"]["classical_RBF"] is not None \
        and out["rmse"]["finetuned"] < out["rmse"]["classical_RBF"]
    out["finetuned_beats_classical"] = bool(beat)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))
    print(f"\n[s2r] RMSE: {out['rmse']}")
    print(f"[s2r] fine-tuned beats classical on real? {out['finetuned_beats_classical']}")
    print(f"[s2r] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

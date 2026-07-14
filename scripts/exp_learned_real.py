#!/usr/bin/env python
"""
Learned reconstruction ON REAL DATA (closes the "learned-vs-classical on real" gap).

Trains a geometry-aware DeepSet interpolator on real held-out-RP prediction, with a
held-out-CELL split (train cells vs test cells — no leakage). Compares on the SAME
test cells against classical IDW/RBF/GP and the wall-aware IDW, and ablates the wall
feature (learned #1 on real). Also reports MC-dropout UNCERTAINTY calibration on real
held-out RPs (real-data UQ — previously only synthetic).

  bash scripts/submit.sh 2g.35gb python scripts/exp_learned_real.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass.eval.stats import bootstrap_ci, paired_wilcoxon  # noqa: E402
from compass.realdata.learned_interp import M_PER_PX, SetInterpolator, build_features, make_samples  # noqa: E402
from compass.realdata.reconstruct import METHODS  # noqa: E402
from compass.realdata.walls import cell_rp_tables_px, load_wall_mask, wall_aware_loro, wall_matrix_m  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
MIN_RP = 10


def load_cells():
    cells = []
    for site, floors in SITES.items():
        for fl in floors:
            bar = load_wall_mask(site, fl)
            if bar is None:
                continue
            for _c, (pos, rss) in cell_rp_tables_px(site, fl, min_rp=MIN_RP).items():
                cells.append((pos, rss, wall_matrix_m(pos, bar)))
    return cells


def train_model(cells, use_walls, device, epochs=120, seed=0):
    torch.manual_seed(seed)
    data = make_samples(cells, use_walls=use_walls)
    if data is None:
        return None
    F, Mk, Cm, Y = (t.to(device) for t in data)
    model = SetInterpolator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b in range(0, n, 256):
            idx = perm[b:b + 256]
            pred = model(F[idx], Mk[idx], Cm[idx])
            loss = ((pred - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def eval_learned(model, cells, use_walls, device, n_mc=16):
    """Held-out-RP abs errors + (pred_std, |err|) pairs for UQ, over test cells."""
    errs, stds = [], []
    for pos, rss, W in cells:
        pm = pos * M_PER_PX
        for i in range(len(pos)):
            de = np.linalg.norm(pm - pm[i], axis=1)
            keep = np.where((de > 1.0) & (np.arange(len(pos)) != i))[0]
            if len(keep) < 4:
                continue
            f, m, c = build_features(pos, rss, W, keep, i, use_walls)
            f = torch.tensor(f[None]).to(device); m = torch.tensor(m[None]).to(device)
            c = torch.tensor([c], dtype=torch.float32).to(device)
            model.train()  # MC-dropout on
            ps = torch.stack([model(f, m, c) for _ in range(n_mc)]).cpu().numpy().ravel()
            errs.append(abs(float(ps.mean()) - rss[i]))
            stds.append(float(ps.std()))
    return errs, stds


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "learned_real.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "learned_real"))
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cells = load_cells()
    print(f"[learned-real] {len(cells)} cells")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cells))
    folds = np.array_split(order, args.folds)

    agg = {"learned+walls": [], "learned-nowalls": [], "IDW(p=2)": [], "RBF(mq)": [],
           "GP/Kriging": [], "wall-aware IDW": []}
    uq_std, uq_err = [], []
    per_cell_learned, per_cell_wallidw = [], []  # for Wilcoxon (learned vs classical)

    for fi in range(args.folds):
        test_idx = set(folds[fi].tolist())
        train_cells = [cells[i] for i in range(len(cells)) if i not in test_idx]
        test_cells = [cells[i] for i in folds[fi]]
        m_w = train_model(train_cells, True, device)
        m_nw = train_model(train_cells, False, device)
        ew, sw = eval_learned(m_w, test_cells, True, device)
        enw, _ = eval_learned(m_nw, test_cells, False, device)
        agg["learned+walls"] += ew
        agg["learned-nowalls"] += enw
        uq_std += sw; uq_err += ew
        # classical + wall-aware on the SAME test cells
        for pos, rss, W in test_cells:
            pm = pos * M_PER_PX
            for label, key in [("IDW(p=2)", "IDW(p=2)"), ("RBF(mq)", "RBF(multiquadric)"),
                               ("GP/Kriging", "GP/Kriging")]:
                for i in range(len(pos)):
                    de = np.linalg.norm(pm - pm[i], axis=1)
                    keep = np.where((de > 1.0) & (np.arange(len(pos)) != i))[0]
                    if len(keep) < 4:
                        continue
                    try:
                        yh = float(METHODS[key](pm[keep], rss[keep], pm[i:i + 1])[0])
                        if np.isfinite(yh):
                            agg[label].append(abs(yh - rss[i]))
                    except Exception:  # noqa: BLE001
                        pass
            agg["wall-aware IDW"] += wall_aware_loro(pos, rss, W, lam=16.0, buffer_m=1.0)
        print(f"  fold {fi}: learned+walls RMSE={rmse(ew):.2f} nowalls={rmse(enw):.2f} "
              f"wall-IDW={rmse(agg['wall-aware IDW']):.2f}")

    out = {"n_cells": len(cells), "folds": args.folds, "rmse": {}, "rmse_ci": {}}
    for m, e in agg.items():
        out["rmse"][m] = round(rmse(e), 3)
        out["rmse_ci"][m] = bootstrap_ci(np.abs(e)) if e else None
    # UQ calibration on real
    us, ue = np.array(uq_std), np.array(uq_err)
    ok = us.std() > 1e-6
    out["real_uq"] = {
        "err_unc_corr": round(float(np.corrcoef(us, ue)[0, 1]), 3) if ok else None,
        "picp_1sigma": round(float(np.mean(ue <= us + 1e-6)), 3) if ok else None,
        "sharpness_db": round(float(us.mean()), 3),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    ms = ["learned+walls", "learned-nowalls", "wall-aware IDW", "RBF(mq)", "IDW(p=2)", "GP/Kriging"]
    vals = [out["rmse"][m] for m in ms]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#3498db", "#8e44ad"]
    ax.bar(range(len(ms)), vals, color=colors)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("held-out-RP RMSE (dB)")
    ax.set_title(f"Learned vs classical reconstruction ON REAL DATA ({len(cells)} cells, cell-CV)")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "learned_real.png", dpi=120); plt.close(fig)

    print("\n[learned-real] held-out-RP RMSE (cell-CV):")
    for m in ms:
        print(f"  {m:18s} {out['rmse'][m]}")
    print(f"[learned-real] real UQ: {out['real_uq']}")
    print(f"[learned-real] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

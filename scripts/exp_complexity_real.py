#!/usr/bin/env python
"""
Complexity stratification on REAL data — does the geometry (wall-aware) advantage GROW
with structural complexity, mirroring the synthetic B1 finding (E13)?

Per cell, complexity = median wall-length crossed among its RP pairs (how walled the
cell is). Stratify cells into terciles and, per tercile, compare plain IDW (geometry-
blind) vs wall-aware IDW (λ tuned). If the advantage grows from low→high complexity,
the #1 mechanism is confirmed to scale with occlusion on real data too.

  bash scripts/submit.sh 1g.18gb python scripts/exp_complexity_real.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.eval.stats import bootstrap_ci  # noqa: E402
from compass.realdata.walls import cell_rp_tables_px, load_wall_mask, wall_aware_loro, wall_matrix_m  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
MIN_RP = 10
LAM = 16.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "complexity_real.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "complexity_real"))
    args = ap.parse_args()

    cells = []  # (complexity, plain_errs, wall_errs)
    for site, floors in SITES.items():
        for fl in floors:
            bar = load_wall_mask(site, fl)
            if bar is None:
                continue
            for _c, (pos, rss) in cell_rp_tables_px(site, fl, min_rp=MIN_RP).items():
                W = wall_matrix_m(pos, bar)
                iu = np.triu_indices(len(pos), 1)
                complexity = float(np.median(W[iu])) if len(iu[0]) else 0.0
                plain = wall_aware_loro(pos, rss, W, lam=0.0, buffer_m=1.0)
                wall = wall_aware_loro(pos, rss, W, lam=LAM, buffer_m=1.0)
                if plain and wall:
                    cells.append((complexity, plain, wall))
    print(f"[cx] cells: {len(cells)}")

    comp = np.array([c[0] for c in cells])
    terciles = np.quantile(comp, [1 / 3, 2 / 3])
    labels = ["low", "medium", "high"]
    out = {"n_cells": len(cells), "lam": LAM, "complexity_terciles_m": [round(float(t), 3) for t in terciles],
           "strata": {}}
    for i, lab in enumerate(labels):
        if i == 0:
            m = comp <= terciles[0]
        elif i == 1:
            m = (comp > terciles[0]) & (comp <= terciles[1])
        else:
            m = comp > terciles[1]
        sub = [cells[j] for j in range(len(cells)) if m[j]]
        pe = [e for c in sub for e in c[1]]
        we = [e for c in sub for e in c[2]]
        plain_rmse = float(np.sqrt(np.mean(np.square(pe))))
        wall_rmse = float(np.sqrt(np.mean(np.square(we))))
        out["strata"][lab] = {
            "n_cells": len(sub),
            "median_wall_len_m": round(float(np.median([c[0] for c in sub])), 3) if sub else None,
            "plain_idw_rmse": round(plain_rmse, 3),
            "wall_aware_rmse": round(wall_rmse, 3),
            "advantage_pct": round(100 * (plain_rmse - wall_rmse) / plain_rmse, 2),
        }

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    adv = [out["strata"][l]["advantage_pct"] for l in labels]
    ax.bar(labels, adv, color=["#95a5a6", "#3498db", "#e74c3c"])
    ax.set_ylabel("wall-aware advantage over plain IDW (%)")
    ax.set_xlabel("cell structural complexity (wall density)")
    ax.set_title(f"Real-data: geometry advantage vs complexity ({len(cells)} cells)")
    for i, a in enumerate(adv):
        ax.text(i, a, f"{a:.1f}%", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "complexity_real.png", dpi=120); plt.close(fig)

    print("[cx] wall-aware advantage by complexity tercile:")
    for lab in labels:
        s = out["strata"][lab]
        print(f"  {lab:6s} (n={s['n_cells']}, wall={s['median_wall_len_m']}m): "
              f"plain {s['plain_idw_rmse']} -> wall-aware {s['wall_aware_rmse']} ({s['advantage_pct']:+.1f}%)")
    print(f"[cx] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Wall-aware active sensing (#5 × #1) — turn Phase C's max-var≈space-fill null into a win.

Phase C found uncertainty-guided ≈ space-filling because a stationary GP's variance is
just distance-to-observed. On a WALLED site, the truly under-covered regions are those
separated from observations by walls (NLoS), not merely far in Euclidean terms. We make
BOTH the acquisition and the reconstruction wall-aware and test whether geometry-aware
acquisition beats geometry-blind space-filling on CMUQ (walled). EC Parking (open) is a
negative control — walls shouldn't matter there.

Strategies (reconstruction is wall-aware IDW for all, so only ACQUISITION differs):
  * wall_aware   — acquire the RP with max wall-aware isolation (d_euc + λ·wall).
  * space_filling— acquire the RP with max Euclidean isolation (walls ignored).
  * random.

  bash scripts/submit.sh 1g.18gb python scripts/exp_active_wall.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.eval.stats import bootstrap_ci, paired_wilcoxon  # noqa: E402
from compass.realdata.walls import (  # noqa: E402
    cell_rp_tables_px,
    load_wall_mask,
    wall_aware_predict,
    wall_isolation,
    wall_matrix_m,
)

REPO = Path(__file__).resolve().parents[1]
SITES = {"cmuq": [1, 2, 3], "ec_parking": [1]}  # walled test + open control
MIN_RP = 14
BUDGETS = [4, 6, 8, 10, 12]
LAM = 8.0


def run_cell(pos, rss, W, strategy, budgets, n_rep=5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pos)
    out = {b: [] for b in budgets}
    for rep in range(n_rep):
        obs = list(rng.choice(n, 4, replace=False))
        for step in range(max(budgets)):
            unobs = [i for i in range(n) if i not in obs]
            if len(unobs) < 2:
                break
            if (len(obs)) in out:
                yhat = wall_aware_predict(pos, rss, W, obs, unobs, lam=LAM)
                out[len(obs)].append(float(np.sqrt(np.mean((yhat - rss[unobs]) ** 2))))
            # acquire next
            if strategy == "random":
                nxt = int(rng.choice(unobs))
            elif strategy == "space_filling":
                pos_m = pos * 0.032
                d = np.array([min(np.hypot(*(pos_m[u] - pos_m[o])) for o in obs) for u in unobs])
                nxt = unobs[int(np.argmax(d))]
            else:  # wall_aware
                iso = wall_isolation(pos, W, obs, unobs, lam=LAM)
                nxt = unobs[int(np.argmax(iso))]
            obs.append(nxt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "active_wall.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "active_wall"))
    args = ap.parse_args()

    out = {"budgets": BUDGETS, "lam": LAM, "per_site": {}}
    for site, floors in SITES.items():
        cells = []
        for fl in floors:
            bar = load_wall_mask(site, fl)
            if bar is None:
                continue
            for _c, (pos, rss) in cell_rp_tables_px(site, fl, min_rp=MIN_RP).items():
                cells.append((pos, rss, wall_matrix_m(pos, bar)))
        if not cells:
            continue
        strat = {s: {b: [] for b in BUDGETS} for s in ("wall_aware", "space_filling", "random")}
        for pos, rss, W in cells:
            for s in strat:
                r = run_cell(pos, rss, W, s, BUDGETS)
                for b in BUDGETS:
                    if r[b]:
                        strat[s][b].append(float(np.mean(r[b])))
        site_out = {"n_cells": len(cells), "curves": {}, "wilcoxon_wall_vs_spacefill": {}}
        for s in strat:
            site_out["curves"][s] = {str(b): bootstrap_ci(strat[s][b]) for b in BUDGETS}
        for b in BUDGETS:
            a, o = strat["wall_aware"][b], strat["space_filling"][b]
            n = min(len(a), len(o))
            site_out["wilcoxon_wall_vs_spacefill"][str(b)] = paired_wilcoxon(np.array(a[:n]), np.array(o[:n]))
        out["per_site"][site] = site_out
        print(f"\n[{site}] {len(cells)} cells — held-out RMSE by budget:")
        for s in strat:
            vals = "  ".join(f"b{b}:{site_out['curves'][s][str(b)]['mean']:.2f}" for b in BUDGETS)
            print(f"  {s:14s} {vals}")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(out["per_site"]), figsize=(6 * len(out["per_site"]), 5), squeeze=False)
    colors = {"wall_aware": "#e74c3c", "space_filling": "#2980b9", "random": "#7f8c8d"}
    for ax, (site, so) in zip(axes[0], out["per_site"].items()):
        for s in ("wall_aware", "space_filling", "random"):
            m = [so["curves"][s][str(b)]["mean"] for b in BUDGETS]
            ax.plot(BUDGETS, m, "-o", color=colors[s], label=s)
        ax.set_title(f"{site} ({so['n_cells']} cells)"); ax.set_xlabel("budget (# RPs)")
        ax.set_ylabel("held-out RMSE (dB)"); ax.legend()
    fig.suptitle("Wall-aware vs geometry-blind active sensing")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "active_wall.png", dpi=120); plt.close(fig)
    print(f"\n[wall-active] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

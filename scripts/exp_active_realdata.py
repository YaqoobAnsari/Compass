#!/usr/bin/env python
"""
Phase C — active crowdsensing on REAL floor geometry (innovation #5).

For each viable cell (CMUQ + EC Parking, >= MIN_RP reference points), run every
acquisition strategy (max_variance / space_filling / coverage / random) through the
discrete-pool loop and record held-out-RP RMSE vs measurement budget. Aggregate
curves across cells with bootstrap CIs; test whether uncertainty-guided acquisition
beats space-filling / random at matched budget on real geometry.

  bash scripts/submit.sh 1g.18gb python scripts/exp_active_realdata.py
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
from compass.realdata.active import STRATEGIES, curve_at_budgets  # noqa: E402
from compass.realdata.reconstruct import cell_rp_tables  # noqa: E402
from compass.realdata.unicellular import UniCellularPaths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
MIN_RP = 14
BUDGETS = [4, 6, 8, 10, 12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "active_realdata.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "active_realdata"))
    args = ap.parse_args()

    paths = UniCellularPaths()
    tables = []
    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            if not paths.coords(site, fl).exists():
                continue
            for _cell, (pos, rss) in cell_rp_tables(site, fl, min_rp=MIN_RP).items():
                tables.append((pos, rss))
    print(f"[C] viable cells (>= {MIN_RP} RP): {len(tables)}")

    # per strategy: {budget: [per-cell mean RMSE]}
    agg = {s: {b: [] for b in BUDGETS} for s in STRATEGIES}
    for pos, rss in tables:
        for s in STRATEGIES:
            c = curve_at_budgets(pos, rss, s, BUDGETS, n_rep=5, length_scale=6.0, noise=3.0)
            for b in BUDGETS:
                if c[b]:
                    agg[s][b].append(float(np.mean(c[b])))

    out = {"n_cells": len(tables), "min_rp": MIN_RP, "budgets": BUDGETS, "curves": {}, "wilcoxon": {}}
    for s in STRATEGIES:
        out["curves"][s] = {str(b): bootstrap_ci(agg[s][b]) for b in BUDGETS}
    # Wilcoxon: max_variance vs each other strategy at each budget (paired over cells)
    for s in STRATEGIES:
        if s == "max_variance":
            continue
        out["wilcoxon"][s] = {}
        for b in BUDGETS:
            a = agg["max_variance"][b]
            o = agg[s][b]
            n = min(len(a), len(o))
            out["wilcoxon"][s][str(b)] = paired_wilcoxon(np.array(a[:n]), np.array(o[:n]))

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figure ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"max_variance": "#e74c3c", "space_filling": "#2980b9",
              "coverage": "#27ae60", "random": "#7f8c8d"}
    for s in STRATEGIES:
        means = [out["curves"][s][str(b)]["mean"] for b in BUDGETS]
        los = [out["curves"][s][str(b)]["lo"] for b in BUDGETS]
        his = [out["curves"][s][str(b)]["hi"] for b in BUDGETS]
        ax.plot(BUDGETS, means, "-o", color=colors[s], label=s)
        ax.fill_between(BUDGETS, los, his, color=colors[s], alpha=0.15)
    ax.set_xlabel("measurement budget (# RPs observed)")
    ax.set_ylabel("held-out-RP RMSE (dB)")
    ax.set_title(f"Phase C: active sensing on real geometry ({len(tables)} cells, 95% CI)")
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "active_realdata.png", dpi=120); plt.close(fig)

    print("\n[C] held-out RMSE by strategy (budget -> mean dB):")
    for s in STRATEGIES:
        vals = "  ".join(f"b{b}:{out['curves'][s][str(b)]['mean']:.2f}" for b in BUDGETS)
        print(f"  {s:14s} {vals}")
    print(f"[C] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

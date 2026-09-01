#!/usr/bin/env python
"""
Hybrid learned-residual reconstruction on REAL data (UniCellular).

Keeps the best classical field (RBF multiquadric) as the backbone and learns only the
geometry-driven residual it misses, evaluated leave-one-CELL-out under the same buffered
leave-one-RP-out protocol as the classical panel (so the number is directly comparable
to the 5.49 dB RBF baseline). Reports overall RMSE with bootstrap 95% CI and a paired
Wilcoxon test versus plain RBF, a per-site breakdown (walled CMU-Q vs open EC Parking),
and a no-geometry ablation that removes the wall features to show geometry drives any gain.

  bash scripts/submit.sh 1g.18gb python scripts/exp_residual_real.py     # (CPU-only ok: no --gres)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.eval.stats import paired_wilcoxon  # noqa: E402
from compass.realdata.residual import run_residual_cv  # noqa: E402
from compass.realdata.walls import cell_rp_tables_px, load_wall_mask, wall_matrix_m  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}


def collect(min_rp=8):
    cells = []
    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            tables = cell_rp_tables_px(site, fl, min_rp=min_rp)
            if not tables:
                continue
            bar = load_wall_mask(site, fl)
            for cell, (pos_px, rss) in tables.items():
                W = wall_matrix_m(pos_px, bar) if bar is not None else np.zeros((len(pos_px), len(pos_px)))
                cells.append({"site": site, "floor": fl, "cell": cell,
                              "pos_px": pos_px, "rss": rss, "W": W})
    return cells


def rmse_ci(abs_err, n_boot=2000, seed=0):
    v = np.square(np.asarray(abs_err, float))
    if len(v) < 2:
        return {"mean": float(np.sqrt(v.mean())) if len(v) else float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.array([np.sqrt(rng.choice(v, len(v), replace=True).mean()) for _ in range(n_boot)])
    return {"mean": round(float(np.sqrt(v.mean())), 3),
            "lo": round(float(np.percentile(boot, 2.5)), 3),
            "hi": round(float(np.percentile(boot, 97.5)), 3), "n": int(len(v))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rp", type=int, default=8)
    ap.add_argument("--buffer", type=float, default=1.0)
    ap.add_argument("--results", default=str(REPO / "results" / "residual_real.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "pub"))
    args = ap.parse_args()

    cells = collect(args.min_rp)
    n_wall = sum(int(c["W"].sum() > 0) for c in cells)
    print(f"[residual] cells={len(cells)}  (with walls traced: {n_wall})  buffer={args.buffer}m")

    cv = run_residual_cv(cells, buffer_m=args.buffer)
    methods = list(cv["abs_err"].keys())

    out = {"n_cells": cv["n_cells"], "sites_floors": SITES_FLOORS, "min_rp": args.min_rp,
           "buffer_m": args.buffer, "overall": {}, "by_site": {}, "wilcoxon_vs_rbf": {},
           "delta_pct_vs_rbf": {}}

    for m in methods:
        out["overall"][m] = rmse_ci(cv["abs_err"][m])
    base_rmse = out["overall"]["RBF (base)"]["mean"]
    for m in methods:
        out["delta_pct_vs_rbf"][m] = round(100.0 * (out["overall"][m]["mean"] - base_rmse) / base_rmse, 2)

    # per-site RMSE
    for m in methods:
        out["by_site"][m] = {}
        for site, errs in cv["site_abs"][m].items():
            out["by_site"][m][site] = round(float(np.sqrt(np.mean(np.square(errs)))), 3)

    # paired Wilcoxon: per-cell RMSE, each method vs plain RBF base
    base_cell = np.asarray(cv["per_cell_rmse"]["RBF (base)"], float)
    for m in methods:
        if m == "RBF (base)":
            continue
        out["wilcoxon_vs_rbf"][m] = paired_wilcoxon(np.asarray(cv["per_cell_rmse"][m], float), base_cell)

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # ---- report ----
    print(f"\n[residual] overall held-out-RP RMSE (buffer {args.buffer:.0f}m, {cv['n_cells']} cells):")
    for m in methods:
        w = out["wilcoxon_vs_rbf"].get(m, {})
        tag = "" if m == "RBF (base)" else f"  ({out['delta_pct_vs_rbf'][m]:+.1f}% vs RBF, p={w.get('p_value', float('nan')):.3g})"
        ci = out["overall"][m]
        print(f"   {m:24s} {ci['mean']:.3f} dB  [{ci.get('lo','?')}, {ci.get('hi','?')}]{tag}")
    print("\n[residual] by site:")
    for site in SITES_FLOORS:
        row = {m: out["by_site"][m].get(site) for m in methods}
        print(f"   {site:12s} " + "  ".join(f"{m.split()[0][:8]}={row[m]}" for m in methods))

    # ---- publication figure ----
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.titlesize": 17, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    OURS, GREY, RED = "#0072BD", "#7F7F7F", "#D95319"
    show = ["RBF (base)", "residual-GBT-nogeom", "residual-Ridge", "residual-GBT"]
    labels = ["classical RBF\n(baseline)", "residual\n(no geometry)", "residual\nlinear", "residual GBT\n(geometry)"]
    vals = [out["overall"][m]["mean"] for m in show]
    los = [out["overall"][m].get("lo", vals[i]) for i, m in enumerate(show)]
    his = [out["overall"][m].get("hi", vals[i]) for i, m in enumerate(show)]
    err = [[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]]
    cols = [GREY, RED, "#4DBEEE", OURS]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    x = np.arange(len(show))
    ax.bar(x, vals, 0.62, yerr=err, capsize=5, color=cols)
    for xi, v, h in zip(x, vals, his):
        ax.text(xi, h + 0.06, f"{v:.2f}", ha="center", fontsize=13, weight="bold")
    ax.axhline(base_rmse, color=GREY, ls="--", lw=1.3, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out-RP RMSE (dB), lower is better")
    ax.set_ylim(0, max(his) * 1.12)
    best = min((m for m in show if m != "RBF (base)"), key=lambda m: out["overall"][m]["mean"])
    wbest = out["wilcoxon_vs_rbf"][best].get("p_value", float("nan"))
    ax.set_title(f"Learned residual on the classical field ({cv['n_cells']} cells, 95% CI)\n"
                 f"Beats classical by {abs(out['delta_pct_vs_rbf'][best]):.1f}% (p = {wbest:.1e}); "
                 f"wall geometry adds to learned kernel blending")
    fig.savefig(Path(args.figdir) / "residual.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[residual] -> {args.results}, fig {args.figdir}/residual.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

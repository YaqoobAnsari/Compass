#!/usr/bin/env python
"""
TX-agnostic DEEP baseline on REAL data (IndoCell), evaluated under the identical
buffered leave-one-RP-out / leave-one-CELL-out protocol as the classical panel and the
learned residual, so every number on this slide is directly comparable.

This closes the one structural gap in the real-data track: every deep radio-map network
in the synthetic panel needs a transmitter and a dense raster, neither of which real
crowdsensing provides, so until now the real comparison had no deep competitor. A
Conditional Neural Process is transmitter-agnostic by construction and learns directly
from scattered points, making it the fair deep counterpart to Kriging here.

  sbatch --job-name=deep_real scripts/slurm_run.sh python scripts/exp_deep_real.py
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
from compass.realdata.deepsets import run_deep_cv  # noqa: E402
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
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--results", default=str(REPO / "results" / "deep_real.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "pub"))
    args = ap.parse_args()

    cells = collect(args.min_rp)
    print(f"[deep-real] cells={len(cells)} buffer={args.buffer}m epochs={args.epochs}", flush=True)

    deep = run_deep_cv(cells, buffer_m=args.buffer, epochs=args.epochs)
    resid = run_residual_cv(cells, buffer_m=args.buffer)

    # merge: classical base + learned residual + TX-agnostic deep, all on the same targets
    abs_err = dict(deep["abs_err"])
    per_cell = dict(deep["per_cell_rmse"])
    for k in ("residual-GBT", "residual-GBT-nogeom"):
        abs_err[k] = resid["abs_err"][k]
        per_cell[k] = resid["per_cell_rmse"][k]

    out = {"n_cells": deep["n_cells"], "buffer_m": args.buffer, "epochs": args.epochs,
           "overall": {}, "wilcoxon_vs_rbf": {}, "delta_pct_vs_rbf": {},
           "mean_uncertainty_db": {k: round(float(np.mean(v)), 3) for k, v in deep["uncertainty"].items() if v}}

    for m in abs_err:
        out["overall"][m] = rmse_ci(abs_err[m])
    base = out["overall"]["RBF (base)"]["mean"]
    for m in abs_err:
        out["delta_pct_vs_rbf"][m] = round(100.0 * (out["overall"][m]["mean"] - base) / base, 2)

    bc = np.asarray(per_cell["RBF (base)"], float)
    for m in abs_err:
        if m == "RBF (base)":
            continue
        v = np.asarray(per_cell[m], float)
        if len(v) == len(bc):
            out["wilcoxon_vs_rbf"][m] = paired_wilcoxon(v, bc)

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    print(f"\n[deep-real] held-out-RP RMSE ({deep['n_cells']} cells, buffer {args.buffer:.0f}m):")
    for m in sorted(out["overall"], key=lambda k: out["overall"][k]["mean"]):
        ci = out["overall"][m]
        w = out["wilcoxon_vs_rbf"].get(m, {})
        p = w.get("p_value")
        tag = "" if m == "RBF (base)" else f"  ({out['delta_pct_vs_rbf'][m]:+.1f}% vs RBF, p={p:.3g})" if p is not None else ""
        print(f"   {m:26s} {ci['mean']:.3f} dB  [{ci.get('lo','?')}, {ci.get('hi','?')}]{tag}")
    if out["mean_uncertainty_db"]:
        print(f"[deep-real] mean predictive sigma: {out['mean_uncertainty_db']}")

    # ---------------- figure ----------------
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.titlesize": 17, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    OURS, GREY, RED, CYAN = "#0072BD", "#7F7F7F", "#D95319", "#4DBEEE"
    show = ["CNP-deep (no geometry)", "CNP-deep (geometry)", "RBF (base)", "residual-GBT"]
    labels = ["deep CNP\n(no geometry)", "deep CNP\n(geometry)", "classical RBF\n(baseline)",
              "learned residual\n(ours)"]
    cols = [RED, GREY, GREY, OURS]
    show = [m for m in show if m in out["overall"]]
    vals = [out["overall"][m]["mean"] for m in show]
    los = [out["overall"][m].get("lo", v) for m, v in zip(show, vals)]
    his = [out["overall"][m].get("hi", v) for m, v in zip(show, vals)]
    err = [[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    x = np.arange(len(show))
    ax.bar(x, vals, 0.62, yerr=err, capsize=5, color=cols[:len(show)])
    for xi, v, h in zip(x, vals, his):
        ax.text(xi, h + 0.05, f"{v:.2f}", ha="center", fontsize=13, weight="bold")
    ax.axhline(base, color=GREY, ls="--", lw=1.3, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels[:len(show)])
    ax.set_ylabel("Held-out-RP RMSE (dB), lower is better")
    ax.set_ylim(0, max(his) * 1.15)
    ax.set_title(f"On real sparse data a transmitter-agnostic deep model does not beat tuned classical\n"
                 f"interpolation. The hybrid residual does ({deep['n_cells']} cells, leave-one-cell-out, 95% CI)")
    fig.savefig(Path(args.figdir) / "deep_real.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[deep-real] -> {args.results}, fig {args.figdir}/deep_real.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

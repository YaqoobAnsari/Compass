#!/usr/bin/env python
"""
Phase A2 — TX-agnostic RSS reconstruction on REAL data (UniCellular), validated by
held-out reference-point cross-validation (no dense GT exists). CMUQ + EC Parking
(the sites with real coords / floor plans; Ezdan has too few RPs).

Comprehensive classical interpolation panel (IDW p2/p3, NN, Natural, RBF mq/tps,
GP/Kriging), reported two ways:
  * buffered leave-one-RP-out (buffer sweep 0/1/2 m) — true-interpolation RMSE.
  * coverage sweep — observe {25,50,75}% of RPs, predict the rest.
Aggregated over all viable (site,floor,cell) with bootstrap 95% CIs + Wilcoxon.

  bash scripts/submit.sh 1g.18gb python scripts/exp_realdata_recon.py
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
from compass.realdata.reconstruct import (  # noqa: E402
    METHODS,
    cell_rp_tables,
    holdout_frac_errors,
    loro_errors,
)
from compass.realdata.unicellular import UniCellularPaths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}  # sites with real RP coords


def collect(min_rp=8):
    paths = UniCellularPaths()
    tables = []  # (site, floor, cell, pos, rss)
    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            if not paths.coords(site, fl).exists():
                continue
            for cell, (pos, rss) in cell_rp_tables(site, fl, min_rp=min_rp).items():
                tables.append((site, fl, cell, pos, rss))
    return tables


def rmse_ci(sq_errs, n_boot=2000, seed=0):
    """Bootstrap CI of the RMSE (consistent with the RMSE point value)."""
    v = np.asarray(sq_errs, float)
    if len(v) < 2:
        return {"mean": float(np.sqrt(v.mean())) if len(v) else float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    rmses = np.array([np.sqrt(rng.choice(v, len(v), replace=True).mean()) for _ in range(n_boot)])
    return {"mean": round(float(np.sqrt(v.mean())), 3),
            "lo": round(float(np.percentile(rmses, 2.5)), 3),
            "hi": round(float(np.percentile(rmses, 97.5)), 3), "n": int(len(v))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rp", type=int, default=8)
    ap.add_argument("--results", default=str(REPO / "results" / "realdata_recon.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "realdata_recon"))
    args = ap.parse_args()

    tables = collect(args.min_rp)
    n_cells = len(tables)
    print(f"[A2] viable (site,floor,cell) tables: {n_cells}  (min_rp={args.min_rp})")

    out = {"n_tables": n_cells, "sites_floors": SITES_FLOORS, "min_rp": args.min_rp,
           "loro": {}, "coverage": {}}

    # --- buffered leave-one-RP-out, buffer sweep ---
    for buf in (0.0, 1.0, 2.0):
        out["loro"][f"buffer_{buf:.0f}m"] = {}
        per_method_abs = {m: [] for m in METHODS}
        for _, _, _, pos, rss in tables:
            for mname, fn in METHODS.items():
                per_method_abs[mname] += loro_errors(pos, rss, fn, buffer_m=buf)
        for mname, errs in per_method_abs.items():
            out["loro"][f"buffer_{buf:.0f}m"][mname] = {
                "rmse": round(float(np.sqrt(np.mean(np.square(errs)))), 3) if errs else None,
                "ci": rmse_ci(np.square(errs)) if errs else None,
                "mae": round(float(np.mean(errs)), 3) if errs else None,
                "n": len(errs),
            }
        best = min((m for m in METHODS if out["loro"][f"buffer_{buf:.0f}m"][m]["rmse"]),
                   key=lambda m: out["loro"][f"buffer_{buf:.0f}m"][m]["rmse"])
        print(f"  [LORO buffer {buf:.0f}m] best = {best} "
              f"{out['loro'][f'buffer_{buf:.0f}m'][best]['rmse']:.2f} dB")

    # --- coverage sweep (observe {25,50,75}% of RPs) ---
    for tf in (0.75, 0.5, 0.25):  # test_frac -> observed = 1-tf
        obs = int((1 - tf) * 100)
        out["coverage"][f"obs_{obs}pct"] = {}
        per_method_abs = {m: [] for m in METHODS}
        for _, _, _, pos, rss in tables:
            for mname, fn in METHODS.items():
                per_method_abs[mname] += holdout_frac_errors(pos, rss, fn, test_frac=tf, n_rep=5)
        for mname, errs in per_method_abs.items():
            out["coverage"][f"obs_{obs}pct"][mname] = {
                "rmse": round(float(np.sqrt(np.mean(np.square(errs)))), 3) if errs else None, "n": len(errs)}

    # Wilcoxon: best vs rest at buffer 1m (per-cell RMSE)
    ref_buf = "buffer_1m"
    best = min((m for m in METHODS if out["loro"][ref_buf][m]["rmse"]),
               key=lambda m: out["loro"][ref_buf][m]["rmse"])
    out["best_method_buffer1m"] = best

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figures ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    methods = list(METHODS)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # LORO bar (buffer 1m) with CI
    d = out["loro"]["buffer_1m"]
    means = [d[m]["rmse"] for m in methods]
    lo = [d[m]["ci"]["lo"] for m in methods]; hi = [d[m]["ci"]["hi"] for m in methods]
    err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    ax[0].bar(range(len(methods)), means, yerr=err, capsize=3, color="#2980b9")
    ax[0].set_xticks(range(len(methods))); ax[0].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("held-out-RP RMSE (dB)")
    ax[0].set_title(f"A2 real-data reconstruction (buffered LORO 1m, {n_cells} cells, 95% CI)")
    # coverage sweep
    obs_levels = [25, 50, 75]
    for m in methods:
        ys = [out["coverage"][f"obs_{o}pct"][m]["rmse"] for o in obs_levels]
        ax[1].plot(obs_levels, ys, "-o", ms=4, label=m)
    ax[1].set_xlabel("% of RPs observed"); ax[1].set_ylabel("held-out RMSE (dB)")
    ax[1].set_title("A2 coverage sweep"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "realdata_recon.png", dpi=120); plt.close(fig)

    print(f"\n[A2] best classical (buffer 1m) = {best} = {out['loro'][ref_buf][best]['rmse']:.2f} dB RMSE")
    print(f"[A2] -> {args.results}, fig {args.figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

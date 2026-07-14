#!/usr/bin/env python
"""
Phase A2b — geometry-aware (wall-attenuation) reconstruction on REAL data (#1).

Two-part real-data validation of innovation #1:
  (1) EVIDENCE: at equal Euclidean distance, do wall-crossing RP pairs show larger
      RSS discontinuity than open pairs? (stratified by distance, with CIs).
  (2) EXPLOITATION: does wall-aware IDW (d_eff = d_euc + lambda*wall_len) beat plain
      IDW (lambda=0) on held-out-RP reconstruction? Sweep lambda; report best gain.

CMUQ floors 1–3 (walled academic building) are the test; EC Parking floor 1 (open
parking structure) is the negative control (walls should NOT help there).

  bash scripts/submit.sh 1g.18gb python scripts/exp_walls_realdata.py
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
from compass.realdata.walls import (  # noqa: E402
    M_PER_PX,
    cell_rp_tables_px,
    load_wall_mask,
    wall_aware_loro,
    wall_length_m,
    wall_matrix_m,
)

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1]}
LAMBDAS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
DIST_BINS = [(1, 2), (2, 3), (3, 4), (4, 6)]


def _pooled_rmse(cell_errs, lam):
    e = [x for c in cell_errs for x in c[lam]]
    return float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")


def nested_cv_gain(cell_errs, seed=0):
    """Unbiased gain: 2-fold over cells — pick λ minimising the OTHER fold's RMSE,
    evaluate on the held-out fold. Returns (plain_rmse, cv_rmse, gain_pct, picks)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cell_errs))
    folds = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    test_plain, test_cv, picks = [], [], []
    for f in range(2):
        train = [cell_errs[i] for i in folds[1 - f]]
        test = [cell_errs[i] for i in folds[f]]
        lam = min(LAMBDAS, key=lambda L: _pooled_rmse(train, L))
        picks.append(lam)
        test_plain += [x for c in test for x in c[0.0]]
        test_cv += [x for c in test for x in c[lam]]
    pr = float(np.sqrt(np.mean(np.square(test_plain))))
    cr = float(np.sqrt(np.mean(np.square(test_cv))))
    return pr, cr, round(100 * (pr - cr) / pr, 2), picks


def evidence(tables, bar):
    """Per distance-bin |dRSS| for high- vs low-wall pairs, with bootstrap CIs."""
    d, drss, wl = [], [], []
    for pos, rss in tables:
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dm = float(np.hypot(*(pos[i] - pos[j]))) * M_PER_PX
                if dm < 1 or dm > 6:
                    continue
                d.append(dm); drss.append(abs(rss[i] - rss[j]))
                wl.append(wall_length_m(bar, pos[i], pos[j]))
    d, drss, wl = np.array(d), np.array(drss), np.array(wl)
    res = {}
    for lo, hi in DIST_BINS:
        m = (d >= lo) & (d < hi)
        if m.sum() < 30 or wl[m].std() < 1e-9:
            continue
        thr = np.median(wl[m])
        hiw, low = drss[m][wl[m] > thr], drss[m][wl[m] <= thr]
        res[f"{lo}-{hi}m"] = {
            "n": int(m.sum()),
            "corr_wall_drss": round(float(np.corrcoef(wl[m], drss[m])[0, 1]), 3),
            "drss_high_wall": round(float(hiw.mean()), 2), "drss_high_ci": bootstrap_ci(hiw),
            "drss_low_wall": round(float(low.mean()), 2), "drss_low_ci": bootstrap_ci(low),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "walls_realdata.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "walls_realdata"))
    args = ap.parse_args()

    out = {"lambdas": LAMBDAS, "m_per_px": M_PER_PX, "per_site": {}, "evidence": {}}

    for site, floors in SITES_FLOORS.items():
        cell_errs = []  # per cell: {lam: [abs errs]}
        site_tables = []
        for fl in floors:
            bar = load_wall_mask(site, fl)
            if bar is None:
                print(f"  skip {site}/floor{fl}: no floor plan")
                continue
            tables = list(cell_rp_tables_px(site, fl).values())
            if not tables:
                continue
            site_tables += tables
            for pos, rss in tables:
                W = wall_matrix_m(pos, bar)
                cell_errs.append({lam: wall_aware_loro(pos, rss, W, lam=lam) for lam in LAMBDAS})
            print(f"  {site}/floor{fl}: {len(tables)} cells")
        if not site_tables:
            continue
        rmse = {lam: _pooled_rmse(cell_errs, lam) for lam in LAMBDAS}
        best_lam = min(LAMBDAS, key=lambda L: rmse[L])
        gain = 100 * (rmse[0.0] - rmse[best_lam]) / rmse[0.0]
        cv_plain, cv_rmse, cv_gain, cv_picks = nested_cv_gain(cell_errs)
        out["per_site"][site] = {
            "n_cells": len(site_tables),
            "rmse_by_lambda": {str(L): round(rmse[L], 3) for L in LAMBDAS},
            "plain_rmse": round(rmse[0.0], 3),
            "best_lambda": best_lam,
            "best_rmse": round(rmse[best_lam], 3),
            "gain_pct_oracle": round(gain, 2),
            "cv_plain_rmse": round(cv_plain, 3),
            "cv_rmse": round(cv_rmse, 3),
            "cv_gain_pct": cv_gain,
            "cv_lambda_picks": cv_picks,
        }
        # evidence stratification (use the site's first floor's mask + all its cells)
        bar0 = load_wall_mask(site, floors[0])
        if bar0 is not None:
            out["evidence"][site] = evidence(site_tables, bar0)
        print(f"  [{site}] plain IDW {rmse[0.0]:.2f} -> oracle-λ={best_lam} {rmse[best_lam]:.2f} "
              f"({gain:+.1f}%) | nested-CV {cv_plain:.2f}->{cv_rmse:.2f} ({cv_gain:+.1f}%, picks={cv_picks})")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figures ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for site, d in out["per_site"].items():
        ys = [d["rmse_by_lambda"][str(L)] for L in LAMBDAS]
        ax[0].plot(LAMBDAS, ys, "-o", label=f"{site} (n={d['n_cells']})")
    ax[0].set_xlabel("λ (wall-attenuation weight)"); ax[0].set_ylabel("held-out-RP RMSE (dB)")
    ax[0].set_title("A2b wall-aware reconstruction (λ=0 is plain IDW)"); ax[0].legend()
    # evidence bars for CMUQ
    if "cmuq" in out["evidence"]:
        ev = out["evidence"]["cmuq"]
        bins = list(ev)
        hiw = [ev[b]["drss_high_wall"] for b in bins]
        low = [ev[b]["drss_low_wall"] for b in bins]
        x = np.arange(len(bins))
        ax[1].bar(x - 0.2, hiw, 0.4, label="wall-crossing pairs", color="#c0392b")
        ax[1].bar(x + 0.2, low, 0.4, label="open pairs", color="#27ae60")
        ax[1].set_xticks(x); ax[1].set_xticklabels(bins)
        ax[1].set_xlabel("Euclidean distance bin"); ax[1].set_ylabel("|ΔRSS| (dB)")
        ax[1].set_title("A2b CMUQ: walls attenuate at equal distance"); ax[1].legend()
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "walls_realdata.png", dpi=120); plt.close(fig)

    print(f"\n[A2b] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

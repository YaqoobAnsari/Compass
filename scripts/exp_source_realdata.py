#!/usr/bin/env python
"""
Phase A5 — source / booster location approximation on REAL data, with a reliability
gate (the user's "confirm the approximation trends reliably before trusting it").

Real cells have NO ground-truth source and many carry a booster sharing the cell ID.
So we estimate a source per cell (log-distance fit + argmax + power-centroid) and,
without GT, decide trust from: (a) fit R² and a physically decaying slope, (b)
bootstrap stability, (c) agreement between the three estimators. We report what
FRACTION of cells yield a trustworthy point source, and characterise the rest.

  bash scripts/submit.sh 1g.18gb python scripts/exp_source_realdata.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.realdata.source import reliability  # noqa: E402
from compass.realdata.walls import cell_rp_tables_px  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rp", type=int, default=8)
    ap.add_argument("--results", default=str(REPO / "results" / "source_realdata.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "source_realdata"))
    args = ap.parse_args()

    cells = []
    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            for cell, (pos, rss) in cell_rp_tables_px(site, fl, min_rp=args.min_rp).items():
                rel = reliability(pos, rss)
                rel.update({"site": site, "floor": fl, "cell": int(cell), "n_rp": len(pos)})
                cells.append(rel)
        print(f"  {site}: cumulative cells={len(cells)}")

    n = len(cells)
    trust = [c for c in cells if c["trustworthy"]]
    r2 = np.array([c["r2"] for c in cells])
    n_exp = np.array([c["pathloss_n"] for c in cells])
    stab = np.array([c["stability_p90_m"] for c in cells])
    agree = np.array([c["method_agreement_m"] for c in cells])

    out = {
        "n_cells": n, "min_rp": args.min_rp,
        "n_trustworthy": len(trust),
        "frac_trustworthy": round(len(trust) / n, 3) if n else None,
        "r2_median": round(float(np.median(r2)), 3),
        "pathloss_n_median": round(float(np.median(n_exp)), 2),
        "pathloss_n_median_trusted": round(float(np.median([c["pathloss_n"] for c in trust])), 2) if trust else None,
        "stability_p90_median_m": round(float(np.median(stab[np.isfinite(stab)])), 2),
        "agreement_median_m": round(float(np.median(agree)), 2),
        "gate": {"r2_min": 0.3, "move_max_m": 8.0, "agree_max_m": 10.0},
        "cells": cells,
        "interpretation": (
            "Only cells passing the reliability gate get a point-source location; the "
            "rest are consistent with boosters / distributed sources and are left "
            "TX-agnostic. This is why COMPASS's real-data model is TX-agnostic by default."
        ),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figures ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(r2, bins=25, color="#2980b9"); ax[0].axvline(0.3, color="r", ls="--", label="gate R²=0.3")
    ax[0].set_xlabel("log-distance fit R²"); ax[0].set_ylabel("# cells")
    ax[0].set_title(f"A5 source fit quality (n={n})"); ax[0].legend()
    ax[1].hist(n_exp[np.isfinite(n_exp)], bins=25, color="#8e44ad")
    ax[1].set_xlabel("estimated pathloss exponent n"); ax[1].set_title("A5 pathloss slope")
    fin = np.isfinite(stab)
    ax[2].scatter(r2[fin], stab[fin], c=["#27ae60" if c["trustworthy"] else "#c0392b" for c, f in zip(cells, fin) if f], s=18)
    ax[2].set_xlabel("R²"); ax[2].set_ylabel("bootstrap stability p90 (m)")
    ax[2].set_title(f"A5 trustworthy = {len(trust)}/{n} (green)")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "source_realdata.png", dpi=120); plt.close(fig)

    print(f"\n[A5] cells={n}  trustworthy point-source={len(trust)} ({out['frac_trustworthy']*100:.0f}%)")
    print(f"[A5] fit R² median={out['r2_median']}  pathloss n median={out['pathloss_n_median']} "
          f"(trusted {out['pathloss_n_median_trusted']})")
    print(f"[A5] stability p90 median={out['stability_p90_median_m']} m  "
          f"method-agreement median={out['agreement_median_m']} m")
    print(f"[A5] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

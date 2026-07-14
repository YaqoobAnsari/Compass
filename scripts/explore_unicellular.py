#!/usr/bin/env python
"""
Explore Eric's UniCellular real dataset and TEST the two core COMPASS assumptions
on REAL data (the synthetic Phase-2.1 left them ambiguous):

  Q1 DEVICE HETEROGENEITY (Pillar 2 / spike 3): at a fixed reference point and
     transmitter, do different phones report systematically different RSS?
     -> measures the real per-device offset spread (we assumed ~4 dB).

  Q2 TRAJECTORY CORRELATION (Pillar 1+2 / Phase 2.1): along a real mobile walk,
     for a fixed transmitter, is consecutive-scan RSS autocorrelated (NOT i.i.d.)?
     -> if yes, REAL trajectories carry the structure our model wants to exploit.

Saves figures to figures/realdata/ and a results JSON. Read-only on the dataset.

Run on the cluster:
  bash scripts/submit.sh 1g.18gb python scripts/explore_unicellular.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.realdata.unicellular import (  # noqa: E402
    RSS_COL,
    UniCellularPaths,
    best_shared_rp,
    device_offsets_at_rp,
    load_floor,
    mobile_sequences,
    top_transmitters,
)

REPO = Path(__file__).resolve().parents[1]


def autocorr(x: np.ndarray, max_lag: int = 30) -> np.ndarray:
    x = x - x.mean()
    var = (x * x).mean()
    if var <= 0:
        return np.zeros(max_lag)
    return np.array([1.0] + [float((x[:-k] * x[k:]).mean() / var) for k in range(1, max_lag)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="ezdan_tower4")
    ap.add_argument("--floor", type=int, default=5)
    ap.add_argument("--fig-dir", default=str(REPO / "figures" / "realdata"))
    ap.add_argument("--results", default=str(REPO / "results" / "unicellular_exploration.json"))
    args = ap.parse_args()

    paths = UniCellularPaths()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {"site": args.site, "floor": args.floor}

    stat = load_floor(args.site, "stationary", args.floor)
    mob = load_floor(args.site, "mobile", args.floor)
    out["stationary_rows"] = int(len(stat))
    out["mobile_rows"] = int(len(mob))
    out["phones"] = sorted(stat["phoneName"].unique().tolist())
    out["n_transmitters_stationary"] = int(stat["transmitter_id"].nunique())
    print(f"[real] {args.site} floor{args.floor}: stationary={len(stat)} mobile={len(mob)} "
          f"phones={len(out['phones'])} tx={out['n_transmitters_stationary']}")

    # ---- Q1: device heterogeneity ----
    tx = top_transmitters(stat, 1)[0]
    rp = best_shared_rp(stat, tx)
    offs = device_offsets_at_rp(stat, tx, rp)
    spread = float(offs.max() - offs.min())
    out["device_heterogeneity"] = {
        "transmitter_id": int(tx), "reference_point": int(rp),
        "per_phone_mean_rss_dbm": {k: round(float(v), 1) for k, v in offs.items()},
        "spread_db": round(spread, 1),
        "std_db": round(float(offs.std()), 1),
    }
    print(f"[Q1] device spread at RP{rp}, tx{tx}: {spread:.1f} dB across phones "
          f"({', '.join(f'{k}:{v:.0f}' for k, v in offs.items())})")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(offs)), offs.values, color="#8e44ad")
    ax.set_xticks(range(len(offs)))
    ax.set_xticklabels(offs.index, rotation=25, fontsize=8)
    ax.set_ylabel("mean RSS (dBm)")
    ax.set_title(f"Q1 device heterogeneity — same RP{rp}, same tx{tx}\nspread = {spread:.1f} dB across phones")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.site}_f{args.floor}_device_heterogeneity.png", dpi=120)
    plt.close(fig)

    # ---- Q2: real trajectory autocorrelation ----
    mtx = top_transmitters(mob, 1)[0]
    seqs = mobile_sequences(mob, mtx)
    phone = max(seqs, key=lambda p: len(seqs[p]))
    series = seqs[phone][RSS_COL].to_numpy(dtype=float)
    ac = autocorr(series, max_lag=30)
    # i.i.d. reference: shuffle destroys order
    rng = np.random.default_rng(0)
    ac_shuf = autocorr(rng.permutation(series), max_lag=30)
    lag1 = float(ac[1]) if len(ac) > 1 else 0.0
    out["trajectory_correlation"] = {
        "transmitter_id": int(mtx), "phone": phone, "n_scans": int(len(series)),
        "lag1_autocorr": round(lag1, 3),
        "lag1_autocorr_shuffled": round(float(ac_shuf[1]), 3),
        "correlated_not_iid": bool(lag1 > 0.2 and lag1 > 3 * abs(ac_shuf[1])),
    }
    print(f"[Q2] real walk autocorr (tx{mtx}, {phone}, n={len(series)}): "
          f"lag-1 = {lag1:.2f} (shuffled {ac_shuf[1]:.2f}) -> "
          f"{'CORRELATED' if out['trajectory_correlation']['correlated_not_iid'] else 'weak/iid'}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(series, lw=0.8, color="#c0392b")
    ax[0].set_title(f"Q2 real mobile walk RSS (tx{mtx}, {phone})")
    ax[0].set_xlabel("scan index (walk order)")
    ax[0].set_ylabel("RSS (dBm)")
    ax[1].plot(ac, "b-o", ms=3, label="true order")
    ax[1].plot(ac_shuf, "gray", ls="--", label="shuffled (i.i.d. ref)")
    ax[1].axhline(np.exp(-1), color="k", ls=":", lw=0.8, label="1/e")
    ax[1].set_xlabel("lag (scans)")
    ax[1].set_ylabel("autocorrelation")
    ax[1].set_title(f"lag-1 = {lag1:.2f} (real) vs {ac_shuf[1]:.2f} (shuffled)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.site}_f{args.floor}_trajectory_autocorr.png", dpi=120)
    plt.close(fig)

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))
    print(f"\n[real] figures -> {fig_dir}")
    print(f"[real] results -> {args.results}")
    print(f"[real] VERDICT: device spread {spread:.1f} dB (real, vs our 4 dB assumption); "
          f"trajectory lag-1 autocorr {lag1:.2f} (real walks are "
          f"{'correlated' if lag1 > 0.2 else 'weakly correlated'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

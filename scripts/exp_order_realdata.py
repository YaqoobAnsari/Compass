#!/usr/bin/env python
"""
Phase A3 — does temporal ORDER carry usable signal in REAL crowdsensed RSS streams?
(tests COMPASS innovation #3 on real mobile walks; position-free.)

For every (site, floor, transmitter, phone) mobile sequence of length >= MIN_LEN:
  * lag-1 autocorrelation (true vs shuffled order),
  * gap-filling RMSE: impute held-out scans from temporal neighbours on the TRUE
    order vs a SHUFFLED order vs the series MEAN.
Aggregated with bootstrap CIs; per-sequence paired Wilcoxon (true RMSE vs shuffled).

  bash scripts/submit.sh 1g.18gb python scripts/exp_order_realdata.py
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
from compass.realdata.order import gap_fill_errors, lag1_autocorr, rmse  # noqa: E402
from compass.realdata.unicellular import (  # noqa: E402
    RECON_COLS,
    RSS_COL,
    load_floor,
    mobile_sequences,
    top_transmitters,
)

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [0, 1, 2]}
MIN_LEN = 30
TOP_TX = 8


def rmse_ci(abs_errs, n_boot=2000, seed=0):
    """Bootstrap CI of the RMSE (consistent with the plotted RMSE bar)."""
    v = np.square(np.asarray(abs_errs, float))
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    rmses = np.array([np.sqrt(rng.choice(v, len(v), replace=True).mean()) for _ in range(n_boot)])
    return {"mean": round(float(np.sqrt(v.mean())), 3),
            "lo": round(float(np.percentile(rmses, 2.5)), 3),
            "hi": round(float(np.percentile(rmses, 97.5)), 3), "n": int(len(v))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "order_realdata.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "order_realdata"))
    args = ap.parse_args()

    autocorr_true, autocorr_shuf = [], []
    err = {"true": [], "shuffled": [], "mean": []}
    seq_rmse = {"true": [], "shuffled": []}  # per-sequence, for paired Wilcoxon
    n_seq = 0
    per_site = {}

    for site, floors in SITES_FLOORS.items():
        per_site.setdefault(site, {"n_seq": 0, "true": [], "shuffled": [], "mean": [], "ac": []})
        for fl in floors:
            try:
                df = load_floor(site, "mobile", fl, usecols=RECON_COLS)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  skip {site}/floor{fl}: {e}")
                continue
            for tx in top_transmitters(df, n=TOP_TX, min_phones=2):
                for phone, seq in mobile_sequences(df, tx).items():
                    y = seq[RSS_COL].to_numpy(dtype=float)
                    if len(y) < MIN_LEN or y.std() < 1e-6:
                        continue
                    ac = lag1_autocorr(y)
                    if not np.isfinite(ac):
                        continue
                    rng_perm = np.random.default_rng(n_seq)
                    e = gap_fill_errors(y, test_frac=0.4, n_rep=5, seed=n_seq)
                    if not e["true"]:
                        continue
                    n_seq += 1
                    autocorr_true.append(ac)
                    autocorr_shuf.append(lag1_autocorr(y[rng_perm.permutation(len(y))]))
                    for k in err:
                        err[k] += e[k]
                        per_site[site][k] += e[k]
                    seq_rmse["true"].append(rmse(e["true"]))
                    seq_rmse["shuffled"].append(rmse(e["shuffled"]))
                    per_site[site]["n_seq"] += 1
                    per_site[site]["ac"].append(ac)
            print(f"  {site}/floor{fl}: cumulative sequences={n_seq}")

    out = {
        "n_sequences": n_seq, "min_len": MIN_LEN, "sites_floors": SITES_FLOORS,
        "autocorr_true_median": round(float(np.median(autocorr_true)), 4),
        "autocorr_true_mean": round(float(np.mean(autocorr_true)), 4),
        "autocorr_shuffled_median": round(float(np.median(autocorr_shuf)), 4),
        "gap_fill_rmse": {k: round(rmse(err[k]), 3) for k in err},
        "gap_fill_rmse_ci": {k: rmse_ci(err[k]) for k in err},
        "wilcoxon_true_vs_shuffled": paired_wilcoxon(seq_rmse["true"], seq_rmse["shuffled"]),
        "per_site": {},
    }
    for site, d in per_site.items():
        if d["n_seq"]:
            out["per_site"][site] = {
                "n_seq": d["n_seq"],
                "autocorr_true_median": round(float(np.median(d["ac"])), 4),
                "gap_fill_rmse": {k: round(rmse(d[k]), 3) for k in ("true", "shuffled", "mean")},
            }

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figures ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(autocorr_true, bins=30, alpha=0.75, label="true order", color="#27ae60")
    ax[0].hist(autocorr_shuf, bins=30, alpha=0.6, label="shuffled", color="#c0392b")
    ax[0].axvline(out["autocorr_true_median"], color="#27ae60", ls="--")
    ax[0].set_xlabel("lag-1 autocorrelation"); ax[0].set_ylabel("# sequences")
    ax[0].set_title(f"A3 order signal in real walks (n={n_seq})"); ax[0].legend()
    ks = ["true", "shuffled", "mean"]
    vals = [out["gap_fill_rmse"][k] for k in ks]
    ci = out["gap_fill_rmse_ci"]
    yerr = [[vals[i] - ci[k]["lo"] for i, k in enumerate(ks)],
            [ci[k]["hi"] - vals[i] for i, k in enumerate(ks)]]
    ax[1].bar(ks, vals, yerr=yerr, capsize=4, color=["#27ae60", "#c0392b", "#7f8c8d"])
    ax[1].set_ylabel("gap-fill RMSE (dB)")
    ax[1].set_title("A3 temporal gap-filling (true vs shuffled vs mean)")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "order_realdata.png", dpi=120); plt.close(fig)

    print(f"\n[A3] n_sequences={n_seq}")
    print(f"[A3] lag-1 autocorr median: true={out['autocorr_true_median']:.3f} "
          f"shuffled={out['autocorr_shuffled_median']:.3f}")
    print(f"[A3] gap-fill RMSE: true={out['gap_fill_rmse']['true']:.2f} "
          f"shuffled={out['gap_fill_rmse']['shuffled']:.2f} mean={out['gap_fill_rmse']['mean']:.2f} dB")
    w = out["wilcoxon_true_vs_shuffled"]
    print(f"[A3] Wilcoxon true<shuffled: p={w.get('p_value')}, n={w.get('n')}")
    print(f"[A3] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

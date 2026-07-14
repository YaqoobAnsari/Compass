#!/usr/bin/env python
"""
Cross-device generalization (#4) — can we calibrate an UNSEEN phone from a few anchor
measurements, using the crowd of other phones as the reference frame?

Leave-one-phone-out on real UniCellular: for each held-out phone p, the reference RSS
at each (transmitter, RP) is the mean over the OTHER phones (the crowd consensus). p's
device offset is estimated from k random ANCHOR cells (rss_p − reference) and applied to
predict p's RSS at its remaining cells. We sweep k and compare calibrated vs
uncalibrated error — testing whether few anchors suffice to calibrate a new device.

  bash scripts/submit.sh 1g.18gb python scripts/exp_cross_device.py
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
from compass.realdata.device import device_long_table  # noqa: E402
from compass.realdata.unicellular import PHONE_COL, RSS_COL, TX_COL  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
K_ANCHORS = [1, 2, 3, 5, 8]
MIN_TEST = 6  # p needs this many non-anchor cells to evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "cross_device.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "cross_device"))
    args = ap.parse_args()

    # accumulate abs errors per k, plus the no-calibration baseline
    errs = {k: [] for k in K_ANCHORS}
    errs["nocal"] = []
    true_offsets = []
    rng = np.random.default_rng(0)
    n_devices = 0

    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            try:
                g = device_long_table(site, fl)
            except (FileNotFoundError, RuntimeError):
                continue
            phones = sorted(g[PHONE_COL].unique())
            if len(phones) < 2:
                continue
            for p in phones:
                gp = g[g[PHONE_COL] == p]
                others = g[g[PHONE_COL] != p]
                ref = others.groupby([TX_COL, "rpNumber"])[RSS_COL].mean()
                # residuals of p vs reference at shared (tx,rp)
                resid = []
                for _, row in gp.iterrows():
                    key = (row[TX_COL], row["rpNumber"])
                    if key in ref.index:
                        resid.append(row[RSS_COL] - float(ref[key]))
                resid = np.array(resid)
                if len(resid) < max(K_ANCHORS) + MIN_TEST:
                    continue
                n_devices += 1
                true_offsets.append(float(resid.mean()))
                for k in K_ANCHORS:
                    for _rep in range(10):
                        idx = rng.permutation(len(resid))
                        anchors, test = idx[:k], idx[k:]
                        b_hat = float(resid[anchors].mean())
                        errs[k] += list(np.abs(resid[test] - b_hat))
                # no-calibration baseline (predict reference, offset=0)
                errs["nocal"] += list(np.abs(resid))
        print(f"  {site}: cumulative held-out devices={n_devices}")

    out = {
        "n_device_instances": n_devices,
        "true_offset_mean_abs_db": round(float(np.mean(np.abs(true_offsets))), 3) if true_offsets else None,
        "true_offset_p90_abs_db": round(float(np.percentile(np.abs(true_offsets), 90)), 3) if true_offsets else None,
        "rmse_by_k": {}, "rmse_nocal": None,
    }
    out["rmse_nocal"] = round(float(np.sqrt(np.mean(np.square(errs["nocal"])))), 3)
    for k in K_ANCHORS:
        out["rmse_by_k"][str(k)] = round(float(np.sqrt(np.mean(np.square(errs[k])))), 3)
    out["rmse_ci_by_k"] = {str(k): bootstrap_ci(np.abs(errs[k])) for k in K_ANCHORS}

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = K_ANCHORS
    ys = [out["rmse_by_k"][str(k)] for k in ks]
    ax.axhline(out["rmse_nocal"], color="#7f8c8d", ls="--", label=f"no calibration ({out['rmse_nocal']} dB)")
    ax.plot(ks, ys, "-o", color="#27ae60", label="anchor-calibrated new device")
    ax.set_xlabel("# anchor measurements for the new device")
    ax.set_ylabel("cross-device prediction RMSE (dB)")
    ax.set_title(f"Cross-device calibration of UNSEEN phones ({n_devices} device instances)")
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "cross_device.png", dpi=120); plt.close(fig)

    print(f"\n[cross-device] {n_devices} held-out device instances")
    print(f"  true offset |mean|={out['true_offset_mean_abs_db']} p90={out['true_offset_p90_abs_db']} dB")
    print(f"  no-calibration RMSE = {out['rmse_nocal']} dB")
    for k in K_ANCHORS:
        print(f"  k={k} anchors -> {out['rmse_by_k'][str(k)]} dB")
    print(f"[cross-device] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

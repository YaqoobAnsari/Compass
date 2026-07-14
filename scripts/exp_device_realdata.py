#!/usr/bin/env python
"""
Phase A4 — device heterogeneity & cross-device calibration on REAL data (#4).

1. Magnitude: cross-phone RSS disagreement at shared reference points.
2. Estimate a per-device offset (one-way fixed effect) and validate RELIABILITY
   (offsets estimated on half the shared cells predict the other half + remove
   cross-device variance) — the "trends reliably" bar before we trust it.
3. Reconstruction impact: does calibrating per-phone offsets before pooling improve
   held-out-RP reconstruction vs pooling raw multi-device RSS?

  bash scripts/submit.sh 1g.18gb python scripts/exp_device_realdata.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.realdata.device import (  # noqa: E402
    calibration_reliability,
    cell_offset_spread,
    device_long_table,
    estimate_offsets,
    shared_cells,
)
from compass.realdata.reconstruct import M_PER_PX, METHODS, loro_errors  # noqa: E402
from compass.realdata.unicellular import (  # noqa: E402
    PHONE_COL,
    RSS_COL,
    TX_COL,
    load_rp_coords,
)

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
RECON_METHOD = "RBF(multiquadric)"  # best from A2


def recon_impact(g, coords, offsets, min_rp=8):
    """Held-out-RP abs errors pooling multi-device RSS, raw vs device-calibrated.
    Returns (raw_errs, cal_errs) over all cells with >= min_rp coord'd RPs."""
    raw_e, cal_e = [], []
    g = g.copy()
    g["off"] = g[PHONE_COL].map(offsets).fillna(0.0)
    g["cal"] = g[RSS_COL] - g["off"]
    fn = METHODS[RECON_METHOD]
    for cell, sub in g.groupby(TX_COL):
        raw = sub.groupby("rpNumber")[RSS_COL].mean()
        cal = sub.groupby("rpNumber")["cal"].mean()
        rps = [int(r) for r in raw.index if int(r) in coords]
        if len(rps) < min_rp:
            continue
        pos = np.array([coords[r] for r in rps], float) * M_PER_PX
        raw_e += loro_errors(pos, np.array([raw[r] for r in rps]), fn, buffer_m=1.0)
        cal_e += loro_errors(pos, np.array([cal[r] for r in rps]), fn, buffer_m=1.0)
    return raw_e, cal_e


def recon_impact_single_device(g, coords, offsets, min_rp=8, seed=0, n_rep=5):
    """Realistic crowdsensing regime: assign each RP to ONE (random) phone that
    observed it, so the field carries device-induced spatial bias. Estimate offsets
    from shared anchors (passed in), apply, and compare held-out-RP RMSE raw vs cal."""
    rng = np.random.default_rng(seed)
    raw_e, cal_e = [], []
    g = g.copy()
    g["off"] = g[PHONE_COL].map(offsets).fillna(0.0)
    g["cal"] = g[RSS_COL] - g["off"]
    fn = METHODS[RECON_METHOD]
    for _cell, sub in g.groupby(TX_COL):
        rps = [int(r) for r in sub["rpNumber"].unique() if int(r) in coords]
        if len(rps) < min_rp:
            continue
        pos = np.array([coords[r] for r in rps], float) * M_PER_PX
        by_rp = {r: sub[sub["rpNumber"] == r] for r in rps}
        for _ in range(n_rep):
            raw_v, cal_v = [], []
            for r in rps:
                rr = by_rp[r]
                pick = rr.iloc[int(rng.integers(len(rr)))]
                raw_v.append(pick[RSS_COL]); cal_v.append(pick["cal"])
            raw_e += loro_errors(pos, np.array(raw_v), fn, buffer_m=1.0)
            cal_e += loro_errors(pos, np.array(cal_v), fn, buffer_m=1.0)
    return raw_e, cal_e


def _contiguous_labels(pos, k):
    """Split RPs into k spatially-contiguous zones along their principal axis."""
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    t = c @ vt[0]
    ranks = np.argsort(np.argsort(t))
    return np.clip(ranks * k // len(t), 0, k - 1)


def recon_impact_contiguous(g, coords, offsets, min_rp=8, seed=0, n_rep=5):
    """Most realistic regime: each spatially-contiguous zone is covered by ONE phone
    (real readings), so device bias is spatially COHERENT. Estimate offsets from
    anchors, apply, compare held-out-RP RMSE raw vs cal."""
    rng = np.random.default_rng(seed)
    raw_e, cal_e = [], []
    g = g.copy()
    g["off"] = g[PHONE_COL].map(offsets).fillna(0.0)
    fn = METHODS[RECON_METHOD]
    for _cell, sub in g.groupby(TX_COL):
        rps = [int(r) for r in sub["rpNumber"].unique() if int(r) in coords]
        if len(rps) < min_rp:
            continue
        cell_phones = sorted(sub[PHONE_COL].unique())
        k = len(cell_phones)
        if k < 2:
            continue
        pos = np.array([coords[r] for r in rps], float) * M_PER_PX
        labels = _contiguous_labels(pos, k)
        by_rp = {r: sub[sub["rpNumber"] == r] for r in rps}
        for _ in range(n_rep):
            perm = list(rng.permutation(cell_phones))  # randomise which zone gets which phone
            raw_v, cal_v = [], []
            for r, lab in zip(rps, labels):
                ph = perm[int(lab)]
                rr = by_rp[r]
                sel = rr[rr[PHONE_COL] == ph]
                pick = sel.iloc[0] if len(sel) else rr.iloc[int(rng.integers(len(rr)))]
                raw_v.append(pick[RSS_COL]); cal_v.append(pick[RSS_COL] - pick["off"])
            raw_e += loro_errors(pos, np.array(raw_v), fn, buffer_m=1.0)
            cal_e += loro_errors(pos, np.array(cal_v), fn, buffer_m=1.0)
    return raw_e, cal_e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "device_realdata.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "device_realdata"))
    args = ap.parse_args()

    all_spread = []
    all_raw, all_cal = [], []
    all_sd_raw, all_sd_cal = [], []  # single-device-per-RP (random crowdsensing)
    all_cz_raw, all_cz_cal = [], []  # contiguous device zones (realistic crowdsensing)
    per_floor = {}
    global_offsets = {}

    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            try:
                g = device_long_table(site, fl)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  skip {site}/floor{fl}: {e}")
                continue
            sc = shared_cells(g, min_phones=2)
            if sc.empty or sc[PHONE_COL].nunique() < 2:
                print(f"  {site}/floor{fl}: no shared multi-phone cells")
                continue
            spread = cell_offset_spread(sc)
            off = estimate_offsets(sc)
            rel = calibration_reliability(sc, n_rep=10)
            coords = load_rp_coords(site, fl)
            raw_e, cal_e = ([], [])
            sd_raw, sd_cal = ([], [])
            cz_raw, cz_cal = ([], [])
            if coords:
                raw_e, cal_e = recon_impact(g, coords, off)
                sd_raw, sd_cal = recon_impact_single_device(g, coords, off)
                cz_raw, cz_cal = recon_impact_contiguous(g, coords, off)
            all_spread += list(spread)
            all_raw += raw_e; all_cal += cal_e
            all_sd_raw += sd_raw; all_sd_cal += sd_cal
            all_cz_raw += cz_raw; all_cz_cal += cz_cal
            per_floor[f"{site}/floor{fl}"] = {
                "n_shared_cells": int(sc.groupby([TX_COL, 'rpNumber']).ngroups),
                "n_phones": int(sc[PHONE_COL].nunique()),
                "offset_spread_median_dB": round(float(np.median(spread)), 2),
                "offset_spread_p90_dB": round(float(np.percentile(spread, 90)), 2),
                "device_offsets_dB": {k: round(float(v), 2) for k, v in off.items()},
                "reliability": rel,
                "recon_raw_rmse": round(float(np.sqrt(np.mean(np.square(raw_e)))), 3) if raw_e else None,
                "recon_cal_rmse": round(float(np.sqrt(np.mean(np.square(cal_e)))), 3) if cal_e else None,
                "recon_sd_raw_rmse": round(float(np.sqrt(np.mean(np.square(sd_raw)))), 3) if sd_raw else None,
                "recon_sd_cal_rmse": round(float(np.sqrt(np.mean(np.square(sd_cal)))), 3) if sd_cal else None,
            }
            global_offsets[f"{site}/floor{fl}"] = per_floor[f"{site}/floor{fl}"]["device_offsets_dB"]
            print(f"  {site}/floor{fl}: shared={per_floor[f'{site}/floor{fl}']['n_shared_cells']} "
                  f"spread_med={np.median(spread):.1f}dB corr={rel['offset_corr_mean']} "
                  f"varRed={rel['var_reduction_mean']} "
                  f"recon raw={per_floor[f'{site}/floor{fl}']['recon_raw_rmse']} "
                  f"cal={per_floor[f'{site}/floor{fl}']['recon_cal_rmse']}")

    def _rmse(x):
        return float(np.sqrt(np.mean(np.square(x)))) if x else None

    raw_rmse, cal_rmse = _rmse(all_raw), _rmse(all_cal)
    sd_raw_rmse, sd_cal_rmse = _rmse(all_sd_raw), _rmse(all_sd_cal)
    cz_raw_rmse, cz_cal_rmse = _rmse(all_cz_raw), _rmse(all_cz_cal)

    def _imp(a, b):
        return round(100 * (a - b) / a, 2) if (a and b) else None
    out = {
        "n_shared_cells_total": int(len(all_spread)),
        "offset_spread_median_dB": round(float(np.median(all_spread)), 2) if all_spread else None,
        "offset_spread_p90_dB": round(float(np.percentile(all_spread, 90)), 2) if all_spread else None,
        "offset_spread_max_dB": round(float(np.max(all_spread)), 2) if all_spread else None,
        "recon_method": RECON_METHOD,
        "dense_multiphone": {
            "recon_raw_rmse": round(raw_rmse, 3) if raw_rmse else None,
            "recon_cal_rmse": round(cal_rmse, 3) if cal_rmse else None,
            "improvement_pct": round(100 * (raw_rmse - cal_rmse) / raw_rmse, 2)
            if (raw_rmse and cal_rmse) else None,
        },
        "single_device_per_rp": {  # random single-device crowdsensing
            "recon_raw_rmse": round(sd_raw_rmse, 3) if sd_raw_rmse else None,
            "recon_cal_rmse": round(sd_cal_rmse, 3) if sd_cal_rmse else None,
            "improvement_pct": _imp(sd_raw_rmse, sd_cal_rmse),
        },
        "contiguous_device_zones": {  # realistic crowdsensing (each worker covers a zone)
            "recon_raw_rmse": round(cz_raw_rmse, 3) if cz_raw_rmse else None,
            "recon_cal_rmse": round(cz_cal_rmse, 3) if cz_cal_rmse else None,
            "improvement_pct": _imp(cz_raw_rmse, cz_cal_rmse),
        },
        "per_floor": per_floor,
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figures ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(all_spread, bins=30, color="#8e44ad", alpha=0.8)
    ax[0].axvline(out["offset_spread_median_dB"], color="k", ls="--",
                  label=f"median {out['offset_spread_median_dB']} dB")
    ax[0].set_xlabel("cross-phone RSS disagreement at shared RP (dB)")
    ax[0].set_ylabel("# shared cells"); ax[0].legend()
    ax[0].set_title(f"A4 device heterogeneity (n={len(all_spread)} shared cells)")
    labels, raws, cals = [], [], []
    if raw_rmse and cal_rmse:
        labels.append("dense\nmulti-phone"); raws.append(raw_rmse); cals.append(cal_rmse)
    if sd_raw_rmse and sd_cal_rmse:
        labels.append("1 device/RP\n(random)"); raws.append(sd_raw_rmse); cals.append(sd_cal_rmse)
    if cz_raw_rmse and cz_cal_rmse:
        labels.append("contiguous\nzones\n(realistic)"); raws.append(cz_raw_rmse); cals.append(cz_cal_rmse)
    if labels:
        x = np.arange(len(labels))
        ax[1].bar(x - 0.2, raws, 0.4, label="raw pool", color="#7f8c8d")
        ax[1].bar(x + 0.2, cals, 0.4, label="device-calibrated", color="#27ae60")
        ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
        ax[1].set_ylabel("held-out-RP RMSE (dB)"); ax[1].legend()
        ax[1].set_title("A4 calibration impact by coverage regime")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "device_realdata.png", dpi=120); plt.close(fig)

    print(f"\n[A4] device offset spread: median={out['offset_spread_median_dB']} "
          f"p90={out['offset_spread_p90_dB']} max={out['offset_spread_max_dB']} dB")
    dm, sd, cz = out["dense_multiphone"], out["single_device_per_rp"], out["contiguous_device_zones"]
    print(f"[A4] dense multi-phone:      raw={dm['recon_raw_rmse']} cal={dm['recon_cal_rmse']} "
          f"({dm['improvement_pct']}% better)")
    print(f"[A4] 1 device/RP (random):   raw={sd['recon_raw_rmse']} cal={sd['recon_cal_rmse']} "
          f"({sd['improvement_pct']}% better)")
    print(f"[A4] contiguous zones (real):raw={cz['recon_raw_rmse']} cal={cz['recon_cal_rmse']} "
          f"({cz['improvement_pct']}% better)")
    print(f"[A4] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Whole-dataset inventory + cross-site validation of the two core COMPASS
assumptions (device heterogeneity, trajectory autocorrelation) on REAL data,
across every site and floor that is present. Writes a summary JSON + an
overview figure.

Run on the cluster:
  bash scripts/submit.sh 1g.18gb python scripts/summarize_unicellular.py
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
    SITES,
    UniCellularPaths,
    best_shared_rp,
    device_offsets_at_rp,
    load_floor,
    mobile_sequences,
    top_transmitters,
    valid_measurements,
)

REPO = Path(__file__).resolve().parents[1]


def autocorr_lag1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    var = (x * x).mean()
    if var <= 0 or len(x) < 3:
        return float("nan")
    return float((x[:-1] * x[1:]).mean() / var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "unicellular_summary.json"))
    ap.add_argument("--fig", default=str(REPO / "figures" / "realdata" / "dataset_overview.png"))
    args = ap.parse_args()
    paths = UniCellularPaths()

    summary = {"sites": {}}
    dev_spreads, traj_acs = [], []
    for site in SITES:
        floors = paths.list_floors(site, "stationary")
        mob_floors = paths.list_floors(site, "mobile")
        if not floors and not mob_floors:
            continue
        site_info = {"stationary_floors": floors, "mobile_floors": mob_floors, "floors": {}}
        tot_stat = tot_mob = 0
        phones_all, tx_all = set(), set()
        for fl in floors:
            try:
                df = load_floor(site, "stationary", fl)
            except Exception as e:  # noqa: BLE001
                site_info["floors"][fl] = {"error": str(e)[:80]}
                continue
            v = valid_measurements(df)
            tot_stat += len(df)
            phones_all |= set(df["phoneName"].unique())
            tx_all |= set(v["transmitter_id"].unique())
            rec = {"stationary_rows": int(len(df)), "valid_rows": int(len(v)),
                   "n_real_tx": int(v["transmitter_id"].nunique())}
            # device heterogeneity on the best-shared cell/RP
            txs = top_transmitters(df, 1)
            if txs:
                rp = best_shared_rp(df, txs[0])
                offs = device_offsets_at_rp(df, txs[0], rp)
                if len(offs) >= 2:
                    sp = float(offs.max() - offs.min())
                    rec["device_spread_db"] = round(sp, 1)
                    rec["n_phones_shared"] = int(len(offs))
                    dev_spreads.append(sp)
            site_info["floors"][fl] = rec
        # one mobile autocorr per site (best cell, longest walk)
        if mob_floors:
            try:
                mdf = load_floor(site, "mobile", mob_floors[0])
                mtx = top_transmitters(mdf, 1)
                if mtx:
                    seqs = mobile_sequences(mdf, mtx[0])
                    if seqs:
                        ph = max(seqs, key=lambda p: len(seqs[p]))
                        ac = autocorr_lag1(seqs[ph][RSS_COL].to_numpy())
                        site_info["mobile_lag1_autocorr"] = round(ac, 3)
                        if np.isfinite(ac):
                            traj_acs.append(ac)
                for fl in mob_floors:
                    tot_mob += int(len(load_floor(site, "mobile", fl)))
            except Exception as e:  # noqa: BLE001
                site_info["mobile_error"] = str(e)[:80]
        site_info["total_stationary_rows"] = tot_stat
        site_info["total_mobile_rows"] = tot_mob
        site_info["n_phones"] = len(phones_all)
        site_info["n_real_transmitters"] = len(tx_all)
        summary["sites"][site] = site_info
        print(f"[{site:14s}] stat_floors={len(floors):2d} mob_floors={len(mob_floors):2d} "
              f"stat_rows={tot_stat:>8d} mob_rows={tot_mob:>7d} phones={len(phones_all)} "
              f"tx={len(tx_all)} lag1ac={site_info.get('mobile_lag1_autocorr','-')}")

    summary["aggregate"] = {
        "median_device_spread_db": round(float(np.median(dev_spreads)), 2) if dev_spreads else None,
        "device_spread_range_db": [round(min(dev_spreads), 1), round(max(dev_spreads), 1)] if dev_spreads else None,
        "median_mobile_lag1_autocorr": round(float(np.median(traj_acs)), 3) if traj_acs else None,
        "n_floors_measured": len(dev_spreads),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(summary, indent=2))

    # --- overview figure ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    if dev_spreads:
        ax[0].hist(dev_spreads, bins=20, color="#8e44ad", alpha=0.8)
        ax[0].axvline(np.median(dev_spreads), color="k", ls="--",
                      label=f"median {np.median(dev_spreads):.1f} dB")
        ax[0].set_xlabel("device RSS spread per floor (dB)")
        ax[0].set_ylabel("# floors")
        ax[0].set_title(f"Device heterogeneity across {len(dev_spreads)} floors (real)")
        ax[0].legend()
    if traj_acs:
        ax[1].hist(traj_acs, bins=15, color="#c0392b", alpha=0.8)
        ax[1].axvline(np.median(traj_acs), color="k", ls="--",
                      label=f"median {np.median(traj_acs):.2f}")
        ax[1].set_xlabel("mobile-walk lag-1 autocorr (per site)")
        ax[1].set_ylabel("# sites")
        ax[1].set_title("Real-trajectory order structure")
        ax[1].legend()
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120)
    plt.close(fig)

    agg = summary["aggregate"]
    print(f"\n[summary] device spread: median {agg['median_device_spread_db']} dB, "
          f"range {agg['device_spread_range_db']} (across {agg['n_floors_measured']} floors)")
    print(f"[summary] mobile lag-1 autocorr: median {agg['median_mobile_lag1_autocorr']} (real walks)")
    print(f"[summary] results -> {args.results}\n[summary] figure -> {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Feasibility inspection for cross-device fingerprint LOCALIZATION on UniCellular.
Reports, per usable site/floor: #phones, #RPs with coords, #fingerprints, cells/fp,
per-phone RP coverage and fingerprint counts, RP overlap across phones (needed for
leave-one-phone-out), and the per-(rp,cell) RSS spread across phones (the device
offset magnitude that a localizer must survive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from compass.realdata import unicellular as U

M_PER_PX = 0.032
SITES = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1), ("ec_parking", 2)]

RSS = getattr(U, "RSS_COL", "transmitter_rss")
TX = getattr(U, "TX_COL", "transmitter_id")
PHONE = getattr(U, "PHONE_COL", "phoneName")


def main():
    for site, fl in SITES:
        try:
            df = U.load_floor(site, "stationary", fl, usecols=U.RECON_COLS)
        except Exception as e:  # noqa: BLE001
            print(f"=== {site} f{fl}: LOAD FAIL {e}")
            continue
        v = U.valid_measurements(df)
        v = v[v["rpNumber"] >= 1]
        coords = U.load_rp_coords(site, fl)
        if not coords:
            print(f"=== {site} f{fl}: no coords")
            continue
        v = v[v["rpNumber"].astype(int).isin(coords.keys())]
        phones = sorted(v[PHONE].astype(str).unique())
        rps = sorted(set(int(r) for r in v["rpNumber"].unique()))
        fps = v.groupby([PHONE, "rpNumber", "scanNumber"])
        cells_per_fp = fps[TX].nunique()
        print(f"\n=== {site} f{fl}: phones={len(phones)} RPs(coords)={len(rps)} "
              f"fingerprints={fps.ngroups} cells/fp med={cells_per_fp.median():.0f} "
              f"[{cells_per_fp.min()}-{cells_per_fp.max()}] uniq_cells={v[TX].nunique()}")
        # per-phone coverage
        for p in phones:
            vp = v[v[PHONE].astype(str) == p]
            prps = sorted(set(int(r) for r in vp["rpNumber"].unique()))
            nfp = vp.groupby(["rpNumber", "scanNumber"]).ngroups
            print(f"   {p:>16s}: RPs={len(prps):3d}  fingerprints={nfp:4d}")
        # RP overlap: RPs visited by >=2 phones (needed for LOPO map)
        rp_phone_ct = v.groupby("rpNumber")[PHONE].nunique()
        print(f"   RPs seen by >=2 phones: {(rp_phone_ct >= 2).sum()}/{len(rps)}; "
              f"by all {len(phones)}: {(rp_phone_ct >= len(phones)).sum()}")
        # device offset magnitude: per (rp,cell) spread across phones
        pc = v.groupby(["rpNumber", TX, PHONE])[RSS].mean().reset_index()
        spread = pc.groupby(["rpNumber", TX])[RSS].agg(["max", "min", "nunique"])
        spread = spread[spread["nunique"] >= 2]
        if len(spread):
            d = (spread["max"] - spread["min"])
            print(f"   per-(rp,cell) cross-phone spread dB: median={d.median():.1f} "
                  f"p90={d.quantile(0.9):.1f} max={d.max():.1f} (n={len(spread)} shared)")
    print("\n[inspect] done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Phase A1 — feasibility survey of the UniCellular real data for TX-agnostic
reconstruction. Determines what is VALID to attempt before designing experiments:
per (site, floor): #reference points, spatial extent (m, at 0.032 m/px), #cells,
how many cells are observed by ENOUGH RPs to support held-out-RP cross-validation,
scan density, and mobile-walk lengths.

  bash scripts/submit.sh 1g.18gb python scripts/realdata_survey.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compass.realdata.unicellular import (
    RECON_COLS,
    SITES,
    UniCellularPaths,
    load_floor,
    load_rp_coords,
    valid_measurements,
)

REPO = Path(__file__).resolve().parents[1]
M_PER_PX = 0.032


def main() -> int:
    paths = UniCellularPaths()
    survey = {"m_per_px": M_PER_PX, "sites": {}}
    total_viable8 = 0

    for site in SITES:
        floors = paths.list_floors(site, "stationary")
        if not floors:
            continue
        site_info = {"floors": {}, "has_floor_plan": paths.floor_plan(site, floors[0]).exists()}
        for fl in floors:
            try:
                df = load_floor(site, "stationary", fl, usecols=RECON_COLS)
            except Exception as e:  # noqa: BLE001
                site_info["floors"][fl] = {"error": str(e)[:60]}
                continue
            v = valid_measurements(df)
            v = v[v["rpNumber"] >= 1]
            if v.empty:
                continue
            rp_pos = v.groupby("rpNumber")[["x", "y"]].first()
            n_rp = int(rp_pos.shape[0])
            ext_x = (rp_pos["x"].max() - rp_pos["x"].min()) * M_PER_PX
            ext_y = (rp_pos["y"].max() - rp_pos["y"].min()) * M_PER_PX
            per_cell_rp = v.groupby("transmitter_id")["rpNumber"].nunique()
            n_cells = int(per_cell_rp.shape[0])
            cells_ge8 = int((per_cell_rp >= 8).sum())
            cells_ge12 = int((per_cell_rp >= 12).sum())
            scans_per_rp_cell = v.groupby(["rpNumber", "transmitter_id"]).size().median()
            coords = load_rp_coords(site, fl)
            site_info["floors"][fl] = {
                "n_rp": n_rp, "n_cells": n_cells,
                "cells_obs_by_ge8_rp": cells_ge8, "cells_obs_by_ge12_rp": cells_ge12,
                "extent_m": [round(float(ext_x), 1), round(float(ext_y), 1)],
                "median_scans_per_rp_cell": int(scans_per_rp_cell),
                "n_phones": int(v["phoneName"].nunique()),
                "has_coords_json": bool(coords),
            }
            total_viable8 += cells_ge8
        # mobile walk lengths (one floor sample)
        mob_floors = paths.list_floors(site, "mobile")
        if mob_floors:
            try:
                m = load_floor(site, "mobile", mob_floors[0], usecols=RECON_COLS)
                mv = valid_measurements(m)
                wl = mv.groupby("phoneName")["scanNumber"].nunique()
                site_info["mobile_sample"] = {"floor": mob_floors[0], "n_phones": int(len(wl)),
                                              "median_walk_len": int(wl.median()), "n_mobile_floors": len(mob_floors)}
            except Exception:  # noqa: BLE001
                pass
        survey["sites"][site] = site_info
        f0 = min(site_info["floors"])
        fi = site_info["floors"][f0]
        print(f"[{site:14s}] floors={len(site_info['floors'])} plan={site_info['has_floor_plan']} | "
              f"floor{f0}: RP={fi.get('n_rp','?')} cells={fi.get('n_cells','?')} "
              f"cells>=8RP={fi.get('cells_obs_by_ge8_rp','?')} extent={fi.get('extent_m','?')}m")

    survey["total_viable_cells_ge8rp"] = total_viable8
    out = REPO / "results" / "realdata_survey.json"
    out.write_text(json.dumps(survey, indent=2))
    print(f"\n[survey] total (site,floor,cell) combos with >=8 RPs (viable for held-out-RP CV): {total_viable8}")
    # feasibility verdict
    n_cmuq_ec = sum(
        f.get("cells_obs_by_ge8_rp", 0)
        for s in ("cmuq", "ec_parking") if s in survey["sites"]
        for f in survey["sites"][s]["floors"].values()
    )
    print(f"[survey] of which CMUQ+EC (real floor plans): {n_cmuq_ec}")
    print(f"[survey] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

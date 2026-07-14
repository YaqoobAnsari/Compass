#!/usr/bin/env python
"""
Phase 1 verification: prove the RadioMapSeer loader is CORRECT, and produce
figures + a results JSON so we can visually and numerically affirm it.

Checks, per sample (the two facts v1 got wrong are checks #3 and #4):
  1. shapes/dtypes/ranges (radio_map in [-1,1], dBm in [-186,-47])
  2. street fraction is plausible (0.3-0.97)
  3. POLARITY: mean gain on streets >> mean gain in buildings (>10 dB margin)
  4. TX FLIP: gain-argmax pixel == computed TX pixel (row=H-1-y, col=x)

Run:  python scripts/verify_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compass.data import (
    RadioMapSeerDataset,
    list_map_ids,
    split_map_ids,
)
from compass.utils.visualize import plot_dataset_sample

REPO = Path(__file__).resolve().parents[1]


def check_sample(sample: dict) -> dict:
    building = sample["building_map"].numpy()[0]
    free = sample["free_mask"].numpy()[0].astype(bool)
    radio = sample["radio_map"].numpy()[0]
    dbm = sample["radio_map_dbm"].numpy()[0]
    tx_row, tx_col = (int(v) for v in sample["tx_rowcol"].numpy())

    street_db = float(dbm[free].mean())
    bldg_db = float(dbm[~free].mean()) if (~free).any() else float("nan")
    arg_row, arg_col = (int(v) for v in np.unravel_index(int(np.argmax(dbm)), dbm.shape))
    tx_err = max(abs(arg_row - tx_row), abs(arg_col - tx_col))

    checks = {
        "shape_ok": building.shape == (256, 256),
        "building_binary": set(np.unique(building).tolist()).issubset({0.0, 1.0}),
        "radio_in_unit": bool(radio.min() >= -1.0001 and radio.max() <= 1.0001),
        "dbm_in_range": bool(dbm.min() >= -186.01 and dbm.max() <= -46.99),
        "street_fraction_ok": 0.30 <= float(free.mean()) <= 0.97,
        "polarity_street_stronger": (street_db - bldg_db) > 10.0,   # check #3
        # check #4: TX coord transform. DPM peaks exactly at the TX; IRT2/IRT4
        # ray-tracing can put the single strongest pixel a few px off the antenna
        # via a reflection hotspot, so allow 3px (the transform itself is exact).
        "tx_at_gain_peak": tx_err <= 3,
    }
    return {
        "map_id": int(sample["map_id"]),
        "tx_id": int(sample["tx_id"]),
        "street_fraction": float(free.mean()),
        "mean_dbm_street": street_db,
        "mean_dbm_building": bldg_db,
        "dbm_margin": street_db - bldg_db,
        "tx_pixel": [tx_row, tx_col],
        "gain_argmax_pixel": [arg_row, arg_col],
        "tx_pixel_error_px": tx_err,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--n-maps", type=int, default=4, help="distinct test maps to sample")
    ap.add_argument("--tx-per-map", type=int, default=3)
    ap.add_argument("--fig-dir", default=str(REPO / "figures" / "dataset"))
    ap.add_argument("--results", default=str(REPO / "results" / "dataset_verification.json"))
    args = ap.parse_args()

    splits = split_map_ids(list_map_ids(args.root))
    test_maps = splits["test"][: args.n_maps]
    ds = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)
    print(f"[verify] variant={args.variant}  test maps used={test_maps}  "
          f"|train/val/test maps|={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    fig_dir = Path(args.fig_dir)
    results = []
    for mi, map_id in enumerate(test_maps):
        for tx in range(args.tx_per_map):
            idx = ds.samples.index((map_id, tx))
            sample = ds[idx]
            r = check_sample(sample)
            results.append(r)
            tag = "OK " if r["passed"] else "FAIL"
            print(f"  [{tag}] map {map_id} tx {tx}: "
                  f"street {r['mean_dbm_street']:.1f} vs bldg {r['mean_dbm_building']:.1f} dBm "
                  f"(Δ{r['dbm_margin']:.1f}) | TX err {r['tx_pixel_error_px']}px")
            if tx == 0:  # one figure per map
                plot_dataset_sample(
                    sample, fig_dir / f"map{map_id}_tx{tx}_{args.variant}.png",
                    title=f"RadioMapSeer map {map_id} tx {tx} ({args.variant})",
                )

    n_pass = sum(r["passed"] for r in results)
    summary = {
        "variant": args.variant,
        "n_samples": len(results),
        "n_passed": n_pass,
        "all_passed": n_pass == len(results),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_dbm_margin": float(np.mean([r["dbm_margin"] for r in results])),
        "max_tx_pixel_error_px": int(max(r["tx_pixel_error_px"] for r in results)),
        "samples": results,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n[verify] {n_pass}/{len(results)} samples passed all checks")
    print(f"[verify] mean dBm street-vs-building margin = {summary['mean_dbm_margin']:.1f} dB")
    print(f"[verify] max TX-pixel error = {summary['max_tx_pixel_error_px']} px (0 = perfect coord transform)")
    print(f"[verify] figures -> {fig_dir}")
    print(f"[verify] results -> {out}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

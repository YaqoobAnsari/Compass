#!/usr/bin/env python
"""
Phase 2 verification: prove trajectories are generated ON STREETS (the v1 bug),
that they therefore sample REAL signal (not the noise floor), and that the
realistic correlated-noise model behaves as intended. Saves figures + JSON.

Checks per sample:
  1. >=99% of trajectory points lie on the street/free mask (v1 walked in buildings)
  2. clean RSS sampled along walks is well above the building noise floor (real signal)
  3. coverage fraction is small & plausible (0.1%-5%)
  4. ordered sequence features are finite with monotonically increasing time
  5. shadowing field std ~ sigma; its decorrelation length grows with decorr_m

Run:  python scripts/verify_trajectories.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids, street_mask
from compass.data.conditioning import make_conditioning, sample_clean_dbm
from compass.data.noise import MeasurementNoise, correlated_shadowing_field, measured_decorrelation_px
from compass.data.trajectory import TrajectorySampler
from compass.utils.visualize import plot_shadowing_field, plot_trajectory_sample

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--n-maps", type=int, default=3)
    ap.add_argument("--tx-per-map", type=int, default=2)
    ap.add_argument("--k", type=int, default=3, help="trajectories per sample")
    ap.add_argument("--n-points", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", default=str(REPO / "figures" / "trajectories"))
    ap.add_argument("--results", default=str(REPO / "results" / "trajectory_verification.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    ds = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)
    sampler = TrajectorySampler(seed=args.seed)
    noise = MeasurementNoise()
    fig_dir = Path(args.fig_dir)

    results = []
    for mi, map_id in enumerate(test_maps):
        building_raw = ds.load_building(map_id)          # {0,255}
        free = street_mask(building_raw)
        bldg_floor = None
        for tx in range(args.tx_per_map):
            idx = ds.samples.index((map_id, tx))
            sample = ds[idx]
            dbm = sample["radio_map_dbm"].numpy()[0]
            tx_rc = tuple(int(v) for v in sample["tx_rowcol"].numpy())
            if bldg_floor is None:
                bldg_floor = float(dbm[~free].mean())

            cond = make_conditioning(building_raw, dbm, tx_rc, sampler, noise, rng,
                                     k=args.k, n_points=args.n_points)

            on_street, clean_means, seq_ok = [], [], True
            for tr, seq in zip(cond.trajectories, cond.sequences):
                ri = np.clip(np.round(tr.rows).astype(int), 0, 255)
                ci = np.clip(np.round(tr.cols).astype(int), 0, 255)
                on_street.append(float(free[ri, ci].mean()))
                clean_means.append(float(sample_clean_dbm(dbm, tr.rows, tr.cols).mean()))
                seq_ok &= bool(np.isfinite(seq).all() and np.all(np.diff(tr.t) >= -1e-6))
            on_street_frac = float(np.mean(on_street))
            clean_mean = float(np.mean(clean_means))

            checks = {
                "on_street": on_street_frac >= 0.99,                 # check #1
                "real_signal": (clean_mean - bldg_floor) > 5.0,       # check #2
                "coverage_plausible": 0.001 <= cond.coverage_fraction <= 0.05,  # check #3
                "sequence_ok": seq_ok,                                # check #4
            }
            r = {
                "map_id": map_id, "tx_id": tx,
                "on_street_frac": on_street_frac,
                "clean_mean_dbm": clean_mean, "building_floor_dbm": bldg_floor,
                "signal_above_floor_db": clean_mean - bldg_floor,
                "coverage_fraction": cond.coverage_fraction,
                "seq_shape": list(cond.sequences[0].shape),
                "checks": checks, "passed": all(checks.values()),
            }
            results.append(r)
            tag = "OK " if r["passed"] else "FAIL"
            print(f"  [{tag}] map {map_id} tx {tx}: on-street {on_street_frac*100:.1f}% | "
                  f"signal {clean_mean:.1f} dBm ({r['signal_above_floor_db']:.1f} above floor) | "
                  f"cov {cond.coverage_fraction*100:.2f}%")
            if tx == 0:
                plot_trajectory_sample(sample, cond, fig_dir / f"map{map_id}_traj.png",
                                       title=f"map {map_id}: on-street trajectories + realistic noise")

    # --- noise model behaviour: std fidelity + decorrelation vs decorr_m ---
    noise_check = {}
    for L in (5.0, 20.0, 40.0):
        f = correlated_shadowing_field(sigma_db=6.0, decorr_m=L, rng=rng)
        noise_check[f"decorr_m={L}"] = {
            "field_std_db": float(f.std()),
            "measured_decorr_px": measured_decorrelation_px(f),
        }
    f40 = correlated_shadowing_field(sigma_db=6.0, decorr_m=40.0, rng=rng)
    plot_shadowing_field(f40, fig_dir / "shadowing_field_decorr40m.png",
                         decorr_px=measured_decorrelation_px(f40),
                         title="correlated shadow fading (sigma=6 dB, decorr=40 m)")
    decs = [noise_check[k]["measured_decorr_px"] for k in noise_check]
    std_ok = all(0.7 <= noise_check[k]["field_std_db"] / 6.0 <= 1.3 for k in noise_check)
    decorr_monotonic = decs[0] < decs[1] < decs[2]

    n_pass = sum(r["passed"] for r in results)
    summary = {
        "variant": args.variant, "n_samples": len(results), "n_passed": n_pass,
        "all_passed": n_pass == len(results) and std_ok and decorr_monotonic,
        "mean_signal_above_floor_db": float(np.mean([r["signal_above_floor_db"] for r in results])),
        "mean_on_street_frac": float(np.mean([r["on_street_frac"] for r in results])),
        "noise_model": noise_check, "noise_std_ok": std_ok, "decorr_monotonic": decorr_monotonic,
        "samples": results,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n[verify] {n_pass}/{len(results)} trajectory samples passed")
    print(f"[verify] mean signal above floor = {summary['mean_signal_above_floor_db']:.1f} dB (v1 sampled AT the floor)")
    print(f"[verify] noise std fidelity ok={std_ok} | decorrelation grows with decorr_m={decorr_monotonic}")
    print(f"[verify] decorrelation (px) by decorr_m: " +
          ", ".join(f"{k}->{v['measured_decorr_px']:.0f}" for k, v in noise_check.items()))
    print(f"[verify] figures -> {fig_dir}\n[verify] results -> {out}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

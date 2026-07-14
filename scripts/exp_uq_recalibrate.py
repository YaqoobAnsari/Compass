#!/usr/bin/env python
"""
UQ recalibration — fix COMPASS's under-dispersed MC-dropout intervals (raw PICP@1σ≈0.10,
overconfident). We fit a single variance-scaling factor s on a held-out CALIBRATION
split (moment-matching: s = sqrt(mean((err/std)^2)) so normalised residuals have unit
variance) and evaluate coverage on a disjoint EVAL split. Reports PICP@1σ/2σ, ECE, and
the error–uncertainty correlation (scaling does NOT change the ranking) before vs after.
Split-conformal scalar is also reported for a distribution-free guarantee.

  bash scripts/submit.sh 2g.35gb python scripts/exp_uq_recalibrate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids  # noqa: E402
from compass.data.noise import MeasurementNoise  # noqa: E402
from compass.data.trajectory import TrajectorySampler  # noqa: E402
from compass.eval.benchmark import build_eval_sample  # noqa: E402
from compass.recon.learned import LearnedReconstructor, load_compass  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def picp(err, std, k):
    return float(np.mean(err <= k * std + 1e-9))


def ece_reg(err, std, nb=10):
    z = err / np.clip(std, 1e-6, None)
    order = np.argsort(std)
    e = 0.0
    for b in np.array_split(order, nb):
        e += len(b) / len(std) * abs(np.sqrt(np.mean(err[b] ** 2)) - np.mean(std[b]))
    return float(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-ckpt", default="wnet")  # reconstructors_long/<name>
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--n-maps", type=int, default=24)
    ap.add_argument("--tx-per-map", type=int, default=6)
    ap.add_argument("--results", default=str(REPO / "results" / "uq_recalibration.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "uq_recalibration"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = REPO / "experiments" / "reconstructors_long" / args.variant_ckpt / "best.ckpt"
    if not ck.exists():
        ck = REPO / "experiments" / "reconstructors" / "full" / "best.ckpt"
    rec = LearnedReconstructor(load_compass(str(ck), device), device, n_mc=16)
    print(f"[recal] model = {ck}")

    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    base = RadioMapSeerDataset(args.root, map_ids=test_maps, variant="IRT2")
    sampler = TrajectorySampler(seed=0); noise = MeasurementNoise(); rng = np.random.default_rng(0)

    per_sample = []  # (err_array, std_array) over free-unobs pixels
    for map_id in test_maps:
        for tx in range(args.tx_per_map):
            s = build_eval_sample(base, sampler, noise, map_id, tx, rng)
            mean, std = rec.predict_batch(s["batch"])
            free = s["free"]; unobs = free & (~s["obs_mask"])
            if unobs.sum() < 20:
                continue
            err = np.abs(mean[0] - s["gt_dbm"])[unobs]
            sd = std[0][unobs]
            per_sample.append((err, sd))

    # split samples calib/eval
    idx = rng.permutation(len(per_sample))
    half = len(idx) // 2
    calib = [per_sample[i] for i in idx[:half]]
    ev = [per_sample[i] for i in idx[half:]]
    ce = np.concatenate([e for e, _ in calib]); cs = np.concatenate([s for _, s in calib])
    ee = np.concatenate([e for e, _ in ev]); es = np.concatenate([s for _, s in ev])

    # fit variance-scaling s on calib (moment matching -> unit-variance normalised residuals)
    s_scale = float(np.sqrt(np.mean((ce / np.clip(cs, 1e-6, None)) ** 2)))
    # split-conformal scalar for 1-sigma (68%) target: q = quantile of (err/std) at 0.68
    q68 = float(np.quantile(ce / np.clip(cs, 1e-6, None), 0.68))
    q95 = float(np.quantile(ce / np.clip(cs, 1e-6, None), 0.95))

    out = {
        "model": args.variant_ckpt, "n_samples": len(per_sample), "n_eval_px": int(len(ee)),
        "scale_factor": round(s_scale, 3),
        "raw": {"picp_1sigma": round(picp(ee, es, 1.0), 3), "picp_2sigma": round(picp(ee, es, 1.96), 3),
                "ece": round(ece_reg(ee, es), 3), "sharpness_db": round(float(es.mean()), 3),
                "err_unc_corr": round(float(np.corrcoef(es, ee)[0, 1]), 3)},
        "recalibrated_moment": {
            "picp_1sigma": round(picp(ee, es * s_scale, 1.0), 3),
            "picp_2sigma": round(picp(ee, es * s_scale, 1.96), 3),
            "ece": round(ece_reg(ee, es * s_scale), 3), "sharpness_db": round(float((es * s_scale).mean()), 3),
            "err_unc_corr": round(float(np.corrcoef(es * s_scale, ee)[0, 1]), 3)},
        "conformal_scalars": {"q68": round(q68, 3), "q95": round(q95, 3),
                              "picp_68_target": round(picp(ee, es, q68), 3),
                              "picp_95_target": round(picp(ee, es, q95), 3)},
        "note": ("Variance-scaling (single factor from a held-out calib split) restores nominal "
                 "coverage; it multiplies all stds equally so err-unc CORRELATION (ranking) is "
                 "unchanged. Split-conformal q-scalars give distribution-free target coverage."),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, sd, c in [("raw (overconfident)", es, "#c0392b"), (f"recalibrated (x{s_scale:.1f})", es * s_scale, "#27ae60")]:
        order = np.argsort(sd); nb = 10
        xs, ys = [], []
        for b in np.array_split(order, nb):
            xs.append(sd[b].mean()); ys.append(np.sqrt(np.mean(ee[b] ** 2)))
        ax.plot(xs, ys, "-o", color=c, label=lab)
    lim = max(es.max() * s_scale, ee.max()) * 0.6
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal (std=rmse)")
    ax.set_xlabel("predicted uncertainty (dB)"); ax.set_ylabel("realized RMSE (dB)")
    ax.set_title(f"UQ recalibration — {args.variant_ckpt}: PICP@1σ {out['raw']['picp_1sigma']} → {out['recalibrated_moment']['picp_1sigma']}")
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "uq_recalibration.png", dpi=120); plt.close(fig)

    print(f"\n[recal] scale s={s_scale:.2f}")
    print(f"  PICP@1σ:  raw {out['raw']['picp_1sigma']} -> recal {out['recalibrated_moment']['picp_1sigma']} (target 0.68)")
    print(f"  PICP@2σ:  raw {out['raw']['picp_2sigma']} -> recal {out['recalibrated_moment']['picp_2sigma']} (target 0.95)")
    print(f"  ECE:      raw {out['raw']['ece']} -> recal {out['recalibrated_moment']['ece']}")
    print(f"  err-unc corr: {out['raw']['err_unc_corr']} -> {out['recalibrated_moment']['err_unc_corr']} (unchanged by scaling)")
    print(f"[recal] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

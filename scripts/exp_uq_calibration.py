#!/usr/bin/env python
"""
UQ calibration — COMPASS's headline differentiator over RadioUNet/PMNet.

On the synthetic test set, quantify uncertainty QUALITY for the methods that produce
uncertainty (COMPASS via MC-dropout, GP via posterior variance) and show that the
strong point baselines (RadioUNet, PMNet, SparseUNet) provide NONE. Metrics:
error–uncertainty correlation, ECE, PICP@1σ/2σ (coverage), sharpness (mean std),
plus a reliability curve (binned predicted-std vs realized RMSE).

  bash scripts/submit.sh 2g.35gb python scripts/exp_uq_calibration.py
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
from compass.eval.stats import bootstrap_ci  # noqa: E402
from compass.models.baselines.wrapper import BaselineReconstructor, load_baseline  # noqa: E402
from compass.recon import GPReconstructor  # noqa: E402
from compass.recon.learned import LearnedReconstructor, load_compass  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def uq_stats(pred, gt, std, free, obs):
    """Per-sample UQ stats over free-unobserved pixels."""
    unobs = free & (~obs)
    if unobs.sum() < 10:
        return None
    err = np.abs(pred - gt)[unobs]
    s = std[unobs]
    if s.std() < 1e-6:  # degenerate/no uncertainty (point estimator)
        return {"corr": float("nan"), "picp1": float("nan"), "picp2": float("nan"),
                "sharp": float(np.mean(s)), "has_uq": False,
                "bins": None}
    corr = float(np.corrcoef(s, err)[0, 1])
    picp1 = float(np.mean(err <= s + 1e-6))
    picp2 = float(np.mean(err <= 1.96 * s + 1e-6))
    # reliability bins: predicted std quantile -> realized rmse
    order = np.argsort(s)
    nb = 8
    bins = []
    for b in np.array_split(order, nb):
        bins.append((float(np.mean(s[b])), float(np.sqrt(np.mean(err[b] ** 2)))))
    return {"corr": corr, "picp1": picp1, "picp2": picp2, "sharp": float(np.mean(s)),
            "has_uq": True, "bins": bins}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--n-maps", type=int, default=20)
    ap.add_argument("--tx-per-map", type=int, default=6)
    ap.add_argument("--results", default=str(REPO / "results" / "uq_calibration.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "uq_calibration"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    base = RadioMapSeerDataset(args.root, map_ids=test_maps, variant="IRT2")

    methods = {}
    ckf = REPO / "experiments" / "reconstructors_long" / "full" / "best.ckpt"
    if not ckf.exists():
        ckf = REPO / "experiments" / "reconstructors" / "full" / "best.ckpt"
    methods["COMPASS-full"] = ("learned", LearnedReconstructor(load_compass(str(ckf), device), device, n_mc=12))
    methods["GP(Kriging)"] = ("classical", GPReconstructor(length_scale=10.0, noise_std=2.0))
    for arch in ("radiounet", "pmnet"):
        d = REPO / "experiments" / "dl_baselines" / arch / "best.ckpt"
        if d.exists():
            methods[arch] = ("learned", BaselineReconstructor(load_baseline(str(d), device), device, n_mc=8))

    sampler = TrajectorySampler(seed=0)
    noise = MeasurementNoise()
    rng = np.random.default_rng(0)
    agg = {m: {"corr": [], "picp1": [], "picp2": [], "sharp": [], "bins": [], "has_uq": True} for m in methods}

    for map_id in test_maps:
        for tx in range(args.tx_per_map):
            sample = build_eval_sample(base, sampler, noise, map_id, tx, rng)
            gt, free = sample["gt_dbm"], sample["free"]
            obs = sample["obs_mask"]
            for m, (kind, rec) in methods.items():
                if kind == "learned":
                    mean, std = rec.predict_batch(sample["batch"])
                    mean, std = mean[0], std[0]
                else:
                    r = rec.reconstruct(sample["obs"], free)
                    mean, std = r.mean, r.std
                st = uq_stats(mean, gt, std, free, obs)
                if st is None:
                    continue
                agg[m]["has_uq"] = agg[m]["has_uq"] and st["has_uq"]
                for k in ("corr", "picp1", "picp2", "sharp"):
                    if not np.isnan(st[k]):
                        agg[m][k].append(st[k])
                if st["bins"]:
                    agg[m]["bins"].append(st["bins"])

    out = {"n_samples": len(test_maps) * args.tx_per_map, "methods": {}}
    for m, a in agg.items():
        out["methods"][m] = {
            "has_uncertainty": bool(a["has_uq"] and len(a["corr"]) > 0),
            "err_unc_corr": bootstrap_ci(a["corr"]) if a["corr"] else None,
            "picp_1sigma": round(float(np.mean(a["picp1"])), 3) if a["picp1"] else None,
            "picp_2sigma": round(float(np.mean(a["picp2"])), 3) if a["picp2"] else None,
            "sharpness_db": round(float(np.mean(a["sharp"])), 3) if a["sharp"] else None,
        }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # reliability curves (methods with UQ)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, a in agg.items():
        if not a["bins"]:
            continue
        arr = np.array(a["bins"])  # (n_samples, nb, 2)
        xs = arr[:, :, 0].mean(0); ys = arr[:, :, 1].mean(0)
        ax.plot(xs, ys, "-o", label=f"{m}")
    lim = 30
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal (std=rmse)")
    ax.set_xlabel("predicted uncertainty (dB)"); ax.set_ylabel("realized RMSE (dB)")
    ax.set_title(f"UQ reliability — COMPASS vs GP (point baselines have NO UQ)\n{out['n_samples']} samples")
    ax.legend()
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "uq_reliability.png", dpi=120); plt.close(fig)

    print(f"\n=== UQ calibration ({out['n_samples']} samples) ===")
    print(f"{'method':16s} {'hasUQ':6s} {'corr':>6s} {'PICP1σ':>7s} {'PICP2σ':>7s} {'sharp':>6s}")
    for m, v in out["methods"].items():
        c = v["err_unc_corr"]["mean"] if v["err_unc_corr"] else float("nan")
        print(f"{m:16s} {str(v['has_uncertainty']):6s} {c:6.3f} "
              f"{str(v['picp_1sigma']):>7s} {str(v['picp_2sigma']):>7s} {str(v['sharpness_db']):>6s}")
    print(f"[UQ] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

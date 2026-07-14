#!/usr/bin/env python
"""
Phase B — close the last SYNTHETIC questions:

  B1. COMPLEXITY STRATIFICATION: bin test maps by building density (terciles) and
      compare learned (full) vs best tuned classical (RBF) vs no_building per bin.
      Does the ~15 dB learned advantage HOLD on the hardest / densest layouts?

  B2. RCL PLAUSIBILITY: compare `full` vs `no_rcl` on PHYSICAL-plausibility metrics
      (behind-building / NLoS error + ray-monotonicity violation rate), stratified
      by complexity. The Ray-Consistency Loss should show value HERE (not in RMSE).

Writes results/phaseB.json + figures/phaseB/*.png. Deterministic point estimates.

  bash scripts/submit.sh 2g.35gb python scripts/exp_phaseB.py
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
from compass.data.conventions import signed_unit_to_dbm  # noqa: E402
from compass.data.noise import MeasurementNoise  # noqa: E402
from compass.data.trajectory import TrajectorySampler  # noqa: E402
from compass.eval import all_metrics, ray_monotonicity_violation  # noqa: E402
from compass.eval.benchmark import build_eval_sample, score_classical  # noqa: E402
from compass.eval.stats import bootstrap_ci, paired_wilcoxon  # noqa: E402
from compass.recon import GPReconstructor, RBFReconstructor  # noqa: E402
from compass.recon.learned import load_compass  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "experiments" / "reconstructors"


@torch.no_grad()
def predict_dbm(model, batch, device):
    model.eval()
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return signed_unit_to_dbm(model(b).cpu().numpy()[0, 0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--n-maps", type=int, default=24)
    ap.add_argument("--tx-per-map", type=int, default=8)
    ap.add_argument("--results", default=str(REPO / "results" / "phaseB.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "phaseB"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_compass(str(CKPT / "full" / "best.ckpt"), device)
    no_rcl = load_compass(str(CKPT / "no_rcl" / "best.ckpt"), device)
    no_bld = load_compass(str(CKPT / "no_building" / "best.ckpt"), device)
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    base = RadioMapSeerDataset(args.root, map_ids=test_maps)
    sampler = TrajectorySampler(seed=0)
    noise = MeasurementNoise()
    rng = np.random.default_rng(0)
    classical = {"RBF(mq)": lambda: RBFReconstructor(kernel="multiquadric", epsilon=20.0),
                 "GP": lambda: GPReconstructor(length_scale=10.0)}

    per = []  # per-sample record
    for map_id in test_maps:
        building_frac = float((base.load_building(map_id) == 255).mean())
        for tx in range(args.tx_per_map):
            s = build_eval_sample(base, sampler, noise, map_id, tx, rng, k=3, n_points=100)
            gt, free, om, tx_rc, b01 = s["gt_dbm"], s["free"], s["obs_mask"], s["tx_rc"], s["building01"]
            rec = {"building_frac": building_frac}
            # learned models
            for name, mdl in (("full", full), ("no_rcl", no_rcl), ("no_building", no_bld)):
                pred = predict_dbm(mdl, s["batch"], device)
                m = all_metrics(pred, gt, np.zeros_like(pred), b01, free, om, tx_rc)
                rec[f"{name}_rmse"] = m["rmse_free_unobs"]
                rec[f"{name}_nlos"] = m["rmse_nlos"]
                rec[f"{name}_monoviol"] = ray_monotonicity_violation(pred, tx_rc, free)
            # classical
            for name, mk in classical.items():
                m = score_classical(mk(), s)
                rec[f"{name}_rmse"] = m["rmse_free_unobs"]
                rec[f"{name}_nlos"] = m["rmse_nlos"]
            per.append(rec)
        print(f"  map {map_id} (bldg {building_frac*100:.0f}%) done")

    per = np.array(per)  # list of dicts -> keep as list
    fr = np.array([r["building_frac"] for r in per])
    q1, q2 = np.quantile(fr, [1 / 3, 2 / 3])
    bins = {"low (sparse)": fr <= q1, "mid": (fr > q1) & (fr <= q2), "high (dense)": fr > q2}

    def col(name):
        return np.array([r.get(name, np.nan) for r in per])

    out = {"n_samples": len(per), "building_frac_terciles": [round(float(q1), 3), round(float(q2), 3)],
           "B1_complexity": {}, "B2_rcl": {}}

    # B1: RMSE per method per complexity bin
    methods = ["full", "no_building", "RBF(mq)", "GP"]
    for bname, mask in bins.items():
        out["B1_complexity"][bname] = {"n": int(mask.sum())}
        for meth in methods:
            out["B1_complexity"][bname][meth] = bootstrap_ci(col(f"{meth}_rmse")[mask])

    # B2: full vs no_rcl plausibility (overall + per bin)
    for scope, mask in [("overall", np.ones(len(per), bool)), *bins.items()]:
        out["B2_rcl"][scope] = {
            "full_nlos": bootstrap_ci(col("full_nlos")[mask]),
            "no_rcl_nlos": bootstrap_ci(col("no_rcl_nlos")[mask]),
            "full_monoviol": bootstrap_ci(col("full_monoviol")[mask]),
            "no_rcl_monoviol": bootstrap_ci(col("no_rcl_monoviol")[mask]),
            "wilcoxon_monoviol_full_vs_norcl": paired_wilcoxon(col("full_monoviol")[mask], col("no_rcl_monoviol")[mask]),
            "wilcoxon_nlos_full_vs_norcl": paired_wilcoxon(col("full_nlos")[mask], col("no_rcl_nlos")[mask]),
        }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figures ---
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    bnames = list(bins)
    # B1
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(bnames)); w = 0.2
    cols = {"full": "#1e7d4f", "no_building": "#e67e22", "RBF(mq)": "#7f8c8d", "GP": "#95a5a6"}
    for i, meth in enumerate(methods):
        ys = [out["B1_complexity"][b][meth]["mean"] for b in bnames]
        ax.bar(x + (i - 1.5) * w, ys, w, label=meth, color=cols[meth])
    ax.set_xticks(x); ax.set_xticklabels(bnames)
    ax.set_ylabel("free-unobserved RMSE (dBm)"); ax.set_xlabel("scene complexity (building density)")
    ax.set_title("B1: learned vs classical by scene complexity"); ax.legend()
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "B1_complexity.png", dpi=120); plt.close(fig)
    # B2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, (metric, title) in enumerate([("nlos", "behind-building (NLoS) RMSE (dB)"),
                                          ("monoviol", "ray-monotonicity violation rate")]):
        fu = [out["B2_rcl"][b][f"full_{metric}"]["mean"] for b in bnames]
        nr = [out["B2_rcl"][b][f"no_rcl_{metric}"]["mean"] for b in bnames]
        ax[j].bar(x - 0.2, fu, 0.4, label="full (RCL on)", color="#1e7d4f")
        ax[j].bar(x + 0.2, nr, 0.4, label="no_rcl (RCL off)", color="#c0392b")
        ax[j].set_xticks(x); ax[j].set_xticklabels(bnames); ax[j].set_title(title); ax[j].legend()
    fig.suptitle("B2: does the Ray-Consistency Loss improve physical plausibility?")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "B2_rcl_plausibility.png", dpi=120); plt.close(fig)

    print("\n[B1] free-unobs RMSE by complexity (full vs RBF):")
    for b in bnames:
        f = out["B1_complexity"][b]["full"]["mean"]; r = out["B1_complexity"][b]["RBF(mq)"]["mean"]
        print(f"    {b:14s} full={f:.2f}  RBF={r:.2f}  (margin {r-f:.1f} dB, n={out['B1_complexity'][b]['n']})")
    ov = out["B2_rcl"]["overall"]
    print(f"[B2] RCL plausibility (overall): mono-viol full={ov['full_monoviol']['mean']:.3f} vs "
          f"no_rcl={ov['no_rcl_monoviol']['mean']:.3f} (p={ov['wilcoxon_monoviol_full_vs_norcl']['p_value']:.1e}); "
          f"NLoS full={ov['full_nlos']['mean']:.2f} vs no_rcl={ov['no_rcl_nlos']['mean']:.2f}")
    print(f"[phaseB] -> {args.results}, figs {args.figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Complexity-stratified benchmark — does RadioUNet DEGRADE on hard/dense scenes while
COMPASS-WNet holds? Stratify the synthetic test set into easy/medium/hard by BUILDING
DENSITY (occlusion), then score EVERY method (all DL baselines + classical) per stratum.

If RadioUNet's error grows faster than COMPASS-WNet's from easy->hard (especially in
NLoS/behind-building regions), we have a genuine ACCURACY regime where COMPASS wins —
much stronger than "we provide uncertainty and it does not".

  bash scripts/submit.sh 2g.35gb python scripts/exp_complexity_benchmark.py
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
from compass.eval.benchmark import build_eval_sample, score_classical, score_learned  # noqa: E402
from compass.eval.stats import bootstrap_ci, paired_wilcoxon  # noqa: E402
from compass.models.baselines.wrapper import BaselineReconstructor, load_baseline  # noqa: E402
from compass.recon import GPReconstructor, IDWReconstructor, RBFReconstructor  # noqa: E402
from compass.recon.learned import LearnedReconstructor, load_compass  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-maps", type=int, default=35)
    ap.add_argument("--tx-per-map", type=int, default=6)
    ap.add_argument("--skip", default="", help="comma-sep methods to skip (e.g. rmdm)")
    ap.add_argument("--results", default=str(REPO / "results" / "complexity_benchmark.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "complexity_benchmark"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_maps = split_map_ids(list_map_ids(str(REPO / "data" / "raw")))["test"][: args.n_maps]
    base = RadioMapSeerDataset(str(REPO / "data" / "raw"), map_ids=test_maps, variant="IRT2")

    classical = {"RBF(mq)": RBFReconstructor(kernel="multiquadric", epsilon=20.0),
                 "IDW": IDWReconstructor(power=1.0),
                 "GP": GPReconstructor(length_scale=10.0, noise_std=2.0)}
    learned = {}
    for nm in ("full", "wnet", "wnet_occ", "no_tx"):
        ck = REPO / "experiments" / "reconstructors_long" / nm / "best.ckpt"
        if ck.exists():
            learned[f"COMPASS-{nm}"] = LearnedReconstructor(load_compass(str(ck), device), device, n_mc=6)
    skip = set(x.strip() for x in args.skip.split(",") if x.strip())
    for d in sorted((REPO / "experiments" / "dl_baselines").glob("*/best.ckpt")):
        arch = d.parent.name
        if arch in skip:
            continue
        try:
            nmc = 4 if arch == "rmdm" else 6
            learned[arch] = BaselineReconstructor(load_baseline(str(d), device), device, n_mc=nmc)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {arch}: {e}")

    all_methods = list(classical) + list(learned)
    rows = {m: [] for m in all_methods}  # each: (complexity, rmse_unobs, rmse_nlos)
    sampler = TrajectorySampler(seed=0); noise = MeasurementNoise(); rng = np.random.default_rng(0)

    n = 0
    for map_id in test_maps:
        for tx in range(args.tx_per_map):
            sample = build_eval_sample(base, sampler, noise, map_id, tx, rng)
            complexity = float(sample["building01"].mean())  # building-pixel fraction = density
            for nm, rec in classical.items():
                s = score_classical(rec, sample)
                rows[nm].append((complexity, s["rmse_free_unobs"], s.get("rmse_nlos", np.nan)))
            for nm, lr in learned.items():
                s = score_learned(lr, sample)
                rows[nm].append((complexity, s["rmse_free_unobs"], s.get("rmse_nlos", np.nan)))
            n += 1
        print(f"  scored map {map_id} ({n} samples)")

    # stratify by complexity terciles
    comp_all = np.array([r[0] for r in rows[all_methods[0]]])
    t1, t2 = np.quantile(comp_all, [1 / 3, 2 / 3])
    strata = {"easy": comp_all <= t1, "medium": (comp_all > t1) & (comp_all <= t2), "hard": comp_all > t2}

    out = {"n_samples": n, "density_terciles": [round(float(t1), 4), round(float(t2), 4)],
           "per_method": {}}
    for m in all_methods:
        arr = np.array(rows[m])  # (N,3)
        md = {"overall_rmse": round(float(np.sqrt(np.nanmean(arr[:, 1] ** 2))), 3), "strata": {}}
        for s, mask in strata.items():
            r = arr[mask]
            md["strata"][s] = {
                "rmse": round(float(np.sqrt(np.nanmean(r[:, 1] ** 2))), 3),
                "rmse_nlos": round(float(np.sqrt(np.nanmean(r[:, 2] ** 2))), 3),
                "n": int(mask.sum())}
        md["degradation_easy_to_hard"] = round(md["strata"]["hard"]["rmse"] - md["strata"]["easy"]["rmse"], 3)
        out["per_method"][m] = md

    # head-to-head: COMPASS-wnet vs RadioUNet per stratum (paired Wilcoxon on rmse)
    if "COMPASS-wnet" in rows and "radiounet" in rows:
        wn = np.array(rows["COMPASS-wnet"]); ru = np.array(rows["radiounet"])
        out["wnet_vs_radiounet"] = {}
        for s, mask in strata.items():
            w = paired_wilcoxon(wn[mask, 1], ru[mask, 1])
            out["wnet_vs_radiounet"][s] = {
                "wnet_rmse": round(float(np.sqrt(np.nanmean(wn[mask, 1] ** 2))), 3),
                "radiounet_rmse": round(float(np.sqrt(np.nanmean(ru[mask, 1] ** 2))), 3),
                "median_diff": w.get("median_diff"), "p_value": w.get("p_value")}
        # hardest quartile (top 25% density) — where the crossover should be sharpest
        hq = comp_all > np.quantile(comp_all, 0.75)
        w = paired_wilcoxon(wn[hq, 1], ru[hq, 1])
        out["wnet_vs_radiounet"]["hardest_quartile"] = {
            "wnet_rmse": round(float(np.sqrt(np.nanmean(wn[hq, 1] ** 2))), 3),
            "radiounet_rmse": round(float(np.sqrt(np.nanmean(ru[hq, 1] ** 2))), 3),
            "median_diff": w.get("median_diff"), "p_value": w.get("p_value"), "n": int(hq.sum())}

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # figure: RMSE vs difficulty for the key methods
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    key = [m for m in ("COMPASS-wnet", "radiounet", "COMPASS-full", "pmnet", "radiogan", "RBF(mq)") if m in out["per_method"]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    xs = [0, 1, 2]
    for m in key:
        ys = [out["per_method"][m]["strata"][s]["rmse"] for s in ("easy", "medium", "hard")]
        lw = 3 if "COMPASS-wnet" in m or m == "radiounet" else 1.5
        ax[0].plot(xs, ys, "-o", label=m, linewidth=lw)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(["easy", "medium", "hard"])
    ax[0].set_ylabel("free-unobs RMSE (dB)"); ax[0].set_title("Overall RMSE vs scene density")
    ax[0].legend(fontsize=8)
    for m in key:
        ys = [out["per_method"][m]["strata"][s]["rmse_nlos"] for s in ("easy", "medium", "hard")]
        lw = 3 if "COMPASS-wnet" in m or m == "radiounet" else 1.5
        ax[1].plot(xs, ys, "-o", label=m, linewidth=lw)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(["easy", "medium", "hard"])
    ax[1].set_ylabel("NLoS RMSE (dB)"); ax[1].set_title("Behind-building (NLoS) RMSE vs density"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "complexity_benchmark.png", dpi=120); plt.close(fig)

    print("\n=== RMSE by scene density (easy / medium / hard) ===")
    for m in all_methods:
        s = out["per_method"][m]["strata"]
        print(f"  {m:16s} {s['easy']['rmse']:6.2f} {s['medium']['rmse']:6.2f} {s['hard']['rmse']:6.2f}  "
              f"(Δ easy→hard {out['per_method'][m]['degradation_easy_to_hard']:+.2f})")
    if "wnet_vs_radiounet" in out:
        print("\n=== COMPASS-WNet vs RadioUNet per stratum ===")
        for s, v in out["wnet_vs_radiounet"].items():
            print(f"  {s:8s} wnet={v['wnet_rmse']:.2f} radiounet={v['radiounet_rmse']:.2f} "
                  f"(diff {v['median_diff']}, p={v['p_value']})")
    print(f"[cx-bench] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

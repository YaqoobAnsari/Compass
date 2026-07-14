#!/usr/bin/env python
"""
Comprehensive, FAIR classical-baseline benchmark (no strawman):
  1. TUNE each tunable family (GP length-scale, IDW power, RBF kernel/smoothing)
     on a validation subset -> best config per family.
  2. EVALUATE the full panel (best configs + NN + natural-neighbour) on a large
     held-out test set with bootstrap 95% CIs across ALL metrics.
  3. Repeat at MULTIPLE coverage levels (k trajectories) to show whether classical
     interpolation is weak everywhere or only at extreme sparsity.

This is what lets us state — or refute — "classical interpolation is weak for
trajectory-sparse radio maps" with evidence.

  bash scripts/submit.sh 2g.35gb python scripts/benchmark_classical.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids  # noqa: E402
from compass.data.noise import MeasurementNoise  # noqa: E402
from compass.data.trajectory import TrajectorySampler  # noqa: E402
from compass.eval.benchmark import build_eval_sample, score_classical  # noqa: E402
from compass.eval.stats import bootstrap_ci  # noqa: E402
from compass.recon import (  # noqa: E402
    GeodesicNearestReconstructor,
    GPReconstructor,
    IDWReconstructor,
    NaturalNeighborReconstructor,
    NearestNeighborReconstructor,
    OrdinaryKrigingReconstructor,
    RBFReconstructor,
)

REPO = Path(__file__).resolve().parents[1]
KEY = "rmse_free_unobs"


def candidate_configs():
    cfgs = {}
    # GP / Simple Kriging: length-scale sweep
    for ls in (10.0, 20.0, 30.0, 50.0, 80.0):
        cfgs[f"GP(ls={ls:.0f})"] = lambda ls=ls: GPReconstructor(length_scale=ls, noise_std=2.0)
    # Ordinary Kriging: spherical variogram range x nugget sweep
    for vr in (20.0, 40.0, 80.0):
        for ng in (1.0, 4.0):
            cfgs[f"OK(range={vr:.0f},nug={ng:.0f})"] = lambda vr=vr, ng=ng: OrdinaryKrigingReconstructor(vrange=vr, nugget=ng)
    # IDW: power sweep
    for p in (1.0, 2.0, 3.0):
        cfgs[f"IDW(p={p:.0f})"] = lambda p=p: IDWReconstructor(power=p)
    # RBF multiquadric: epsilon sweep (scale)
    for eps in (10.0, 20.0, 50.0):
        cfgs[f"RBF(mq,eps={eps:.0f})"] = lambda eps=eps: RBFReconstructor(kernel="multiquadric", epsilon=eps, smoothing=1.0)
    # RBF thin-plate-spline: smoothing sweep
    for sm in (0.0, 1.0, 10.0):
        cfgs[f"RBF(tps,sm={sm:.0f})"] = lambda sm=sm: RBFReconstructor(kernel="thin_plate_spline", smoothing=sm)
    cfgs["RBF(linear)"] = lambda: RBFReconstructor(kernel="linear", smoothing=1.0)
    cfgs["NearestNeighbor"] = lambda: NearestNeighborReconstructor()
    cfgs["NaturalNeighbor"] = lambda: NaturalNeighborReconstructor()
    cfgs["GeodesicNearest"] = lambda: GeodesicNearestReconstructor()  # building-aware
    return cfgs


def eval_method(make_rec, base, sampler, noise, maps, tx_list, k, n_points, seed):
    rng = np.random.default_rng(seed)
    metrics = []
    for m in maps:
        for tx in tx_list:
            sample = build_eval_sample(base, sampler, noise, m, tx, rng, k=k, n_points=n_points)
            metrics.append(score_classical(make_rec(), sample))
    return metrics


def family(name):  # group configs by family for "best per family"
    return name.split("(")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--tune-maps", type=int, default=6)
    ap.add_argument("--tune-tx", type=int, default=4)
    ap.add_argument("--test-maps", type=int, default=24)
    ap.add_argument("--test-tx", type=int, default=8)
    ap.add_argument("--coverages", type=int, nargs="+", default=[1, 3, 5], help="k trajectories")
    ap.add_argument("--n-points", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "results" / "benchmark_classical.json"))
    ap.add_argument("--fig", default=str(REPO / "figures" / "benchmark" / "classical_coverage.png"))
    args = ap.parse_args()

    splits = split_map_ids(list_map_ids(args.root))
    val_maps = splits["val"][: args.tune_maps]
    test_maps = splits["test"][: args.test_maps]
    base_val = RadioMapSeerDataset(args.root, map_ids=val_maps, variant=args.variant)
    base_test = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)
    sampler = TrajectorySampler(seed=0)
    noise = MeasurementNoise()
    cfgs = candidate_configs()

    # --- 1. tune on validation at default coverage (k=3) ---
    print("[tune] selecting best config per family on validation set...")
    val_scores = {}
    for name, mk in cfgs.items():
        ms = eval_method(mk, base_val, sampler, noise, val_maps, list(range(args.tune_tx)),
                         k=3, n_points=args.n_points, seed=1)
        val_scores[name] = float(np.nanmean([m[KEY] for m in ms]))
        print(f"    {name:22s} val {KEY} = {val_scores[name]:.2f} dBm")
    best = {}
    for name in cfgs:
        f = family(name)
        if f not in best or val_scores[name] < val_scores[best[f]]:
            best[f] = name
    chosen = {best[f]: cfgs[best[f]] for f in best}
    print(f"[tune] chosen per family: {list(chosen)}")

    # --- 2 & 3. evaluate chosen methods on TEST across coverage levels ---
    out = {"chosen": list(chosen), "val_scores": {k: round(v, 3) for k, v in val_scores.items()},
           "n_test_samples": args.test_maps * args.test_tx, "coverage_levels_k": args.coverages,
           "results": {}}
    for k in args.coverages:
        out["results"][f"k={k}"] = {}
        print(f"\n[test] coverage k={k} trajectories ...")
        for name, mk in chosen.items():
            ms = eval_method(mk, base_test, sampler, noise, test_maps, list(range(args.test_tx)),
                             k=k, n_points=args.n_points, seed=2)
            arr = np.array([m[KEY] for m in ms])
            ci = bootstrap_ci(arr)
            agg = {KEY: ci}
            for mk2 in ms[0]:
                if mk2 == KEY:
                    continue  # don't overwrite the CI dict with a plain float
                agg[mk2] = round(float(np.nanmean([m[mk2] for m in ms])), 4)
            out["results"][f"k={k}"][name] = agg
            cov = np.mean([m["n_free_unobs"] for m in ms])
            print(f"    {name:22s} {KEY}={ci['mean']:.2f} [{ci['lo']:.2f},{ci['hi']:.2f}] "
                  f"ssim={agg['ssim_free']:.3f} rmse_obs={agg['rmse_free_obs']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # --- figure: RMSE vs coverage per method ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    methods = list(chosen)
    for name in methods:
        xs = args.coverages
        ys = [out["results"][f"k={k}"][name][KEY]["mean"] for k in xs]
        lo = [out["results"][f"k={k}"][name][KEY]["lo"] for k in xs]
        hi = [out["results"][f"k={k}"][name][KEY]["hi"] for k in xs]
        ax.plot(xs, ys, "-o", ms=5, label=name)
        ax.fill_between(xs, lo, hi, alpha=0.12)
    ax.set_xlabel("coverage (# trajectories, k)")
    ax.set_ylabel("free-unobserved RMSE (dBm)")
    ax.set_title(f"Tuned classical baselines vs coverage ({out['n_test_samples']} test samples, 95% CI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120)
    plt.close(fig)

    # best classical at default coverage
    k_def = 3 if 3 in args.coverages else args.coverages[0]
    ranked = sorted(out["results"][f"k={k_def}"].items(), key=lambda kv: kv[1][KEY]["mean"])
    print(f"\n[classical] BEST tuned classical at k={k_def}: "
          f"{ranked[0][0]} = {ranked[0][1][KEY]['mean']:.2f} dBm")
    print(f"[classical] -> {args.out}, figure -> {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

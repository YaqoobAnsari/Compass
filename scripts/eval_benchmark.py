#!/usr/bin/env python
"""
Publication benchmark: score every classical baseline + every trained learned
variant on the held-out test maps, with full metrics + bootstrap CI + Wilcoxon.
Writes summary JSON, a CSV table, a LaTeX table, and a main-results bar figure
(with CIs). Learned checkpoints are auto-discovered under experiments/reconstructors/.

  bash scripts/submit.sh 2g.35gb python scripts/eval_benchmark.py
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
from compass.eval import run_benchmark  # noqa: E402
from compass.recon import (  # noqa: E402
    GeodesicNearestReconstructor,
    GPReconstructor,
    IDWReconstructor,
    NaturalNeighborReconstructor,
    NearestNeighborReconstructor,
    OrdinaryKrigingReconstructor,
    RBFReconstructor,
)
from compass.recon.learned import LearnedReconstructor, load_compass  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--n-maps", type=int, default=20)
    ap.add_argument("--tx-per-map", type=int, default=6)
    ap.add_argument("--ckpt-dir", default=str(REPO / "experiments" / "reconstructors"))
    ap.add_argument("--out", default=str(REPO / "results" / "benchmark.json"))
    ap.add_argument("--fig", default=str(REPO / "figures" / "benchmark" / "main_results.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    tx_list = list(range(args.tx_per_map))
    base = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)

    # full classical panel (best-tuned configs from benchmark_classical.py)
    classical = {
        "GP(Kriging)": GPReconstructor(length_scale=10.0, noise_std=2.0),
        "OrdinaryKriging": OrdinaryKrigingReconstructor(vrange=40.0, nugget=1.0),
        "IDW(p=1)": IDWReconstructor(power=1.0),
        "RBF(mq)": RBFReconstructor(kernel="multiquadric", epsilon=20.0),
        "NearestNeighbor": NearestNeighborReconstructor(),
        "NaturalNeighbor": NaturalNeighborReconstructor(),
        "GeodesicNearest": GeodesicNearestReconstructor(),
    }
    learned = {}
    ck_dir = Path(args.ckpt_dir)
    for d in sorted(ck_dir.glob("*/best.ckpt")) if ck_dir.exists() else []:
        name = d.parent.name
        try:
            learned[name] = LearnedReconstructor(load_compass(str(d), device), device, n_mc=8)
            print(f"[bench] loaded learned: {name}")
        except Exception as e:  # noqa: BLE001
            print(f"[bench] skip {name}: {e}")

    print(f"[bench] {len(test_maps)} maps x {len(tx_list)} tx | "
          f"classical={list(classical)} learned={list(learned)} | device={device}")
    summary, _ = run_benchmark(classical, learned, base, test_maps, tx_list)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    # --- CSV + LaTeX table ---
    rows = sorted(summary.items(), key=lambda kv: kv[1]["rmse_free_unobs"]["mean"])
    cols = ["rmse_free_unobs", "rmse_free", "ssim_free", "unc_err_corr", "rmse_nlos"]
    csv = ["method," + ",".join(cols) + ",ci_lo,ci_hi"]
    tex = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"Method & Unobs.RMSE & Free RMSE & SSIM & UncCorr & NLoS RMSE \\", r"\midrule"]
    for name, v in rows:
        ci = v["rmse_free_unobs"]
        csv.append(f"{name}," + ",".join(f"{v[c]:.3f}" if isinstance(v[c], float) else str(v[c]) for c in cols)
                   + f",{ci['lo']:.3f},{ci['hi']:.3f}")
        tex.append(f"{name} & {ci['mean']:.2f} & {v['rmse_free']:.2f} & {v['ssim_free']:.3f} "
                   f"& {v['unc_err_corr']:.3f} & {v['rmse_nlos']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    Path(args.out).with_suffix(".csv").write_text("\n".join(csv))
    Path(args.out).with_suffix(".tex").write_text("\n".join(tex))

    # --- bar figure with CIs ---
    names = [n for n, _ in rows]
    means = [summary[n]["rmse_free_unobs"]["mean"] for n in names]
    los = [summary[n]["rmse_free_unobs"]["lo"] for n in names]
    his = [summary[n]["rmse_free_unobs"]["hi"] for n in names]
    err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    colors = ["#c0392b" if n == "full" else ("#e67e22" if n in learned else "#7f8c8d") for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(names)), means, yerr=err, color=colors, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("free-unobserved RMSE (dBm) — lower is better")
    ax.set_title(f"COMPASS benchmark ({len(test_maps)*len(tx_list)} test samples, 95% CI)")
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120)
    plt.close(fig)

    print("\n[bench] ranked free-unobs RMSE (dBm):")
    for name, v in rows:
        ci = v["rmse_free_unobs"]
        w = v.get("wilcoxon_vs_full", {})
        p = w.get("p_value")
        pstr = f"  p_vs_full={p:.1e}" if isinstance(p, (int, float)) and p == p else ""
        print(f"  {name:16s} {ci['mean']:6.2f} [{ci['lo']:.2f},{ci['hi']:.2f}] "
              f"ssim {v['ssim_free']:.3f} unc-corr {v['unc_err_corr']:.3f}" + pstr)
    print(f"[bench] -> {args.out} (+ .csv, .tex), figure -> {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

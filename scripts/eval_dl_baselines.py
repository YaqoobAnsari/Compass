#!/usr/bin/env python
"""
Phase G benchmark: score external DL radio-map baselines (RadioUNet, PMNet,
SparseUNet, RMDM) against COMPASS and the classical panel on the IDENTICAL held-out
test set, with bootstrap CI + paired Wilcoxon. Every method sees the same
trajectories; DL baselines were trained on the same data/budget as COMPASS.

  bash scripts/submit.sh 2g.35gb python scripts/eval_dl_baselines.py
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
from compass.models.baselines.wrapper import BaselineReconstructor, load_baseline  # noqa: E402
from compass.recon import (  # noqa: E402
    GPReconstructor,
    IDWReconstructor,
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
    ap.add_argument("--out", default=str(REPO / "results" / "dl_baselines.json"))
    ap.add_argument("--fig", default=str(REPO / "figures" / "dl_baselines" / "dl_vs_compass.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    tx_list = list(range(args.tx_per_map))
    base = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)

    # representative classical anchors (full panel already in the main benchmark)
    classical = {
        "RBF(mq)": RBFReconstructor(kernel="multiquadric", epsilon=20.0),
        "GP(Kriging)": GPReconstructor(length_scale=10.0, noise_std=2.0),
        "IDW(p=1)": IDWReconstructor(power=1.0),
        "OrdinaryKriging": OrdinaryKrigingReconstructor(vrange=40.0, nugget=1.0),
    }

    learned = {}
    # COMPASS variants for context — prefer the matched-budget long run; fall back to 40-ep.
    for name in ("full", "wnet", "wnet_occ", "no_building", "no_tx"):
        ck = REPO / "experiments" / "reconstructors_long" / name / "best.ckpt"
        if not ck.exists():
            ck = REPO / "experiments" / "reconstructors" / name / "best.ckpt"
        if ck.exists():
            learned[f"COMPASS-{name}"] = LearnedReconstructor(load_compass(str(ck), device), device, n_mc=8)
            print(f"[bench] loaded COMPASS-{name} from {ck.parent.parent.name}")
    # external DL baselines
    dl_dir = REPO / "experiments" / "dl_baselines"
    for d in sorted(dl_dir.glob("*/best.ckpt")) if dl_dir.exists() else []:
        arch = d.parent.name
        try:
            learned[arch] = BaselineReconstructor(load_baseline(str(d), device), device, n_mc=8)
            print(f"[bench] loaded DL baseline: {arch}")
        except Exception as e:  # noqa: BLE001
            print(f"[bench] skip {arch}: {e}")

    # run_benchmark uses 'full' as the Wilcoxon reference; alias COMPASS-full to it
    if "COMPASS-full" in learned:
        learned["full"] = learned.pop("COMPASS-full")

    print(f"[bench] {len(test_maps)} maps x {len(tx_list)} tx | classical={list(classical)} "
          f"learned={list(learned)} | device={device}")
    summary, _ = run_benchmark(classical, learned, base, test_maps, tx_list)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    # --- ranking print + figure ---
    rows = sorted(summary.items(), key=lambda kv: kv[1]["rmse_free_unobs"]["mean"])
    print("\n=== DL-baseline benchmark (free-unobs RMSE dB, sorted) ===")
    for name, v in rows:
        ci = v["rmse_free_unobs"]
        w = v.get("wilcoxon_vs_full", {})
        p = w.get("p_value")
        pstr = f"p={p:.1e}" if isinstance(p, (int, float)) and p == p else "ref"
        print(f"  {name:20s} {ci['mean']:6.2f} [{ci['lo']:.2f},{ci['hi']:.2f}]  SSIM={v.get('ssim_free')}  {pstr}")

    names = [n for n, _ in rows]
    means = [v["rmse_free_unobs"]["mean"] for _, v in rows]
    los = [v["rmse_free_unobs"]["lo"] for _, v in rows]
    his = [v["rmse_free_unobs"]["hi"] for _, v in rows]
    err = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
    colors = ["#2ecc71" if n == "full" else ("#3498db" if n in ("radiounet", "pmnet", "sparse_unet", "rmdm")
              else "#95a5a6") for n in names]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), means, yerr=err, capsize=3, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("free-unobserved RMSE (dB)")
    ax.set_title(f"Phase G: COMPASS (green) vs external DL baselines (blue) vs classical (grey) — {len(test_maps)*len(tx_list)} samples")
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.fig, dpi=120); plt.close(fig)
    print(f"\n[bench] -> {args.out}, fig {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

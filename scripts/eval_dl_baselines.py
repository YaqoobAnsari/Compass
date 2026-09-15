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
from compass.eval.stats import paired_wilcoxon  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
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
    for name in ("full", "wnet", "wnet_base", "wnet_occ", "wnet_occ_meas",
                 "unet_occ", "no_building", "no_tx"):
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
    summary, per_method = run_benchmark(classical, learned, base, test_maps, tx_list)

    # run_benchmark only tests every method against the 'full' reference, which leaves
    # the occlusion-by-cascade cells untested against each other. Add the paired
    # comparisons the 2x2 actually turns on, over the identical test samples.
    key = "rmse_free_unobs"
    cells = {
        "unet_noocc": "full",
        "unet_occ": "COMPASS-unet_occ",
        "wnet_noocc": "COMPASS-wnet_base",
        "wnet_occ": "COMPASS-wnet_occ",
        "wnet_occ_meas": "COMPASS-wnet_occ_meas",
    }
    arr = {k: np.array([r[key] for r in per_method[v]])
           for k, v in cells.items() if v in per_method}
    pairs = [
        ("unet_noocc", "unet_occ"),          # does occlusion help the plain U-Net
        ("wnet_noocc", "wnet_occ"),          # does occlusion help the cascade
        ("unet_noocc", "wnet_noocc"),        # does the cascade help without occlusion
        ("unet_occ", "wnet_occ"),            # does the cascade help with occlusion
        ("wnet_noocc", "wnet_occ_meas"),     # is TX-free occlusion better than none
        ("wnet_occ", "wnet_occ_meas"),       # is TX-free occlusion worse than TX-anchored
        ("unet_occ", "wnet_occ_meas"),
    ]
    pw = {}
    for a, b in pairs:
        if a in arr and b in arr:
            pw[f"{a}_vs_{b}"] = {
                "mean_a": round(float(arr[a].mean()), 4),
                "mean_b": round(float(arr[b].mean()), 4),
                "delta_b_minus_a": round(float(arr[b].mean() - arr[a].mean()), 4),
                **paired_wilcoxon(arr[b], arr[a]),
            }
    summary["_pairwise_occlusion_cascade"] = pw

    # the 2x2 interaction, with a paired bootstrap over the same samples
    if all(k in arr for k in ("unet_noocc", "unet_occ", "wnet_noocc", "wnet_occ")):
        inter = ((arr["wnet_occ"] - arr["wnet_noocc"]) - (arr["unet_occ"] - arr["unet_noocc"]))
        rng_b = np.random.default_rng(0)
        boot = np.array([rng_b.choice(inter, len(inter), replace=True).mean() for _ in range(2000)])
        summary["_interaction_occlusion_x_cascade"] = {
            "definition": "(wnet_occ - wnet_noocc) - (unet_occ - unet_noocc); >0 means subadditive",
            "mean": round(float(inter.mean()), 4),
            "lo": round(float(np.percentile(boot, 2.5)), 4),
            "hi": round(float(np.percentile(boot, 97.5)), 4),
            "n": int(len(inter)),
            "wilcoxon_vs_zero": {"p_value": float(wilcoxon(inter).pvalue) if np.any(inter != 0) else float("nan")},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    # --- ranking print + figure ---
    rows = sorted((kv for kv in summary.items() if not kv[0].startswith("_")),
                  key=lambda kv: kv[1]["rmse_free_unobs"]["mean"])
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
    if pw:
        print("\n=== occlusion x cascade, paired Wilcoxon (same 320 samples) ===")
        for k, v in pw.items():
            print(f"  {k:34s} {v['mean_a']:6.2f} -> {v['mean_b']:6.2f}  "
                  f"delta={v['delta_b_minus_a']:+6.2f}  p={v['p_value']:.2e}")
    if "_interaction_occlusion_x_cascade" in summary:
        it = summary["_interaction_occlusion_x_cascade"]
        print(f"\n  INTERACTION {it['mean']:+.3f} dB [{it['lo']:+.3f}, {it['hi']:+.3f}] "
              f"p={it['wilcoxon_vs_zero']['p_value']:.2e}  (>0 = subadditive)")
    print(f"\n[bench] -> {args.out}, fig {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

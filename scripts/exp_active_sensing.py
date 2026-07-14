#!/usr/bin/env python
"""
HEADLINE system-novelty experiment: uncertainty-guided active crowdsensing vs
baselines at MATCHED measurement budget, on RadioMapSeer (dense GT).

For each test map, run every acquisition strategy (max_variance / space_filling /
coverage / random) through the closed loop, averaging RMSE-vs-budget curves across
maps. Saves the curves figure + a results JSON. This is the framework the learned
physics-aware reconstructor will later plug into for the full win.

  bash scripts/submit.sh 2g.35gb python scripts/exp_active_sensing.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compass.active import STRATEGIES, run_active_loop  # noqa: E402
from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids  # noqa: E402
from compass.recon import GPReconstructor  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--n-maps", type=int, default=6)
    ap.add_argument("--n-rounds", type=int, default=14)
    ap.add_argument("--points-per-walk", type=int, default=60)
    ap.add_argument("--length-scale", type=float, default=25.0)
    ap.add_argument("--noise-std", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", default=str(REPO / "figures" / "active" / "rmse_vs_budget.png"))
    ap.add_argument("--results", default=str(REPO / "results" / "active_sensing.json"))
    args = ap.parse_args()

    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    ds = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)
    strategies = list(STRATEGIES)

    # collect per-strategy curves across maps (shared budget grid via round index)
    per_strategy = {s: [] for s in strategies}
    for map_id in test_maps:
        sample = ds[ds.samples.index((map_id, 0))]
        gt = sample["radio_map_dbm"].numpy()[0]
        free = sample["free_mask"].numpy()[0].astype(bool)
        for s in strategies:
            rec = GPReconstructor(length_scale=args.length_scale, noise_std=args.noise_std)
            r = run_active_loop(rec, gt, free, s, n_rounds=args.n_rounds,
                                points_per_walk=args.points_per_walk,
                                noise_std=args.noise_std, seed=args.seed + map_id)
            per_strategy[s].append(r)
        print(f"  map {map_id}: " + " | ".join(
            f"{s}={per_strategy[s][-1].rmse_free_unobs[-1]:.1f}" for s in strategies))

    # average curves by round index (budgets are ~matched per round)
    out = {"config": vars(args), "strategies": {}}
    rounds = args.n_rounds + 1
    for s in strategies:
        rmse = np.array([r.rmse_free_unobs[:rounds] for r in per_strategy[s]])
        bud = np.array([r.budgets[:rounds] for r in per_strategy[s]])
        out["strategies"][s] = {
            "budget_mean": bud.mean(0).round(0).tolist(),
            "rmse_free_unobs_mean": rmse.mean(0).round(3).tolist(),
            "rmse_free_unobs_std": rmse.std(0).round(3).tolist(),
            "final_rmse": round(float(rmse.mean(0)[-1]), 2),
        }

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # --- figure: RMSE vs budget ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"max_variance": "#c0392b", "space_filling": "#27ae60",
              "coverage": "#2980b9", "random": "#7f8c8d"}
    for s in strategies:
        b = np.array(out["strategies"][s]["budget_mean"])
        m = np.array(out["strategies"][s]["rmse_free_unobs_mean"])
        sd = np.array(out["strategies"][s]["rmse_free_unobs_std"])
        ax.plot(b, m, "-o", ms=4, color=colors.get(s), label=f"{s} (final {m[-1]:.1f})")
        ax.fill_between(b, m - sd, m + sd, color=colors.get(s), alpha=0.12)
    ax.set_xlabel("measurement budget (# observations collected)")
    ax.set_ylabel("free-unobserved RMSE (dBm) — lower is better")
    ax.set_title(f"Active crowdsensing on RadioMapSeer ({args.n_maps} maps, GP reconstructor)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120)
    plt.close(fig)

    mv = out["strategies"]["max_variance"]["final_rmse"]
    rnd = out["strategies"]["random"]["final_rmse"]
    print(f"\n[active] final free-unobs RMSE: max_variance {mv} vs random {rnd} dBm "
          f"({rnd - mv:+.1f} dB better)")
    print(f"[active] figure -> {args.fig}\n[active] results -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

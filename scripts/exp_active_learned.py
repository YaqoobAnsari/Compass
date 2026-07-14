#!/usr/bin/env python
"""
System headline: does PHYSICS-AWARE (learned) uncertainty make max-variance
acquisition beat space-filling? Runs the active loop with the learned model and
with the GP, comparing max_variance vs space_filling for each. Expectation:
  GP:      max_variance ~= space_filling   (geometric uncertainty)
  learned: max_variance  >  space_filling   (physics-aware uncertainty)

  bash scripts/submit.sh 2g.35gb python scripts/exp_active_learned.py --ckpt experiments/reconstructors/full/best.ckpt
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

from compass.active import run_active_loop  # noqa: E402
from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids  # noqa: E402
from compass.recon import GPReconstructor  # noqa: E402
from compass.recon.learned import load_compass  # noqa: E402
from compass.recon.learned_active import LearnedActiveReconstructor  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--ckpt", default=str(REPO / "experiments" / "reconstructors" / "full" / "best.ckpt"))
    ap.add_argument("--n-maps", type=int, default=4)
    ap.add_argument("--n-rounds", type=int, default=12)
    ap.add_argument("--fig", default=str(REPO / "figures" / "active" / "learned_vs_gp.png"))
    ap.add_argument("--results", default=str(REPO / "results" / "active_learned.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    base = RadioMapSeerDataset(args.root, map_ids=test_maps)
    model = load_compass(args.ckpt, device)

    cases = {
        "GP/max_var": ("gp", "max_variance"),
        "GP/space_fill": ("gp", "space_filling"),
        "learned/max_var": ("learned", "max_variance"),
        "learned/space_fill": ("learned", "space_filling"),
    }
    curves = {c: [] for c in cases}
    for map_id in test_maps:
        s = base[base.samples.index((map_id, 0))]
        gt = s["radio_map_dbm"].numpy()[0]
        free = s["free_mask"].numpy()[0].astype(bool)
        building01 = s["building_map"].numpy()[0]
        tx_rc = tuple(int(v) for v in s["tx_rowcol"].numpy())
        for cname, (kind, strat) in cases.items():
            if kind == "gp":
                rec = GPReconstructor(length_scale=25.0)
            else:
                rec = LearnedActiveReconstructor(model, building01, free, tx_rc, device=device, n_mc=6)
            r = run_active_loop(rec, gt, free, strat, n_rounds=args.n_rounds,
                                points_per_walk=60, noise_std=2.0, seed=map_id)
            curves[cname].append(r)
        print(f"  map {map_id}: " + " | ".join(
            f"{c}={curves[c][-1].rmse_free_unobs[-1]:.1f}" for c in cases))

    out = {"ckpt": args.ckpt, "cases": {}}
    for c in cases:
        rmse = np.array([r.rmse_free_unobs[: args.n_rounds + 1] for r in curves[c]])
        bud = np.array([r.budgets[: args.n_rounds + 1] for r in curves[c]])
        out["cases"][c] = {"budget": bud.mean(0).round(0).tolist(),
                           "rmse": rmse.mean(0).round(3).tolist(),
                           "final": round(float(rmse.mean(0)[-1]), 2)}
    gp_gap = out["cases"]["GP/space_fill"]["final"] - out["cases"]["GP/max_var"]["final"]
    learned_gap = out["cases"]["learned/space_fill"]["final"] - out["cases"]["learned/max_var"]["final"]
    out["gp_maxvar_minus_spacefill"] = round(-gp_gap, 2)
    out["learned_maxvar_minus_spacefill"] = round(-learned_gap, 2)
    out["physics_aware_uncertainty_helps"] = bool(learned_gap > gp_gap + 0.5)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    styles = {"GP/max_var": ("#c0392b", "-"), "GP/space_fill": ("#c0392b", "--"),
              "learned/max_var": ("#27ae60", "-"), "learned/space_fill": ("#27ae60", "--")}
    for c in cases:
        b = out["cases"][c]["budget"]; m = out["cases"][c]["rmse"]
        col, ls = styles[c]
        ax.plot(b, m, ls, color=col, marker="o", ms=3, label=f"{c} (final {m[-1]:.1f})")
    ax.set_xlabel("measurement budget (# observations)")
    ax.set_ylabel("free-unobserved RMSE (dBm)")
    ax.set_title("Physics-aware (learned) vs geometric (GP) uncertainty for active sensing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120)
    plt.close(fig)

    print(f"\n[active-learned] GP max_var-vs-space_fill: {out['gp_maxvar_minus_spacefill']:+.2f} dB | "
          f"learned: {out['learned_maxvar_minus_spacefill']:+.2f} dB")
    print(f"[active-learned] physics-aware uncertainty helps max_variance: {out['physics_aware_uncertainty_helps']}")
    print(f"[active-learned] -> {args.results}, fig {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

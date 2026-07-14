#!/usr/bin/env python
"""
Stage-1 evidence: reconstruct a RadioMapSeer radio map from trajectory-collected
observations with the building-aware GP, vs IDW. Confirms (a) reconstruction is
well-posed, (b) the GP uncertainty tracks error — the signal the active loop will
steer on. Saves a 6-panel figure per sample + a metrics JSON.

  bash scripts/submit.sh 1g.18gb python scripts/exp_reconstruct_demo.py
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
from compass.data.conventions import PATHLOSS_MAX_DBM, PATHLOSS_MIN_DBM  # noqa: E402
from compass.data.trajectory import TrajectorySampler  # noqa: E402
from compass.active.collect import collect_along_trajectories, observed_mask  # noqa: E402
from compass.recon import GPReconstructor, IDWReconstructor  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--n-maps", type=int, default=4)
    ap.add_argument("--k", type=int, default=4, help="trajectories per sample")
    ap.add_argument("--n-points", type=int, default=120)
    ap.add_argument("--length-scale", type=float, default=25.0)
    ap.add_argument("--noise-std", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig-dir", default=str(REPO / "figures" / "reconstruct"))
    ap.add_argument("--results", default=str(REPO / "results" / "reconstruct_demo.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    ds = RadioMapSeerDataset(args.root, map_ids=test_maps, variant=args.variant)
    sampler = TrajectorySampler(seed=args.seed)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for map_id in test_maps:
        building = ds.load_building(map_id)
        sample = ds[ds.samples.index((map_id, 0))]
        gt = sample["radio_map_dbm"].numpy()[0]
        free = sample["free_mask"].numpy()[0].astype(bool)

        trajs = sampler.sample_many(building, k=args.k, n_points=args.n_points)
        obs = collect_along_trajectories(trajs, gt, rng=rng, noise_std=args.noise_std)
        obs_m = observed_mask(obs, gt.shape) & free

        gp = GPReconstructor(length_scale=args.length_scale, noise_std=args.noise_std).reconstruct(obs, free)
        idw = IDWReconstructor().reconstruct(obs, free)

        gp_m = gp.masked_rmse(gt, obs_m)
        idw_m = idw.masked_rmse(gt, obs_m)
        err = np.abs(gp.mean - gt)
        unobs = free & ~obs_m
        corr = float(np.corrcoef(err[unobs], gp.std[unobs])[0, 1]) if unobs.sum() > 10 else float("nan")
        rec = {
            "map_id": int(map_id), "n_obs": int(len(obs)),
            "coverage_free_pct": round(100 * obs_m.sum() / max(free.sum(), 1), 2),
            "gp_rmse_free_unobs": round(gp_m["rmse_free_unobs"], 2),
            "idw_rmse_free_unobs": round(idw_m["rmse_free_unobs"], 2),
            "gp_uncertainty_error_corr": round(corr, 3),
        }
        results.append(rec)
        print(f"  map {map_id}: n_obs={len(obs)} cov={rec['coverage_free_pct']}% | "
              f"GP free-unobs RMSE {rec['gp_rmse_free_unobs']} vs IDW {rec['idw_rmse_free_unobs']} dBm | "
              f"unc-err corr {rec['gp_uncertainty_error_corr']}")

        # --- 6-panel figure ---
        def show(ax, img, title, mask=None, cmap="viridis", vmin=PATHLOSS_MIN_DBM, vmax=PATHLOSS_MAX_DBM):
            d = np.where(mask, img, np.nan) if mask is not None else img
            im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            return im

        fig, ax = plt.subplots(2, 3, figsize=(14, 9))
        im = show(ax[0, 0], gt, "ground truth (dBm)", free)
        ax[0, 1].imshow(free, cmap="Greys", vmin=0, vmax=3)
        for tr in trajs:
            ax[0, 1].scatter(tr.cols, tr.rows, c=np.arange(len(tr)), cmap="plasma", s=3)
        ax[0, 1].set_title(f"{len(trajs)} trajectories ({rec['coverage_free_pct']}% free cov)", fontsize=10)
        ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
        show(ax[0, 2], gp.mean, f"GP mean — free-unobs RMSE {rec['gp_rmse_free_unobs']} dBm", free)
        show(ax[1, 0], gp.std, f"GP uncertainty (corr w/ err {rec['gp_uncertainty_error_corr']})", free,
             cmap="magma", vmin=0, vmax=float(np.nanpercentile(gp.std[free], 99)))
        show(ax[1, 1], np.where(free, err, np.nan), "GP abs error (dBm)", free,
             cmap="inferno", vmin=0, vmax=float(np.nanpercentile(err[free], 99)))
        show(ax[1, 2], idw.mean, f"IDW mean — free-unobs RMSE {rec['idw_rmse_free_unobs']} dBm", free)
        fig.suptitle(f"RadioMapSeer map {map_id}: trajectory reconstruction + uncertainty", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.savefig(fig_dir / f"map{map_id}_reconstruct.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "config": vars(args),
        "mean_gp_rmse_free_unobs": round(float(np.mean([r["gp_rmse_free_unobs"] for r in results])), 2),
        "mean_idw_rmse_free_unobs": round(float(np.mean([r["idw_rmse_free_unobs"] for r in results])), 2),
        "mean_uncertainty_error_corr": round(float(np.nanmean([r["gp_uncertainty_error_corr"] for r in results])), 3),
        "samples": results,
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(summary, indent=2))
    print(f"\n[recon] GP free-unobs RMSE {summary['mean_gp_rmse_free_unobs']} vs "
          f"IDW {summary['mean_idw_rmse_free_unobs']} dBm | "
          f"uncertainty-error corr {summary['mean_uncertainty_error_corr']}")
    print(f"[recon] figures -> {fig_dir}\n[recon] results -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

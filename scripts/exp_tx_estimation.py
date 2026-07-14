#!/usr/bin/env python
"""
DE-RISK: can we substitute an ESTIMATED transmitter for the unknown real TX?

On RadioMapSeer (true TX known), estimate the TX from ONLY the sparse trajectory
samples, then reconstruct three ways with the SAME trained 'full' model:
  * true TX  (upper bound)
  * estimated TX (our real-data scenario)
  * no-TX model (lower bound, trained without TX)
Reports TX estimation error + how much of the TX value the estimate recovers.

  bash scripts/submit.sh 2g.35gb python scripts/exp_tx_estimation.py
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
from compass.eval.benchmark import build_eval_sample  # noqa: E402
from compass.eval.stats import bootstrap_ci  # noqa: E402
from compass.recon.learned import load_compass  # noqa: E402
from compass.recon.tx_estimate import ESTIMATORS, log_distance_fit, tx_error_px  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def predict_dbm(model, batch, device):
    model.eval()  # dropout off -> deterministic point estimate
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(b).cpu().numpy()[0, 0]
    return signed_unit_to_dbm(out)


def rmse_free_unobs(pred, gt, free, obs_mask):
    m = free & ~obs_mask
    return float(np.sqrt(np.mean((pred - gt)[m] ** 2))) if m.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--full", default=str(REPO / "experiments" / "reconstructors" / "full" / "best.ckpt"))
    ap.add_argument("--notx", default=str(REPO / "experiments" / "reconstructors" / "no_tx" / "best.ckpt"))
    ap.add_argument("--jitter", default=str(REPO / "experiments" / "reconstructors" / "tx_jitter" / "best.ckpt"),
                    help="model trained with TX-jitter; included if the ckpt exists")
    ap.add_argument("--n-maps", type=int, default=12)
    ap.add_argument("--tx-per-map", type=int, default=6)
    ap.add_argument("--results", default=str(REPO / "results" / "tx_estimation.json"))
    ap.add_argument("--fig", default=str(REPO / "figures" / "tx_estimation" / "tx_estimation.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_compass(args.full, device)
    notx = load_compass(args.notx, device)
    jit = load_compass(args.jitter, device) if Path(args.jitter).exists() else None
    if jit is not None:
        print("[tx-est] including TX-jitter-trained model (+estimated TX)")
    test_maps = split_map_ids(list_map_ids(args.root))["test"][: args.n_maps]
    base = RadioMapSeerDataset(args.root, map_ids=test_maps)
    sampler = TrajectorySampler(seed=0)
    noise = MeasurementNoise()
    rng = np.random.default_rng(0)

    tx_err = {k: [] for k in ESTIMATORS}
    cases = {"full + true TX": [], "full + estimated TX": [], "no-TX model": []}
    if jit is not None:
        cases["jitter-trained + estimated TX"] = []
        cases["jitter-trained + true TX"] = []
    # best estimator on synthetic was the strongest-sample / centroid; use weighted_centroid
    from compass.recon.tx_estimate import weighted_centroid
    for map_id in test_maps:
        for tx in range(args.tx_per_map):
            sample = build_eval_sample(base, sampler, noise, map_id, tx, rng, k=3, n_points=100)
            true_tx = sample["tx_rc"]; obs = sample["obs"]
            for name, fn in ESTIMATORS.items():
                tx_err[name].append(tx_error_px(fn(obs), true_tx))
            est = weighted_centroid(obs)
            gt, free, om = sample["gt_dbm"], sample["free"], sample["obs_mask"]
            batch = sample["batch"]
            batch_est = dict(batch)
            batch_est["tx_rowcol"] = torch.tensor([[est[0], est[1]]], dtype=torch.long)

            cases["full + true TX"].append(rmse_free_unobs(predict_dbm(full, batch, device), gt, free, om))
            cases["full + estimated TX"].append(rmse_free_unobs(predict_dbm(full, batch_est, device), gt, free, om))
            cases["no-TX model"].append(rmse_free_unobs(predict_dbm(notx, batch, device), gt, free, om))
            if jit is not None:
                cases["jitter-trained + estimated TX"].append(rmse_free_unobs(predict_dbm(jit, batch_est, device), gt, free, om))
                cases["jitter-trained + true TX"].append(rmse_free_unobs(predict_dbm(jit, batch, device), gt, free, om))
        print(f"  map {map_id} done")
    r_true = cases["full + true TX"]; r_est = cases["full + estimated TX"]; r_notx = cases["no-TX model"]

    # gt and pred are both in dBm -> RMSE is already in dB (no rescaling)
    cis = {name: bootstrap_ci(vals) for name, vals in cases.items()}
    base_true = cis["full + true TX"]["mean"]
    base_notx = cis["no-TX model"]["mean"]
    gap = base_notx - base_true

    def recov(name):
        return round(float((base_notx - cis[name]["mean"]) / gap), 3) if gap > 1e-6 else float("nan")

    out = {
        "n_samples": len(r_true),
        "tx_estimation_error_px": {k: round(float(np.median(v)), 1) for k, v in tx_err.items()},
        "rmse": {name: cis[name] for name in cases},
        "tx_value_db": round(gap, 2),
        "recovery_full_plus_estimated": recov("full + estimated TX"),
    }
    if jit is not None:
        out["recovery_jitter_plus_estimated"] = recov("jitter-trained + estimated TX")
        rj = out["recovery_jitter_plus_estimated"]
        out["verdict"] = ("TX-jitter training salvages estimated TX (recovers most of TX value)"
                          if rj > 0.6 else "partial recovery with jitter training"
                          if rj > 0.2 else "estimated TX insufficient even with jitter training -> use no-TX model")
    else:
        out["verdict"] = "naive estimated TX insufficient; jitter-trained model not yet available"
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    # figure: RMSE bars (all cases) + TX error
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    names = list(cases)
    means = [cis[n]["mean"] for n in names]
    err = [[cis[n]["mean"] - cis[n]["lo"] for n in names], [cis[n]["hi"] - cis[n]["mean"] for n in names]]
    cmap = {"full + true TX": "#1e7d4f", "full + estimated TX": "#c0392b", "no-TX model": "#7f8c8d",
            "jitter-trained + estimated TX": "#e67e22", "jitter-trained + true TX": "#2980b9"}
    ax[0].bar(range(len(names)), means, yerr=err, color=[cmap.get(n, "#888") for n in names], capsize=4)
    ax[0].axhline(base_notx, color="#7f8c8d", ls="--", lw=1, label="no-TX floor")
    ax[0].axhline(base_true, color="#1e7d4f", ls="--", lw=1, label="true-TX ceiling")
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("free-unobserved RMSE (dB)")
    ax[0].set_title(f"TX value = {gap:.1f} dB"); ax[0].legend(fontsize=8)
    for k, v in tx_err.items():
        ax[1].hist(v, bins=25, alpha=0.5, label=f"{k} (med {np.median(v):.0f}px)")
    ax[1].set_xlabel("TX estimation error (px)"); ax[1].set_ylabel("# samples")
    ax[1].set_title("TX localization error by estimator"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120)
    plt.close(fig)

    print(f"\n[tx-est] TX value = {gap:.2f} dB (no-TX {base_notx:.2f} - true {base_true:.2f})")
    for name in cases:
        print(f"  {name:32s} {cis[name]['mean']:6.2f} dB  recovers {recov(name)*100:+.0f}%")
    print(f"[tx-est] median TX error: " + ", ".join(f"{k}={np.median(v):.0f}px" for k, v in tx_err.items()))
    print(f"[tx-est] VERDICT: {out['verdict']}")
    print(f"[tx-est] -> {args.results}, fig {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

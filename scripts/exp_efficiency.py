#!/usr/bin/env python
"""
Efficiency / cost comparison — params, inference latency, and the accuracy–cost
trade-off (a standard reviewer ask). Times every DL method + representative classical
methods on identical 256x256 inputs (GPU for DL, CPU for classical, as deployed).

  bash scripts/submit.sh 2g.35gb python scripts/exp_efficiency.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids
from compass.data.noise import MeasurementNoise
from compass.data.trajectory import TrajectorySampler
from compass.eval.benchmark import build_eval_sample
from compass.models.baselines.wrapper import BaselineReconstructor, load_baseline
from compass.recon import IDWReconstructor, RBFReconstructor
from compass.recon.learned import LearnedReconstructor, load_compass

REPO = Path(__file__).resolve().parents[1]


def time_learned(rec, sample, n=20):
    b = {k: (v.to(rec.device) if torch.is_tensor(v) else v) for k, v in sample["batch"].items()}
    with torch.no_grad():
        for _ in range(3):
            rec.model(b)  # warmup
        if rec.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            rec.model(b)
        if rec.device == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000  # ms/sample (single forward)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "efficiency.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_maps = split_map_ids(list_map_ids(REPO / "data" / "raw"))["test"][:2]
    base = RadioMapSeerDataset(str(REPO / "data" / "raw"), map_ids=test_maps, variant="IRT2")
    sample = build_eval_sample(base, TrajectorySampler(seed=0), MeasurementNoise(),
                               test_maps[0], 0, np.random.default_rng(0))

    out = {"device": device, "methods": {}}

    # DL: COMPASS variants + baselines
    dl = {}
    for name in ("full", "wnet"):
        ck = REPO / "experiments" / "reconstructors_long" / name / "best.ckpt"
        if ck.exists():
            dl[f"COMPASS-{name}"] = load_compass(str(ck), device)
    for d in sorted((REPO / "experiments" / "dl_baselines").glob("*/best.ckpt")):
        try:
            dl[d.parent.name] = load_baseline(str(d), device)
        except Exception:  # noqa: BLE001
            pass
    for name, model in dl.items():
        model.to(device).eval()
        rec = LearnedReconstructor(model, device) if name.startswith("COMPASS") else BaselineReconstructor(model, device)
        params = sum(p.numel() for p in model.parameters())
        mc = getattr(rec, "n_mc", 1)
        single = time_learned(rec, sample)
        out["methods"][name] = {"params_M": round(params / 1e6, 2),
                                "ms_per_forward": round(single, 2),
                                "mc_passes_for_uq": mc,
                                "ms_with_uq": round(single * mc, 2)}
        print(f"  {name:20s} {params/1e6:5.2f}M  {single:6.1f} ms/fwd  (UQ x{mc} = {single*mc:.0f} ms)")

    # classical (CPU, as deployed): time reconstruct
    for name, r in [("IDW", IDWReconstructor(power=1.0)),
                    ("RBF(mq)", RBFReconstructor(kernel="multiquadric", epsilon=20.0))]:
        t0 = time.time()
        for _ in range(5):
            r.reconstruct(sample["obs"], sample["free"])
        ms = (time.time() - t0) / 5 * 1000
        out["methods"][name] = {"params_M": 0.0, "ms_per_forward": round(ms, 2),
                                "mc_passes_for_uq": 1, "ms_with_uq": round(ms, 2)}
        print(f"  {name:20s}  0.00M  {ms:6.1f} ms  (CPU)")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))
    print(f"[efficiency] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Train a COMPASS reconstructor variant. Driven entirely by flags so the full model
and every ablation use one code path.

  bash scripts/submit.sh 7g.141gb python scripts/train_reconstructor.py --name full
  ... --name no_building --no-building   (etc.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compass.data import list_map_ids, split_map_ids
from compass.training.train import TrainConfig, train

REPO = Path(__file__).resolve().parents[1]

# Named ablations: which flag each turns OFF (innovation under test).
ABLATIONS = {
    "full": {},
    "no_building": {"use_building": False},
    "no_tx": {"use_tx": False},
    "order_blind": {"use_order": False},
    "order_shuffle": {"order_shuffle": True},
    "no_device": {"use_device": False},
    "no_rcl": {"use_rcl": False},
    "iid_noise": {"correlated_noise": False},
    "tx_jitter": {"tx_jitter_px": 35.0},  # train with a noisy TX prior (~estimation error) -> robust to estimated TX
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full", choices=list(ABLATIONS))
    ap.add_argument("--arch", default="compass", choices=["compass", "compass_wnet"])
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--train-maps", type=int, default=120)
    ap.add_argument("--val-maps", type=int, default=24)
    ap.add_argument("--tx-per-map", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--use-occlusion", action="store_true",
                    help="add the explicit ray-occlusion / diffraction geometry channel")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    splits = split_map_ids(list_map_ids(args.root))
    out = args.out or str(REPO / "experiments" / "reconstructors" / args.name)
    cfg = TrainConfig(
        root=args.root,
        out_dir=out,
        train_maps=splits["train"][: args.train_maps],
        val_maps=splits["val"][: args.val_maps],
        tx_per_map=args.tx_per_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
        arch=args.arch,
        use_occlusion=args.use_occlusion,
        **ABLATIONS[args.name],
    )
    print(f"[train] variant={args.name} -> {out}  (train={len(cfg.train_maps)} maps x {cfg.tx_per_map} tx)")
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

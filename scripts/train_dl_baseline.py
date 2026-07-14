#!/usr/bin/env python
"""
Train one external DL radio-map baseline on the SAME data/split/budget as COMPASS.

  bash scripts/submit.sh 7g.141gb python scripts/train_dl_baseline.py --arch radiounet
  ... --arch pmnet | sparse_unet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compass.data import list_map_ids, split_map_ids
from compass.models.baselines import BASELINES
from compass.training.train_baseline import BaselineTrainConfig, train_baseline

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(BASELINES))
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--train-maps", type=int, default=120)
    ap.add_argument("--val-maps", type=int, default=24)
    ap.add_argument("--tx-per-map", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    splits = split_map_ids(list_map_ids(args.root))
    out = args.out or str(REPO / "experiments" / "dl_baselines" / args.arch)
    cfg = BaselineTrainConfig(
        arch=args.arch, root=args.root, out_dir=out,
        train_maps=splits["train"][: args.train_maps],
        val_maps=splits["val"][: args.val_maps],
        tx_per_map=args.tx_per_map, epochs=args.epochs, batch_size=args.batch_size,
    )
    print(f"[train-dl] arch={args.arch} -> {out}  (train={len(cfg.train_maps)} maps x {cfg.tx_per_map} tx)")
    train_baseline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

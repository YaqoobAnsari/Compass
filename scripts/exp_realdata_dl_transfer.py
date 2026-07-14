#!/usr/bin/env python
"""
Phase G-real — can the synthetic-trained DL radio-map nets be used on REAL
crowdsensed data (UniCellular held-out-RP reconstruction)?

Two honest findings this quantifies:
  1. APPLICABILITY: RadioUNet & PMNet (the synthetic winners) REQUIRE the TX location
     as input — which real crowdsensing does NOT have (boosters/distributed sources;
     A5: only 18% of cells are locatable). They are therefore inapplicable to the real
     TX-agnostic task. Only TX-agnostic methods (COMPASS-no_tx, classical, wall-aware)
     apply. We state this and evaluate only the applicable ones.
  2. SIM-TO-REAL GAP: COMPASS-no_tx is a dense-image model trained on synthetic OUTDOOR
     pathloss; applied zero-shot to real INDOOR sparse RPs (rasterised, per-cell
     range-normalised, floor-plan building channel), how far is it from the native
     point methods (classical RBF from A2 = 5.49 dB)?

  bash scripts/submit.sh 2g.35gb python scripts/exp_realdata_dl_transfer.py
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
from PIL import Image  # noqa: E402

from compass.eval.stats import bootstrap_ci  # noqa: E402
from compass.realdata.reconstruct import METHODS, M_PER_PX, loro_errors  # noqa: E402
from compass.realdata.walls import cell_rp_tables_px, load_wall_mask  # noqa: E402
from compass.recon.learned import LearnedReconstructor, load_compass  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
S = 256  # model input size


def building256(site, floor):
    p = REPO / "external/unicellular-fingerprint-dataset/floor_plans" / site / f"floor{floor}.png"
    bar = load_wall_mask(site, floor)
    if bar is None:
        return None, None
    im = Image.fromarray((bar * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
    b = (np.asarray(im) > 127).astype(np.float32)  # 1=structure(building proxy)
    return b, bar.shape  # (256,256), native (H,W)


def rasterize(pos_px, rss_norm, native_hw, building, dil=2):
    """Place observed RPs on a 256x256 grid (RP coords are (x=col,y=row) in plan frame)."""
    H, W = native_hw
    sparse = np.zeros((S, S), np.float32)
    mask = np.zeros((S, S), np.float32)
    for (x, y), v in zip(pos_px, rss_norm):
        c = int(np.clip(x / W * S, 0, S - 1))
        r = int(np.clip(y / H * S, 0, S - 1))
        sparse[max(0, r - dil):r + dil + 1, max(0, c - dil):c + dil + 1] = v
        mask[max(0, r - dil):r + dil + 1, max(0, c - dil):c + dil + 1] = 1.0
    cov = mask.copy()
    batch = {
        "sparse_rss": torch.from_numpy(sparse[None, None]),
        "mask": torch.from_numpy(mask[None, None]),
        "coverage": torch.from_numpy(cov[None, None]),
        "building": torch.from_numpy(building[None, None]),
        "tx_rowcol": torch.tensor([[S // 2, S // 2]]),  # dummy (no_tx ignores it)
        "sequence": torch.zeros(1, 64, 11),
    }
    return batch


def rp_to_grid(pos_px, native_hw):
    H, W = native_hw
    return [(int(np.clip(y / H * S, 0, S - 1)), int(np.clip(x / W * S, 0, S - 1))) for x, y in pos_px]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "realdata_dl_transfer.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "realdata_dl_transfer"))
    ap.add_argument("--min-rp", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = REPO / "experiments" / "reconstructors_long" / "no_tx" / "best.ckpt"
    if not ck.exists():
        ck = REPO / "experiments" / "reconstructors" / "no_tx" / "best.ckpt"
    compass_notx = load_compass(str(ck), device)
    print(f"[G-real] loaded COMPASS-no_tx from {ck}")

    rng = np.random.default_rng(0)
    errs = {"COMPASS-no_tx (zero-shot)": [], "RBF(mq) native": [], "IDW(p=2) native": [], "GP native": []}
    n_cells = 0
    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            b256, native = building256(site, fl)
            if b256 is None:
                continue
            for _cell, (pos_px, rss) in cell_rp_tables_px(site, fl, min_rp=args.min_rp).items():
                n = len(pos_px)
                for rep in range(3):
                    idx = rng.permutation(n)
                    ntest = max(1, int(round(n * args.test_frac)))
                    test, train = idx[:ntest], idx[ntest:]
                    if len(train) < 4:
                        continue
                    # per-cell normalisation from TRAIN RPs only (no leakage)
                    lo, hi = rss[train].min(), rss[train].max()
                    rng_db = max(hi - lo, 1.0)
                    norm = 2 * (rss[train] - lo) / rng_db - 1
                    batch = rasterize(pos_px[train], norm, native, b256)
                    with torch.no_grad():
                        out = compass_notx({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
                    pred = out.cpu().numpy()[0, 0]  # (256,256) normalised
                    grid = rp_to_grid(pos_px[test], native)
                    yhat_norm = np.array([pred[r, c] for r, c in grid])
                    yhat_db = (yhat_norm + 1) / 2 * rng_db + lo
                    errs["COMPASS-no_tx (zero-shot)"] += list(np.abs(yhat_db - rss[test]))
                    # native classical on the SAME split (metres)
                    pm = pos_px * M_PER_PX
                    for label, key in [("RBF(mq) native", "RBF(multiquadric)"),
                                       ("IDW(p=2) native", "IDW(p=2)"), ("GP native", "GP/Kriging")]:
                        try:
                            yh = METHODS[key](pm[train], rss[train], pm[test])
                            errs[label] += list(np.abs(np.asarray(yh) - rss[test]))
                        except Exception:  # noqa: BLE001
                            pass
                n_cells += 1
        print(f"  {site}: cumulative cells={n_cells}")

    out = {"n_cells": n_cells, "test_frac": args.test_frac, "min_rp": args.min_rp,
           "note": ("RadioUNet & PMNet require a TX-location input, which real crowdsensing "
                    "lacks (boosters; A5 18% locatable) -> INAPPLICABLE to the TX-agnostic real "
                    "task. Only TX-agnostic methods evaluated."),
           "rmse": {}, "rmse_ci": {}}
    for m, e in errs.items():
        out["rmse"][m] = round(float(np.sqrt(np.mean(np.square(e)))), 3) if e else None
        out["rmse_ci"][m] = bootstrap_ci(np.abs(e)) if e else None
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    ms = list(errs)
    vals = [out["rmse"][m] for m in ms]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if "COMPASS" in m else "#2980b9" for m in ms]
    ax.bar(range(len(ms)), vals, color=colors)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("held-out-RP RMSE (dB)")
    ax.set_title(f"Real-data: synthetic-DL zero-shot vs native classical ({n_cells} cells, 50% obs)")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "realdata_dl_transfer.png", dpi=120); plt.close(fig)

    print(f"\n[G-real] cells={n_cells} (test_frac={args.test_frac})")
    for m in ms:
        print(f"  {m:28s} {out['rmse'][m]}")
    print("[G-real] RadioUNet/PMNet: INAPPLICABLE (require TX, unavailable on real).")
    print(f"[G-real] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

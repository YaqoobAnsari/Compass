#!/usr/bin/env python
"""
A3b — LEARNED order model on real walks (confirms #3 end-to-end, beyond A3's model-free
gap-filling). An order-aware bidirectional GRU vs an order-blind DeepSet, both trained to
impute held-out scans in real mobile RSS streams. If the GRU (which uses adjacency)
beats the set model (which cannot), a learned order-aware model captures the E15 signal.

Sequences split into train/test walks (no leakage). Per-sequence mean-centred.

  bash scripts/submit.sh 2g.35gb python scripts/exp_order_learned_real.py
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
import torch.nn as nn  # noqa: E402

from compass.realdata.unicellular import RECON_COLS, RSS_COL, load_floor, mobile_sequences, top_transmitters  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITES_FLOORS = {"cmuq": [1, 2, 3], "ec_parking": [0, 1, 2]}
MIN_LEN, TOP_TX, TEST_FRAC = 30, 8, 0.4


class GRUImputer(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(2, hidden, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, mask=None):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


class SetImputer(nn.Module):
    """Order-BLIND: per-scan encoder + global pool over observed (permutation-invariant)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        h = self.enc(x)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        h2 = torch.cat([h, pooled[:, None].expand_as(h)], dim=-1)
        return self.head(h2).squeeze(-1)


def load_sequences():
    seqs = []
    for site, floors in SITES_FLOORS.items():
        for fl in floors:
            try:
                df = load_floor(site, "mobile", fl, usecols=RECON_COLS)
            except (FileNotFoundError, RuntimeError):
                continue
            for tx in top_transmitters(df, n=TOP_TX, min_phones=2):
                for _phone, sq in mobile_sequences(df, tx).items():
                    y = sq[RSS_COL].to_numpy(dtype=np.float32)
                    if len(y) >= MIN_LEN and y.std() > 1e-3:
                        seqs.append(y)
    return seqs


def make_masked(y, rng):
    """Return (x (T,2), mask (T,), target (T,), mean). mask=1 where HELD OUT."""
    T = len(y)
    m = y.mean()
    yc = y - m
    test = rng.choice(np.arange(1, T - 1), size=max(1, int(TEST_FRAC * T)), replace=False)
    held = np.zeros(T, np.float32); held[test] = 1.0
    x = np.stack([np.where(held > 0.5, 0.0, yc), 1.0 - held], axis=1).astype(np.float32)  # [rss,obs-mask]
    return x, held, yc.astype(np.float32), m


def run(model, seqs, device, order_aware, epochs=80, seed=0, train=True):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4) if train else None
    n_ep = epochs if train else 1
    errs = []
    for ep in range(n_ep):
        if train:
            model.train()
        else:
            model.eval()
        order = rng.permutation(len(seqs))
        for si in order:
            y = seqs[si]
            x, held, yc, _ = make_masked(y, rng)
            xt = torch.tensor(x[None]).to(device)
            heldt = torch.tensor(held[None]).to(device)
            obs_mask = 1.0 - heldt
            pred = model(xt, obs_mask)
            tgt = torch.tensor(yc[None]).to(device)
            hm = heldt > 0.5
            if train:
                loss = ((pred[hm] - tgt[hm]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            elif ep == 0:
                errs += list(np.abs((pred[hm] - tgt[hm]).detach().cpu().numpy()))
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "order_learned_real.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "order_learned_real"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    seqs = load_sequences()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(seqs))
    cut = int(0.7 * len(seqs))
    train_seqs = [seqs[i] for i in idx[:cut]]
    test_seqs = [seqs[i] for i in idx[cut:]]
    print(f"[A3b] {len(seqs)} walks -> {len(train_seqs)} train / {len(test_seqs)} test")

    gru = GRUImputer().to(device)
    run(gru, train_seqs, device, True, train=True)
    gru_err = run(gru, test_seqs, device, True, train=False)

    setm = SetImputer().to(device)
    run(setm, train_seqs, device, False, train=True)
    set_err = run(setm, test_seqs, device, False, train=False)

    # mean baseline (predict 0 in centred space)
    rng2 = np.random.default_rng(1)
    mean_err = []
    for y in test_seqs:
        _, held, yc, _ = make_masked(y, rng2)
        mean_err += list(np.abs(yc[held > 0.5]))

    def rmse(e):
        return round(float(np.sqrt(np.mean(np.square(e)))), 3) if e else None

    out = {"n_walks": len(seqs), "n_test": len(test_seqs),
           "gru_order_aware_rmse": rmse(gru_err), "set_order_blind_rmse": rmse(set_err),
           "mean_baseline_rmse": rmse(mean_err),
           "gru_params": sum(p.numel() for p in gru.parameters()),
           "set_params": sum(p.numel() for p in setm.parameters())}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["GRU (order-aware)", "DeepSet (order-blind)", "mean baseline"]
    vals = [out["gru_order_aware_rmse"], out["set_order_blind_rmse"], out["mean_baseline_rmse"]]
    ax.bar(labels, vals, color=["#27ae60", "#c0392b", "#7f8c8d"])
    ax.set_ylabel("held-out-scan imputation RMSE (dB)")
    ax.set_title(f"A3b — LEARNED order model on real walks ({len(test_seqs)} test walks)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(Path(args.figdir) / "order_learned_real.png", dpi=120); plt.close(fig)

    print(f"[A3b] GRU(order-aware)={out['gru_order_aware_rmse']}  "
          f"Set(order-blind)={out['set_order_blind_rmse']}  mean={out['mean_baseline_rmse']}")
    print(f"[A3b] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

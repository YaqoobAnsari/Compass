#!/usr/bin/env python
"""
Cross-SITE generalization on real data (a key reviewer ask): does a model trained on
ONE site transfer to a DIFFERENT site? Tests (a) the learned geometry-aware interpolator
and (b) the GRU order model, training on CMUQ and testing on EC Parking and vice versa,
compared with within-site performance. Classical needs no training (same either way) and
anchors the comparison.

  bash scripts/submit.sh 2g.35gb python scripts/exp_cross_site.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from compass.realdata.learned_interp import SetInterpolator, build_features, make_samples
from compass.realdata.reconstruct import METHODS, M_PER_PX
from compass.realdata.unicellular import RECON_COLS, RSS_COL, load_floor, mobile_sequences, top_transmitters
from compass.realdata.walls import cell_rp_tables_px, load_wall_mask, wall_matrix_m

REPO = Path(__file__).resolve().parents[1]
SITES = {"cmuq": [1, 2, 3], "ec_parking": [1, 2]}
MOBILE = {"cmuq": [1, 2, 3], "ec_parking": [0, 1, 2]}


def cells_for(site):
    out = []
    for fl in SITES[site]:
        bar = load_wall_mask(site, fl)
        if bar is None:
            continue
        for _c, (pos, rss) in cell_rp_tables_px(site, fl, min_rp=10).items():
            out.append((pos, rss, wall_matrix_m(pos, bar)))
    return out


def train_interp(cells, device, epochs=120):
    data = make_samples(cells, use_walls=True)
    if data is None:
        return None
    F, Mk, Cm, Y = (t.to(device) for t in data)
    m = SetInterpolator().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Y), device=device)
        for b in range(0, len(Y), 256):
            i = perm[b:b + 256]
            loss = ((m(F[i], Mk[i], Cm[i]) - Y[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def eval_interp(m, cells, device):
    errs = []
    m.eval()
    for pos, rss, W in cells:
        pm = pos * M_PER_PX
        for i in range(len(pos)):
            de = np.linalg.norm(pm - pm[i], axis=1)
            keep = np.where((de > 1.0) & (np.arange(len(pos)) != i))[0]
            if len(keep) < 4:
                continue
            f, msk, c = build_features(pos, rss, W, keep, i, True)
            p = m(torch.tensor(f[None]).to(device), torch.tensor(msk[None]).to(device),
                  torch.tensor([c], dtype=torch.float32).to(device))
            errs.append(abs(float(p) - rss[i]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def classical_rmse(cells, key="RBF(multiquadric)"):
    errs = []
    for pos, rss, _ in cells:
        pm = pos * M_PER_PX
        for i in range(len(pos)):
            de = np.linalg.norm(pm - pm[i], axis=1)
            keep = np.where((de > 1.0) & (np.arange(len(pos)) != i))[0]
            if len(keep) < 4:
                continue
            try:
                yh = float(METHODS[key](pm[keep], rss[keep], pm[i:i + 1])[0])
                if np.isfinite(yh):
                    errs.append(abs(yh - rss[i]))
            except Exception:  # noqa: BLE001
                pass
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


# --- order model (self-contained) ---
class GRUImputer(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(2, hidden, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def walks_for(site):
    seqs = []
    for fl in MOBILE[site]:
        try:
            df = load_floor(site, "mobile", fl, usecols=RECON_COLS)
        except (FileNotFoundError, RuntimeError):
            continue
        for tx in top_transmitters(df, n=8, min_phones=2):
            for _p, sq in mobile_sequences(df, tx).items():
                y = sq[RSS_COL].to_numpy(dtype=np.float32)
                if len(y) >= 30 and y.std() > 1e-3:
                    seqs.append(y)
    return seqs


def _mask(y, rng):
    T = len(y); m = y.mean(); yc = y - m
    test = rng.choice(np.arange(1, T - 1), size=max(1, int(0.4 * T)), replace=False)
    held = np.zeros(T, np.float32); held[test] = 1.0
    x = np.stack([np.where(held > 0.5, 0.0, yc), 1.0 - held], 1).astype(np.float32)
    return x, held, yc.astype(np.float32)


def train_gru(seqs, device, epochs=80, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = GRUImputer().to(device); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for si in rng.permutation(len(seqs)):
            x, held, yc = _mask(seqs[si], rng)
            xt = torch.tensor(x[None]).to(device)
            pred = m(xt); tgt = torch.tensor(yc[None]).to(device); hm = torch.tensor(held[None]).to(device) > 0.5
            loss = ((pred[hm] - tgt[hm]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def eval_gru(m, seqs, device, seed=1):
    m.eval(); rng = np.random.default_rng(seed); errs = []
    for y in seqs:
        x, held, yc = _mask(y, rng)
        pred = m(torch.tensor(x[None]).to(device))
        hm = torch.tensor(held[None]).to(device) > 0.5
        errs += list(np.abs((pred[hm] - torch.tensor(yc[None]).to(device)[hm]).cpu().numpy()))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "cross_site.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cmuq, ec = cells_for("cmuq"), cells_for("ec_parking")
    out = {"n_cmuq_cells": len(cmuq), "n_ec_cells": len(ec), "interpolator": {}, "order_gru": {}, "classical": {}}

    # interpolator: train A -> test B
    m_c = train_interp(cmuq, device); m_e = train_interp(ec, device)
    out["interpolator"] = {
        "train_cmuq_test_ec": round(eval_interp(m_c, ec, device), 3),
        "train_ec_test_cmuq": round(eval_interp(m_e, cmuq, device), 3),
        "within_cmuq": round(eval_interp(m_c, cmuq, device), 3),
        "within_ec": round(eval_interp(m_e, ec, device), 3),
    }
    out["classical"] = {"rbf_cmuq": round(classical_rmse(cmuq), 3), "rbf_ec": round(classical_rmse(ec), 3)}

    # order GRU: train A -> test B
    wc, we = walks_for("cmuq"), walks_for("ec_parking")
    out["n_cmuq_walks"], out["n_ec_walks"] = len(wc), len(we)
    g_c = train_gru(wc, device); g_e = train_gru(we, device)
    out["order_gru"] = {
        "train_cmuq_test_ec": round(eval_gru(g_c, we, device), 3),
        "train_ec_test_cmuq": round(eval_gru(g_e, wc, device), 3),
        "within_cmuq": round(eval_gru(g_c, wc, device), 3),
        "within_ec": round(eval_gru(g_e, we, device), 3),
    }

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))
    print("[cross-site] interpolator:", out["interpolator"])
    print("[cross-site] classical:", out["classical"])
    print("[cross-site] order GRU:", out["order_gru"])
    print(f"[cross-site] -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
DIAL - Detection-aware, Device-Invariant Attention Localizer.

A novel architecture for cross-device indoor localization from sparse crowdsensed
cellular fingerprints, built to target DETECTION heterogeneity (phones detect different
cell sets; heard-set Jaccard ~0.30):

  1. Detection-aware attention encoder. A fingerprint is an attention pool over its
     DETECTED cells; each cell carries a learned identity embedding + its RSS; attention
     is masked to heard cells. This models "which cells are heard" natively (vs a fixed
     dense vector), and the per-cell embedding learns cell reliability.
  2. Adversarial device-invariance. A gradient-reversal device classifier forces the
     embedding to be indistinguishable across phones -> device-invariant by construction.
  3. Detection-dropout augmentation. Randomly hide heard cells to simulate a range of
     receiver sensitivities.

Evaluated leave-one-phone-out vs the plain contrastive embedding and all baselines.

  bash scripts/submit.sh 1g.18gb python scripts/exp_localization_dial.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compass.eval.stats import paired_wilcoxon
from compass.realdata.localize import NOT_HEARD, load_floor

REPO = Path(__file__).resolve().parents[1]
SF = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1)]
RSS_MID, RSS_SCALE = -85.0, 15.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_samples(fd, rng, k=12, per_rp=30):
    """phone -> (rss (N,nc), mask (N,nc), rp_label (N,))."""
    ridx = {r: i for i, r in enumerate(fd.rps)}
    out = {}
    for p in fd.phones:
        R, M, Y = [], [], []
        for r in fd.rps:
            mat = fd.mats[(p, r)]
            n = mat.shape[0]
            if n == 0:
                continue
            for _ in range(per_rp):
                rows = rng.choice(n, min(k, n), replace=False)
                m = np.nanmean(mat[rows], axis=0)
                heard = ~np.isnan(m)
                rss = np.where(heard, (np.nan_to_num(m) - RSS_MID) / RSS_SCALE, 0.0)
                R.append(rss); M.append(heard.astype(np.float32)); Y.append(ridx[r])
        if R:
            out[p] = (np.array(R, np.float32), np.array(M, np.float32), np.array(Y))
    return out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


class DIAL(nn.Module):
    def __init__(self, n_cells, n_dev, d_cell=32, h=128, d=32):
        super().__init__()
        self.cell_emb = nn.Embedding(n_cells, d_cell)
        self.token = nn.Sequential(nn.Linear(d_cell + 2, h), nn.GELU(), nn.Linear(h, h))
        self.q = nn.Parameter(torch.randn(h) * 0.1)
        self.proj = nn.Sequential(nn.GELU(), nn.Linear(h, d))
        self.dev_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, n_dev))
        self.register_buffer("ids", torch.arange(n_cells))

    def embed(self, rss, mask):
        B = rss.shape[0]
        ce = self.cell_emb(self.ids).unsqueeze(0).expand(B, -1, -1)      # (B, nc, d_cell)
        tok_in = torch.cat([ce, rss.unsqueeze(-1), mask.unsqueeze(-1)], dim=-1)
        tok = self.token(tok_in)                                         # (B, nc, h)
        score = (tok * self.q).sum(-1)                                   # (B, nc)
        score = score.masked_fill(mask < 0.5, -1e9)
        attn = torch.softmax(score, dim=1).unsqueeze(-1)
        z = self.proj((attn * tok).sum(1))
        return z / (z.norm(dim=1, keepdim=True) + 1e-8)

    def device_logits(self, z, alpha):
        return self.dev_head(GradReverse.apply(z, alpha))


def supcon(z, y, temp=0.1):
    sim = z @ z.t() / temp
    m = torch.eye(z.shape[0], device=z.device).bool()
    sim = sim.masked_fill(m, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~m
    return -(logp.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1)).mean()


def dropout_cells(rss, mask, p):
    drop = (torch.rand_like(mask) < p) & (mask > 0.5)
    return rss.masked_fill(drop, 0.0), mask.masked_fill(drop, 0.0)


def train(model, Rtr, Mtr, Ytr, Dtr, rng, epochs=60, bs=512, lam=0.3):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    R = torch.tensor(Rtr, device=DEV); M = torch.tensor(Mtr, device=DEV)
    Y = torch.tensor(Ytr, device=DEV); D = torch.tensor(Dtr, device=DEV)
    n = len(Rtr)
    for ep in range(epochs):
        alpha = lam * min(1.0, ep / 15.0)                                # ramp adversary
        idx = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            rss, msk = dropout_cells(R[b], M[b], float(rng.uniform(0.05, 0.4)))
            opt.zero_grad()
            z = model.embed(rss, msk)
            loss = supcon(z, Y[b]) + F.cross_entropy(model.device_logits(z, alpha), D[b])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def emb_all(model, R, M):
    return model.embed(torch.tensor(R, device=DEV), torch.tensor(M, device=DEV)).cpu().numpy()


def protos(Z, y, nrp):
    p = np.zeros((nrp, Z.shape[1])); c = np.zeros(nrp)
    for z, lab in zip(Z, y):
        p[lab] += z; c[lab] += 1
    p = p / np.maximum(c[:, None], 1)
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)


def knn(z, db, xy, k=5):
    d = np.linalg.norm(db - z, axis=1)
    i = np.argsort(d)[:k]
    w = 1.0 / (d[i] + 1e-6)
    return (w[:, None] * xy[i]).sum(0) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "localization_dial.json"))
    args = ap.parse_args()
    print(f"[DIAL] device={DEV}")
    rng = np.random.default_rng(0)
    agg = {"dial": [], "ceiling": []}
    per_floor = {}
    for site, fl in SF:
        fd = load_floor(site, fl)
        if fd is None:
            continue
        nc = len(fd.cells)
        s = build_samples(fd, rng)
        fl_err = []
        for t in fd.phones:
            ref = [p for p in fd.phones if p != t and p in s]
            if t not in s or len(ref) < 2:
                continue
            dmap = {p: i for i, p in enumerate(ref)}
            Rtr = np.vstack([s[p][0] for p in ref]); Mtr = np.vstack([s[p][1] for p in ref])
            Ytr = np.concatenate([s[p][2] for p in ref])
            Dtr = np.concatenate([np.full(len(s[p][2]), dmap[p]) for p in ref])
            torch.manual_seed(0)
            model = train(DIAL(nc, len(ref)).to(DEV), Rtr, Mtr, Ytr, Dtr, rng)
            proto = protos(emb_all(model, Rtr, Mtr), Ytr, len(fd.rps))
            Rte, Mte, Yte = s[t]
            Zte = emb_all(model, Rte, Mte)
            fl_err.append(float(np.median([np.linalg.norm(knn(z, proto, fd.xy) - fd.xy[y])
                                           for z, y in zip(Zte, Yte)])))
        key = f"{site}_f{fl}"
        per_floor[key] = round(float(np.nanmean(fl_err)), 2)
        agg["dial"] += fl_err
        print(f"[{key}] DIAL={per_floor[key]}")

    a = np.array(agg["dial"])
    out = {"per_floor": per_floor, "n_phones": len(a),
           "dial_mean_m": round(float(a.mean()), 2), "dial_median_m": round(float(np.median(a)), 2),
           "per_phone": [round(float(x), 2) for x in a]}
    # compare vs plain learned embedding (from the consolidated benchmark)
    try:
        prev = json.loads((REPO / "results" / "localization_benchmark.json").read_text())
        learned = prev["pooled"]["learned"]["mean"]
        naive = prev["pooled"]["naive"]["mean"]
        out["vs_learned_embedding"] = learned
        out["vs_naive"] = naive
        # paired wilcoxon needs per-phone; recompute learned per-phone unavailable here -> report means
        out["delta_vs_learned_m"] = round(out["dial_mean_m"] - learned, 2)
    except Exception:  # noqa: BLE001
        pass
    Path(args.results).write_text(json.dumps(out, indent=2))
    print("\n===== DIAL cross-device localization (per-phone median err, m) =====")
    print(f"  DIAL (ours)             : {out['dial_mean_m']}")
    if "vs_learned_embedding" in out:
        print(f"  plain learned embedding : {out['vs_learned_embedding']}  (delta {out['delta_vs_learned_m']})")
        print(f"  naive baseline          : {out['vs_naive']}")
    print(f"\n[DIAL] -> {args.results}")


if __name__ == "__main__":
    main()

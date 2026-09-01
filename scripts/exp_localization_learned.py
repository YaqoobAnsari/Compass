#!/usr/bin/env python
"""
Learned device-invariant fingerprint localization (the COMPASS positioning method).

A small encoder maps a sparse, device-biased cellular fingerprint -> a location-
discriminative, device-invariant embedding, trained supervised-contrastive with the
REFERENCE POINT as the class label (so same-location / different-phone fingerprints are
pulled together) and cell-dropout augmentation that simulates receiver detection
heterogeneity. Evaluated leave-one-phone-out against a same-device oracle ceiling.

Also sweeps FEW-ANCHOR self-calibration: the held-out device provides K known-location
fingerprints, added to the map in embedding space (the "walk to a few spots" story).

  bash scripts/submit.sh 1g.18gb python scripts/exp_localization_learned.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from compass.eval.stats import paired_wilcoxon
from compass.realdata.localize import NOT_HEARD, load_floor

REPO = Path(__file__).resolve().parents[1]
SF = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1)]
RSS_MID, RSS_SCALE = -85.0, 15.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def featurize(vec):
    heard = vec > NOT_HEARD + 1e-6
    f_rss = np.where(heard, (vec - RSS_MID) / RSS_SCALE, 0.0)
    return np.concatenate([f_rss, heard.astype(np.float32)])


def agg(mat, rows):
    if len(rows) == 0 or mat.shape[0] == 0:
        return np.full(mat.shape[1], NOT_HEARD)
    m = np.nanmean(mat[rows], axis=0)
    return np.where(np.isnan(m), NOT_HEARD, m)


def build_samples(fd, phones, rng, k=12, per_rp=30):
    """Return dict phone -> (X feats, rp_idx, rp_of_sample) using random scan windows."""
    ridx = {r: i for i, r in enumerate(fd.rps)}
    out = {}
    for p in phones:
        feats, labels = [], []
        for r in fd.rps:
            mat = fd.mats[(p, r)]
            n = mat.shape[0]
            if n == 0:
                continue
            for _ in range(per_rp):
                rows = rng.choice(n, min(k, n), replace=False)
                feats.append(featurize(agg(mat, rows)))
                labels.append(ridx[r])
        if feats:
            out[p] = (np.array(feats, np.float32), np.array(labels))
    return out


class Encoder(nn.Module):
    def __init__(self, din, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=1, keepdim=True) + 1e-8)


def supcon(z, y, temp=0.1):
    sim = z @ z.t() / temp
    n = z.shape[0]
    mask_self = torch.eye(n, device=z.device).bool()
    sim = sim.masked_fill(mask_self, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~mask_self
    denom = pos.sum(1).clamp(min=1)
    return -(logp.masked_fill(~pos, 0).sum(1) / denom).mean()


def cell_dropout_gpu(x, nc, p):
    """Randomly hide heard cells (simulate a less-sensitive receiver), on-device."""
    x = x.clone()
    mask = x[:, nc:]
    drop = (torch.rand_like(mask) < p) & (mask > 0.5)
    x[:, :nc][drop] = 0.0
    mask[drop] = 0.0
    return x


def train_encoder(Xtr, ytr, nc, rng, epochs=60, bs=512, seed=0):
    torch.manual_seed(seed)
    enc = Encoder(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.tensor(Xtr, device=DEV)                     # GPU-resident
    y = torch.tensor(ytr, device=DEV)
    n = len(Xtr)
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            p = float(rng.uniform(0.05, 0.4))             # simulate a range of sensitivities
            xb = cell_dropout_gpu(X[b], nc, p)
            opt.zero_grad()
            loss = supcon(enc(xb), y[b])
            loss.backward()
            opt.step()
    enc.eval()
    return enc


@torch.no_grad()
def embed(enc, X):
    return enc(torch.tensor(X, device=DEV)).cpu().numpy()


def prototypes(Z, y, nrp):
    p = np.zeros((nrp, Z.shape[1])); c = np.zeros(nrp)
    for z, lab in zip(Z, y):
        p[lab] += z; c[lab] += 1
    p = p / np.maximum(c[:, None], 1)
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)


def finetune(base_state, Xa, ya, Xc, yc, rng, steps=30, lr=2e-4):
    """Few-shot device adaptation: fine-tune from the base encoder on the device's K
    anchor samples MIXED with crowd samples (supcon), aligning the device to the crowd
    at shared RPs while preserving the learned location structure."""
    enc = Encoder(Xa.shape[1]).to(DEV)
    enc.load_state_dict(base_state)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    Xa_t = torch.tensor(Xa, device=DEV); ya_t = torch.tensor(ya, device=DEV)
    Xc_t = torch.tensor(Xc, device=DEV); yc_t = torch.tensor(yc, device=DEV)
    nc = len(Xc)
    for _ in range(steps):
        cb = torch.tensor(rng.choice(nc, min(256, nc), replace=False), device=DEV)
        x = torch.cat([Xa_t, Xc_t[cb]]); y = torch.cat([ya_t, yc_t[cb]])
        opt.zero_grad()
        loss = supcon(enc(x), y)
        loss.backward()
        opt.step()
    enc.eval()
    return enc


def loc_knn(qz, db_z, db_xy, knn=5):
    d = np.linalg.norm(db_z - qz, axis=1)
    idx = np.argsort(d)[:knn]
    w = 1.0 / (d[idx] + 1e-6)
    return (w[:, None] * db_xy[idx]).sum(0) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "localization_learned.json"))
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    print(f"[learned] device={DEV}")
    rng = np.random.default_rng(0)
    anchors = [0, 3, 5, 10]
    agg_learned, agg_ceiling = [], []
    agg_anchor = {k: [] for k in anchors}
    per_floor = {}

    for site, fl in SF:
        fd = load_floor(site, fl)
        if fd is None:
            continue
        nc = len(fd.cells)
        ridx = {r: i for i, r in enumerate(fd.rps)}
        samples = build_samples(fd, fd.phones, rng)
        floor_learned, floor_ceiling = [], []
        floor_anchor = {k: [] for k in anchors}
        for t in fd.phones:
            ref = [p for p in fd.phones if p != t and p in samples]
            if t not in samples or len(ref) < 2:
                continue
            Xtr = np.vstack([samples[p][0] for p in ref])
            ytr = np.concatenate([samples[p][1] for p in ref])
            enc = train_encoder(Xtr, ytr, nc, rng, epochs=args.epochs)

            # crowd map: prototype embedding per RP (mean over ref-phone samples)
            proto = np.zeros((len(fd.rps), 32)); cnt = np.zeros(len(fd.rps))
            Zref = embed(enc, Xtr)
            for z, lab in zip(Zref, ytr):
                proto[lab] += z; cnt[lab] += 1
            proto = proto / np.maximum(cnt[:, None], 1)
            proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

            Xte, yte = samples[t]
            Zte = embed(enc, Xte)
            errs = [np.linalg.norm(loc_knn(z, proto, fd.xy) - fd.xy[lab]) for z, lab in zip(Zte, yte)]
            floor_learned.append(float(np.median(errs)))

            # few-anchor self-calibration via few-shot FINE-TUNING (gradient adaptation)
            d = Zte.shape[1]
            te_cnt = np.zeros(len(fd.rps))
            for lab in yte:
                te_cnt[lab] += 1
            avail = [i for i in range(len(fd.rps)) if te_cnt[i] > 0]
            base_state = {kk: vv.detach().clone() for kk, vv in enc.state_dict().items()}
            for K in anchors:
                if K == 0:
                    floor_anchor[K].append(floor_learned[-1]); continue
                rep = []
                for _ in range(4):
                    anc = list(rng.choice(avail, min(K, len(avail)), replace=False))
                    amask = np.isin(yte, anc)
                    Xa, ya = Xte[amask], yte[amask]
                    if len(np.unique(ya)) < 2:
                        continue
                    enc_a = finetune(base_state, Xa, ya, Xtr, ytr, rng)
                    proto_a = prototypes(embed(enc_a, Xtr), ytr, len(fd.rps))
                    Zte_a = embed(enc_a, Xte)
                    for z, lab in zip(Zte_a, yte):
                        if lab in anc:
                            continue
                        rep.append(np.linalg.norm(loc_knn(z, proto_a, fd.xy) - fd.xy[lab]))
                floor_anchor[K].append(float(np.median(rep)) if rep else float("nan"))

            # same-device ceiling in embedding space (shuffle: samples are RP-ordered)
            perm = rng.permutation(len(Zte))
            a_idx, b_idx = perm[: len(perm) // 2], perm[len(perm) // 2:]
            proto_self = np.zeros((len(fd.rps), d)); cs = np.zeros(len(fd.rps))
            for i in a_idx:
                proto_self[yte[i]] += Zte[i]; cs[yte[i]] += 1
            proto_self = proto_self / np.maximum(cs[:, None], 1)
            proto_self = proto_self / (np.linalg.norm(proto_self, axis=1, keepdims=True) + 1e-8)
            ce = [np.linalg.norm(loc_knn(Zte[i], proto_self, fd.xy) - fd.xy[yte[i]])
                  for i in b_idx if cs[yte[i]] > 0]
            floor_ceiling.append(float(np.median(ce)) if ce else float("nan"))

        key = f"{site}_f{fl}"
        per_floor[key] = {"learned": round(float(np.nanmean(floor_learned)), 2),
                          "ceiling": round(float(np.nanmean(floor_ceiling)), 2),
                          "anchor": {k: round(float(np.nanmean(floor_anchor[k])), 2) for k in anchors}}
        agg_learned += floor_learned; agg_ceiling += floor_ceiling
        for k in anchors:
            agg_anchor[k] += floor_anchor[k]
        print(f"[{key}] learned={per_floor[key]['learned']} ceiling={per_floor[key]['ceiling']} "
              f"anchor={per_floor[key]['anchor']}")

    def m(v):
        a = np.array([x for x in v if np.isfinite(x)]); return round(float(a.mean()), 2) if len(a) else None

    out = {"per_floor": per_floor, "n_phones": len(agg_learned),
           "pooled": {"learned": m(agg_learned), "ceiling": m(agg_ceiling),
                      "anchor": {k: m(agg_anchor[k]) for k in anchors}}}
    Path(args.results).write_text(json.dumps(out, indent=2))
    p = out["pooled"]
    print("\n===== LEARNED device-invariant localization (per-phone median err, m) =====")
    print(f"  learned (0 anchor)  : {p['learned']}")
    for k in anchors:
        print(f"  + {k:2d} anchors        : {p['anchor'][k]}")
    print(f"  same-device ceiling : {p['ceiling']}")
    print(f"\n[learned] -> {args.results}")


if __name__ == "__main__":
    main()

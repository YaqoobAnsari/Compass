#!/usr/bin/env python
"""
Phase 2.1 — the make-or-break experiment: does trajectory ORDER carry usable
signal, and does it matter ONLY when measurement noise is spatially CORRELATED?

Controlled denoising task: given a trajectory's ordered per-point features +
*measured* RSS (corrupted), recover the *clean* RSS. We compare, per noise
regime (correlated shadowing vs i.i.d.), an order-aware BiGRU under {true,
shuffled, reversed} order against a permutation-invariant Set control. The
correlated and i.i.d. regimes share the SAME trajectories, clean targets,
device offsets and fast-fade draws — only the spatial correlation of the
shadowing residual differs, so any GRU(true) vs GRU(shuffled) gap is the value
of order, and the corr-vs-iid contrast is the Pillar-1 / Pillar-2 symbiosis.

Run:  python scripts/exp_order_matters.py          # ~few min on CPU
Outputs: results/order_matters.json, figures/experiments/order_matters.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids  # noqa: E402
from compass.data.conditioning import dbm_to_signed_unit, sample_clean_dbm, trajectory_features  # noqa: E402
from compass.data.conventions import PATHLOSS_RANGE_DB  # noqa: E402
from compass.data.noise import correlated_shadowing_field  # noqa: E402
from compass.data.trajectory import TrajectorySampler  # noqa: E402
from compass.models.sequence import GRUDenoiser, SetDenoiser  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DB_PER_UNIT = PATHLOSS_RANGE_DB / 2.0  # 1 normalised unit == 69.5 dB
REF = -130.0
FLOOR = -186.0
QUANT = 1.0


# ----------------------------------------------------------------------------
def build_sequences(root, map_ids, variant, k, n_points, sampler, cfg, master_seed):
    """Generate per-trajectory (geometry, measured_corr, measured_iid, clean) arrays."""
    ds = RadioMapSeerDataset(root, map_ids=map_ids, variant=variant)
    rng = np.random.default_rng(master_seed)
    geom, m_corr, m_iid, clean = [], [], [], []
    for map_id in map_ids:
        building = ds.load_building(map_id)
        shadow = correlated_shadowing_field(sigma_db=cfg["sigma_shadow"], decorr_m=cfg["decorr_m"], rng=rng)
        for tx in range(cfg["tx_per_map"]):
            sample = ds[ds.samples.index((map_id, tx))]
            dbm = sample["radio_map_dbm"].numpy()[0]
            tx_rc = tuple(int(v) for v in sample["tx_rowcol"].numpy())
            for tr in sampler.sample_many(building, k=k, n_points=n_points):
                ri = np.clip(np.round(tr.rows).astype(int), 0, 255)
                ci = np.clip(np.round(tr.cols).astype(int), 0, 255)
                cl = sample_clean_dbm(dbm, tr.rows, tr.cols)
                g_feats = trajectory_features(tr, tx_rc, cl)[:, :10]  # geometry only (drop rss col)
                tr_rng = np.random.default_rng(rng.integers(1 << 31))
                b = tr_rng.normal(0, cfg["device_offset_std"])
                gain = 1.0 + tr_rng.normal(0, cfg["device_gain_std"])
                fast = tr_rng.normal(0, cfg["fast_db"], size=cl.shape)
                base = gain * (cl - REF) + REF + fast + b
                corr_resid = shadow[ri, ci]
                iid_resid = tr_rng.normal(0, cfg["sigma_shadow"], size=cl.shape)
                mc = np.maximum(np.round((base + corr_resid) / QUANT) * QUANT, FLOOR)
                mi = np.maximum(np.round((base + iid_resid) / QUANT) * QUANT, FLOOR)
                geom.append(g_feats)
                m_corr.append(dbm_to_signed_unit(mc))
                m_iid.append(dbm_to_signed_unit(mi))
                clean.append(dbm_to_signed_unit(cl))
    return (np.asarray(geom, np.float32), np.asarray(m_corr, np.float32),
            np.asarray(m_iid, np.float32), np.asarray(clean, np.float32))


def make_inputs(geom, meas):
    """(S,N,10) geometry + measured RSS channel -> (S,N,11) tensor."""
    return torch.from_numpy(np.concatenate([geom, meas[..., None]], axis=-1).astype(np.float32))


def _perm(X, y, order, gen):
    if order == "true":
        return X, y
    B, N = y.shape
    if order == "reversed":
        idx = torch.arange(N - 1, -1, -1, device=X.device).expand(B, N)
    else:  # shuffled — independent permutation per sequence (gen is CPU -> move to X.device)
        idx = torch.argsort(torch.rand(B, N, generator=gen), dim=1).to(X.device)
    Xp = torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, X.size(-1)))
    yp = torch.gather(y, 1, idx)
    return Xp, yp


def run_cell(Xtr, ytr, Xva, yva, kind, order, epochs, seed, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    in_dim = Xtr.size(-1)
    model = (GRUDenoiser(in_dim) if kind == "gru" else SetDenoiser(in_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    n, bs = Xtr.size(0), 64
    for _ in range(epochs):
        model.train()
        for i in torch.randperm(n, generator=gen).split(bs):
            xb, yb = _perm(Xtr[i].to(device), ytr[i].to(device), order, gen)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xva, yva2 = _perm(Xva.to(device), yva.to(device), order, gen)
        rmse_norm = torch.sqrt(loss_fn(model(xva), yva2)).item()
    return rmse_norm * DB_PER_UNIT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--variant", default="IRT2")
    ap.add_argument("--train-maps", type=int, default=50)
    ap.add_argument("--val-maps", type=int, default=12)
    ap.add_argument("--tx-per-map", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sigma-shadow", type=float, default=6.0)
    ap.add_argument("--decorr-m", type=float, default=30.0)
    ap.add_argument("--device-offset-std", type=float, default=4.0)
    ap.add_argument("--device-gain-std", type=float, default=0.05)
    ap.add_argument("--fast-db", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=str(REPO / "results" / "order_matters_data.npz"))
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--results", default=str(REPO / "results" / "order_matters.json"))
    ap.add_argument("--fig", default=str(REPO / "figures" / "experiments" / "order_matters.png"))
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in
           ["sigma_shadow", "decorr_m", "device_offset_std", "device_gain_std", "fast_db", "tx_per_map"]}
    splits = split_map_ids(list_map_ids(args.root))
    train_maps = splits["train"][: args.train_maps]
    val_maps = splits["val"][: args.val_maps]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(args.cache)
    if cache.exists() and not args.regen:
        d = np.load(cache)
        gtr, ctr, itr, ytr_ = d["gtr"], d["ctr"], d["itr"], d["ytr"]
        gva, cva, iva, yva_ = d["gva"], d["cva"], d["iva"], d["yva"]
        print(f"[exp] loaded cached sequences: train={gtr.shape[0]} val={gva.shape[0]}")
    else:
        t0 = time.time()
        samp = TrajectorySampler(seed=args.seed)
        gtr, ctr, itr, ytr_ = build_sequences(args.root, train_maps, args.variant, args.k, args.n_points, samp, cfg, args.seed)
        gva, cva, iva, yva_ = build_sequences(args.root, val_maps, args.variant, args.k, args.n_points, samp, cfg, args.seed + 1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, gtr=gtr, ctr=ctr, itr=itr, ytr=ytr_, gva=gva, cva=cva, iva=iva, yva=yva_)
        print(f"[exp] generated train={gtr.shape[0]} val={gva.shape[0]} sequences in {time.time()-t0:.0f}s")

    ytr = torch.from_numpy(ytr_)
    yva = torch.from_numpy(yva_)
    regimes = {
        "correlated": (make_inputs(gtr, ctr), make_inputs(gva, cva)),
        "iid": (make_inputs(gtr, itr), make_inputs(gva, iva)),
    }
    cells = [("gru", "true"), ("gru", "shuffled"), ("gru", "reversed"), ("set", "true")]

    table = {}
    for regime, (Xtr, Xva) in regimes.items():
        table[regime] = {}
        for kind, order in cells:
            label = f"{kind}_{order}" if not (kind == "set") else "set"
            rmse = run_cell(Xtr, ytr, Xva, yva, kind, order, args.epochs, args.seed, device)
            table[regime][label] = round(rmse, 3)
            print(f"  [{regime:11s}] {label:14s} val RMSE = {rmse:.2f} dB")

    def gap(regime):
        return round(table[regime]["set"] - table[regime]["gru_true"], 3), \
               round(table[regime]["gru_shuffled"] - table[regime]["gru_true"], 3)

    corr_vs_set, corr_vs_shuf = gap("correlated")
    iid_vs_set, iid_vs_shuf = gap("iid")
    # Order helps if, under CORRELATED noise, true order beats shuffled by a margin,
    # AND that advantage is much smaller under IID noise (the symbiosis).
    order_helps = (corr_vs_shuf > 0.3) and (corr_vs_shuf > 2 * max(iid_vs_shuf, 0.05))

    summary = {
        "config": vars(args),
        "device": device,
        "rmse_db": table,
        "gaps_db": {
            "correlated": {"gru_true_beats_set": corr_vs_set, "gru_true_beats_shuffled": corr_vs_shuf},
            "iid": {"gru_true_beats_set": iid_vs_set, "gru_true_beats_shuffled": iid_vs_shuf},
        },
        "order_helps_under_correlated_noise": bool(order_helps),
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # --- figure ---
    labels = ["gru_true", "gru_shuffled", "gru_reversed", "set"]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w / 2, [table["correlated"][l] for l in labels], w, label="correlated noise", color="#c0392b")
    ax.bar(x + w / 2, [table["iid"][l] for l in labels], w, label="i.i.d. noise", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("val RMSE (dB) — lower is better")
    ax.set_title(f"Does order matter?  GRU(true) beats GRU(shuffled) by "
                 f"{corr_vs_shuf:.2f} dB (corr) vs {iid_vs_shuf:.2f} dB (iid)")
    ax.legend()
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[exp] CORRELATED: GRU(true) beats GRU(shuffled) by {corr_vs_shuf:.2f} dB, beats Set by {corr_vs_set:.2f} dB")
    print(f"[exp] IID:        GRU(true) beats GRU(shuffled) by {iid_vs_shuf:.2f} dB, beats Set by {iid_vs_set:.2f} dB")
    print(f"[exp] ORDER HELPS (under correlated noise, not under iid): {order_helps}")
    print(f"[exp] results -> {out}\n[exp] figure -> {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

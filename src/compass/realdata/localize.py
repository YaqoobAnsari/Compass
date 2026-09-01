"""
Cross-device indoor fingerprint localization from sparse crowdsensed cellular RSS.

The device-heterogeneity in this data is dominated by DETECTION heterogeneity: at the
same place, two phones share only ~30% of the cells they detect (receiver-sensitivity
differences), and the RSS magnitude is corrupted enough that naive RSS matching is worse
than binary heard-set matching. The classic cross-device toolbox (mean-normalization,
scalar/per-cell offset calibration, HLF/SSD) targets RSS *offset* and fails here.

This module implements the fingerprint data structures, the baseline matchers, and our
detection-aware matcher (reliability-weighted, sensitivity-normalized), plus a
leave-one-phone-out evaluation harness with a same-device oracle ceiling.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import unicellular as U

warnings.filterwarnings("ignore", category=RuntimeWarning)

NOT_HEARD = -110.0
M_PER_PX = 0.032
RSS = getattr(U, "RSS_COL", "transmitter_rss")
TX = getattr(U, "TX_COL", "transmitter_id")
PHONE = getattr(U, "PHONE_COL", "phoneName")


@dataclass
class FloorData:
    site: str
    floor: int
    phones: List[str]
    rps: List[int]
    xy: np.ndarray                       # (n_rp, 2) metres
    cells: list
    mats: Dict                           # (phone, rp) -> (n_scans, n_cells), NaN=unheard


def load_floor(site: str, floor: int) -> FloorData | None:
    df = U.load_floor(site, "stationary", floor, usecols=U.RECON_COLS)
    v = U.valid_measurements(df)
    v = v[v["rpNumber"] >= 1]
    coords = U.load_rp_coords(site, floor)
    if not coords:
        return None
    v = v[v["rpNumber"].astype(int).isin(coords.keys())].copy()
    v["rp"] = v["rpNumber"].astype(int)
    v[PHONE] = v[PHONE].astype(str)
    cells = sorted(v[TX].unique())
    cidx = {c: i for i, c in enumerate(cells)}
    phones = sorted(v[PHONE].unique())
    rps = sorted(coords.keys())
    nc = len(cells)
    mats: Dict = {}
    for (p, r), sub in v.groupby([PHONE, "rp"]):
        pt = sub.pivot_table(index="scanNumber", columns=TX, values=RSS, aggfunc="mean")
        m = np.full((pt.shape[0], nc), np.nan)
        for c in pt.columns:
            if c in cidx:
                m[:, cidx[c]] = pt[c].to_numpy()
        mats[(p, int(r))] = m
    for p in phones:
        for r in rps:
            mats.setdefault((p, r), np.full((0, nc), np.nan))
    xy = np.array([np.array(coords[r], float) * M_PER_PX for r in rps])
    return FloorData(site, floor, phones, rps, xy, cells, mats)


def fp(mat: np.ndarray, rows=None) -> np.ndarray:
    """Aggregate scan rows -> dense fingerprint (mean RSS per cell, NOT_HEARD if unheard)."""
    if mat.shape[0] == 0:
        return np.full(mat.shape[1], NOT_HEARD)
    m = np.nanmean(mat if rows is None else mat[rows], axis=0)
    return np.where(np.isnan(m), NOT_HEARD, m)


def crowd_db(fd: FloorData, ref: List[str]) -> np.ndarray:
    """Pooled reference-phone fingerprint per RP -> (n_rp, n_cells)."""
    return np.array([fp(np.vstack([fd.mats[(p, r)] for p in ref])) for r in fd.rps])


def reliability(fd: FloorData, ref: List[str]) -> np.ndarray:
    """Per-cell cross-device detection reliability in [0,1]: at RPs where the crowd hears
    a cell, the average fraction of reference phones that also hear it. High = device-robust."""
    nr, nc = len(fd.rps), len(fd.cells)
    H = np.zeros((len(ref), nr, nc), bool)
    for pi, p in enumerate(ref):
        for ri, r in enumerate(fd.rps):
            H[pi, ri] = fp(fd.mats[(p, r)]) > NOT_HEARD + 1e-6
    crowd = H.any(0)                                  # (nr, nc)
    rate = H.mean(0)                                  # (nr, nc)
    num = (rate * crowd).sum(0)
    den = np.maximum(crowd.sum(0), 1e-9)
    return num / den


# ----------------------------------------------------------------- representations
def _heard(x):
    return x > NOT_HEARD + 1e-6


def _ranks(x):
    """Descending RSS rank among heard cells (1=strongest); 0 for unheard. Offset-invariant."""
    r = np.zeros_like(x)
    h = _heard(x)
    if h.sum():
        order = np.argsort(-x[h])
        rk = np.empty(h.sum()); rk[order] = np.arange(1, h.sum() + 1)
        r[h] = rk
    return r


# ----------------------------------------------------------------- distance functions
# each returns distances to every DB RP: shape (n_rp,)
def d_euclid(q, db, ctx):
    return np.linalg.norm(db - q, axis=1)


def d_meannorm(q, db, ctx):
    def mn(v):
        h = _heard(v); o = v.copy()
        if h.sum():
            o[h] = o[h] - o[h].mean()
        return o
    qn = mn(q)
    dbn = ctx["db_mn"]
    return np.linalg.norm(dbn - qn, axis=1)


def d_binary(q, db, ctx):
    hq = _heard(q).astype(float)
    hdb = _heard(db).astype(float)
    inter = (hdb * hq).sum(1)
    union = np.maximum((hdb + hq > 0).sum(1), 1)
    return 1.0 - inter / union


def d_rank(q, db, ctx):
    qr = _ranks(q)
    dbr = ctx["db_rank"]
    shared = _heard(q) & _heard(db)
    cnt = shared.sum(1)
    diff = (np.abs(dbr - qr) * shared).sum(1)
    d = np.where(cnt > 0, diff / np.maximum(cnt, 1), 1e6)
    return d + 5.0 * (1.0 - cnt / max(_heard(q).sum(), 1))     # + mild set-mismatch penalty


def d_wjaccard(q, db, ctx):
    w = ctx["w"]
    hq = _heard(q); hdb = _heard(db)
    both = (hdb & hq) * w
    either = (hdb | hq) * w
    return 1.0 - both.sum(1) / np.maximum(either.sum(1), 1e-9)


def d_hybrid(q, db, ctx, lam=0.5):
    """Detection-aware: reliability-weighted heard-set overlap + offset-invariant rank
    agreement on shared cells. Targets detection heterogeneity, not RSS offset."""
    dj = d_wjaccard(q, db, ctx)                          # in [0,1]
    dr = d_rank(q, db, ctx)
    dr = dr / (dr.max() + 1e-9)                           # normalize to ~[0,1]
    return (1 - lam) * dj + lam * dr


def d_sensmatch(q, db, ctx):
    """Sensitivity-normalized weighted Jaccard: compare the query only against the
    strongest ~n_q DB cells (a device of the query's sensitivity would hear those),
    aligning heard-set sizes before reliability-weighted overlap."""
    w = ctx["w"]
    nq = max(int(_heard(q).sum()), 1)
    # top-nq strongest cells per DB rp
    hq = _heard(q)
    order = np.argsort(-db, axis=1)
    keep = np.zeros_like(db, bool)
    rows = np.arange(db.shape[0])[:, None]
    keep[rows, order[:, :nq]] = True
    hdb = keep & _heard(db)
    both = (hdb & hq) * w
    either = (hdb | hq) * w
    return 1.0 - both.sum(1) / np.maximum(either.sum(1), 1e-9)


METHODS = {
    "naive": d_euclid, "meannorm": d_meannorm, "binary": d_binary,
    "rank": d_rank, "wjaccard": d_wjaccard, "hybrid": d_hybrid, "sensmatch": d_sensmatch,
}


def _localize(q, db, xy, distfn, ctx, knn=3):
    d = distfn(q, db, ctx)
    idx = np.argsort(d)[:knn]
    w = 1.0 / (d[idx] + 1e-6)
    return (w[:, None] * xy[idx]).sum(0) / w.sum()


def evaluate_floor(fd: FloorData, rng, methods=None, k=12, n_query=20, knn=3):
    """Leave-one-phone-out. Returns {method: [per-phone median error]} plus 'ceiling'."""
    methods = methods or list(METHODS)
    rps, xy = fd.rps, fd.xy
    per = {m: [] for m in methods}
    per["ceiling"] = []
    for t in fd.phones:
        ref = [p for p in fd.phones if p != t]
        db = crowd_db(fd, ref)
        ctx = {"w": reliability(fd, ref),
               "db_mn": np.array([_mn(f) for f in db]),
               "db_rank": np.array([_ranks(f) for f in db])}
        errs = {m: [] for m in methods}
        for ri, r in enumerate(rps):
            n = fd.mats[(t, r)].shape[0]
            if n == 0:
                continue
            for _ in range(n_query):
                q = fp(fd.mats[(t, r)], rng.choice(n, min(k, n), replace=False))
                for m in methods:
                    pred = _localize(q, db, xy, METHODS[m], ctx, knn)
                    errs[m].append(np.linalg.norm(pred - xy[ri]))
        for m in methods:
            per[m].append(float(np.median(errs[m])) if errs[m] else float("nan"))

        # same-device ceiling
        db_self = np.array([fp(fd.mats[(t, rr)][: max(1, fd.mats[(t, rr)].shape[0] // 2)]) for rr in rps])
        ce = []
        for ri, r in enumerate(rps):
            n = fd.mats[(t, r)].shape[0]
            if n < 2:
                continue
            qrows = np.arange(n // 2, n)
            for _ in range(n_query):
                q = fp(fd.mats[(t, r)], rng.choice(qrows, min(k, len(qrows)), replace=False))
                ce.append(np.linalg.norm(_localize(q, db_self, xy, d_euclid, {}, knn) - xy[ri]))
        per["ceiling"].append(float(np.median(ce)) if ce else float("nan"))
    return per


def _mn(v):
    h = _heard(v); o = v.copy()
    if h.sum():
        o[h] = o[h] - o[h].mean()
    return o


# ============================ learned device-invariant embedding ============================
RSS_MID, RSS_SCALE = -85.0, 15.0


def featurize(vec):
    heard = _heard(vec)
    f_rss = np.where(heard, (vec - RSS_MID) / RSS_SCALE, 0.0)
    return np.concatenate([f_rss, heard.astype(np.float32)])


def build_samples(fd: FloorData, phones, rng, k=12, per_rp=30):
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
                feats.append(featurize(fp(mat, rng.choice(n, min(k, n), replace=False))))
                labels.append(ridx[r])
        if feats:
            out[p] = (np.array(feats, np.float32), np.array(labels))
    return out


def learned_eval_floor(fd: FloorData, rng, epochs=60, aug=True):
    """LOPO learned device-invariant localization; returns list of per-phone median error (m)
    and the embedding same-device ceiling. Requires torch (import guarded).
    aug=False disables detection-dropout augmentation (ablation)."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"

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
        m = torch.eye(z.shape[0], device=z.device).bool()
        sim = sim.masked_fill(m, -1e9)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        pos = (y[:, None] == y[None, :]) & ~m
        return -(logp.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1)).mean()

    def dropout_gpu(x, nc, p):
        x = x.clone(); mask = x[:, nc:]
        drop = (torch.rand_like(mask) < p) & (mask > 0.5)
        x[:, :nc][drop] = 0.0; mask[drop] = 0.0
        return x

    def train(Xtr, ytr, nc):
        torch.manual_seed(0)
        enc = Encoder(Xtr.shape[1]).to(dev)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-4)
        X = torch.tensor(Xtr, device=dev); y = torch.tensor(ytr, device=dev)
        for _ in range(epochs):
            idx = torch.randperm(len(Xtr), device=dev)
            for s in range(0, len(Xtr), 512):
                b = idx[s:s + 512]
                xb = dropout_gpu(X[b], nc, float(rng.uniform(0.05, 0.4))) if aug else X[b]
                opt.zero_grad()
                supcon(enc(xb), y[b]).backward()
                opt.step()
        enc.eval()
        return enc

    @torch.no_grad()
    def emb(enc, X):
        return enc(torch.tensor(X, device=dev)).cpu().numpy()

    def protos(Z, y, nrp):
        p = np.zeros((nrp, Z.shape[1])); c = np.zeros(nrp)
        for z, lab in zip(Z, y):
            p[lab] += z; c[lab] += 1
        p = p / np.maximum(c[:, None], 1)
        return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)

    nc = len(fd.cells)
    samples = build_samples(fd, fd.phones, rng)
    per, ceil = [], []
    for t in fd.phones:
        ref = [p for p in fd.phones if p != t and p in samples]
        if t not in samples or len(ref) < 2:
            continue
        Xtr = np.vstack([samples[p][0] for p in ref])
        ytr = np.concatenate([samples[p][1] for p in ref])
        enc = train(Xtr, ytr, nc)
        proto = protos(emb(enc, Xtr), ytr, len(fd.rps))
        Xte, yte = samples[t]
        Zte = emb(enc, Xte)
        per.append(float(np.median([np.linalg.norm(_knn(z, proto, fd.xy) - fd.xy[lab])
                                    for z, lab in zip(Zte, yte)])))
        perm = rng.permutation(len(Zte)); h = len(perm) // 2
        ps = protos(Zte[perm[:h]], yte[perm[:h]], len(fd.rps)); cs = np.zeros(len(fd.rps))
        for lab in yte[perm[:h]]:
            cs[lab] += 1
        ce = [np.linalg.norm(_knn(Zte[i], ps, fd.xy) - fd.xy[yte[i]]) for i in perm[h:] if cs[yte[i]] > 0]
        ceil.append(float(np.median(ce)) if ce else float("nan"))
    return per, ceil


def _knn(z, db, xy, knn=5):
    d = np.linalg.norm(db - z, axis=1)
    idx = np.argsort(d)[:knn]
    w = 1.0 / (d[idx] + 1e-6)
    return (w[:, None] * xy[idx]).sum(0) / w.sum()

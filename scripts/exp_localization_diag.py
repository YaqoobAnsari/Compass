#!/usr/bin/env python
"""
MECHANISM DIAGNOSTIC for the cross-device localization penalty.

The gate showed a 5.4 m device penalty that neither mean-normalization nor scalar
per-device calibration fixes. This decomposes WHY, to determine the method:
  - per-cell anchor calibration  (is the offset per-cell rather than global?)
  - common-cell distance         (does sentinel-dominated matching hide magnitude info?)
  - binary heard-set matching     (how much penalty is 'different phones hear different cells'?)
  - cross-device heard-set Jaccard + per-(phone,cell) offset consistency (descriptive)

  bash scripts/submit.sh 1g.18gb python scripts/exp_localization_diag.py   (CPU ok, no --gres)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from compass.realdata import unicellular as U

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[1]
M_PER_PX, NOT_HEARD = 0.032, -110.0
RSS = getattr(U, "RSS_COL", "transmitter_rss")
TX = getattr(U, "TX_COL", "transmitter_id")
PHONE = getattr(U, "PHONE_COL", "phoneName")
SF = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1)]


def load_floor(site, fl):
    df = U.load_floor(site, "stationary", fl, usecols=U.RECON_COLS)
    v = U.valid_measurements(df)
    v = v[v["rpNumber"] >= 1]
    coords = U.load_rp_coords(site, fl)
    if not coords:
        return None
    v = v[v["rpNumber"].astype(int).isin(coords.keys())].copy()
    v["rp"] = v["rpNumber"].astype(int)
    v[PHONE] = v[PHONE].astype(str)
    return v, coords


def build_mats(v, phones, rps, cidx):
    nc = len(cidx)
    mats = {}
    for (p, r), sub in v.groupby([PHONE, "rp"]):
        pt = sub.pivot_table(index="scanNumber", columns=TX, values=RSS, aggfunc="mean")
        mat = np.full((pt.shape[0], nc), np.nan)
        for c in pt.columns:
            if c in cidx:
                mat[:, cidx[c]] = pt[c].to_numpy()
        mats[(p, int(r))] = mat
    for p in phones:
        for r in rps:
            mats.setdefault((p, r), np.full((0, nc), np.nan))
    return mats


def fp_rows(mat, rows=None):
    if mat.shape[0] == 0:
        return np.full(mat.shape[1], NOT_HEARD)
    m = np.nanmean(mat if rows is None else mat[rows], axis=0)
    return np.where(np.isnan(m), NOT_HEARD, m)


def knn_idx(dists, knn=3):
    return np.argsort(dists)[:knn]


def loc_from_idx(idx, dists, rp_arr):
    w = 1.0 / (dists[idx] + 1e-6)
    return (w[:, None] * rp_arr[idx]).sum(0) / w.sum()


def euclid(q, db):
    return np.linalg.norm(db - q, axis=1)


def common_dist(q, db):
    hq = q > NOT_HEARD + 1e-6
    hdb = db > NOT_HEARD + 1e-6
    both = hdb & hq
    cnt = both.sum(1)
    diff = np.where(both, np.abs(db - q), 0.0).sum(1)
    d = np.where(cnt > 0, diff / np.maximum(cnt, 1), 1e6)
    return d


def binary_dist(q, db):
    hq = (q > NOT_HEARD + 1e-6).astype(float)
    hdb = (db > NOT_HEARD + 1e-6).astype(float)
    inter = (hdb * hq).sum(1)
    union = np.maximum((hdb + hq > 0).sum(1), 1)
    return 1.0 - inter / union            # Jaccard distance


def run_floor(mats, phones, rps, coords, rng, k=12, n_query=15, knn=3, anchors=(5, 10)):
    xy = {r: np.array(coords[r], float) * M_PER_PX for r in rps}
    rp_arr = np.array([xy[r] for r in rps])
    ridx = {r: i for i, r in enumerate(rps)}
    full = {(p, r): fp_rows(mats[(p, r)]) for p in phones for r in rps}

    def query(p, r):
        n = mats[(p, r)].shape[0]
        return None if n == 0 else fp_rows(mats[(p, r)], rng.choice(n, min(k, n), replace=False))

    res = {c: [] for c in ("naive", "common", "binary", "percell5", "percell10", "common_pc10")}
    jacc, off_consistency = [], []
    for t in phones:
        ref = [p for p in phones if p != t]
        db = np.array([fp_rows(np.vstack([mats[(p, r)] for p in ref])) for r in rps])

        # heard-set Jaccard: test phone vs crowd DB at each rp
        for r in rps:
            hq = full[(t, r)] > NOT_HEARD + 1e-6
            hd = db[ridx[r]] > NOT_HEARD + 1e-6
            u = (hq | hd).sum()
            if u:
                jacc.append((hq & hd).sum() / u)

        # per-cell offset consistency: for test phone, per-cell offset vs db across rps;
        # consistency = 1 - (std/|mean|) proxy -> report std of per-cell offset across rps
        percell_off = {}   # cell -> list of (query-db) across rps where both heard
        for r in rps:
            q, d = full[(t, r)], db[ridx[r]]
            both = (q > NOT_HEARD + 1e-6) & (d > NOT_HEARD + 1e-6)
            for ci in np.where(both)[0]:
                percell_off.setdefault(ci, []).append(q[ci] - d[ci])
        stds = [np.std(o) for o in percell_off.values() if len(o) >= 3]
        if stds:
            off_consistency.append(float(np.median(stds)))

        # localization conditions
        en, ec, eb, e5, e10, ecp = [], [], [], [], [], []
        # precompute per-cell delta from anchors (K=10) once per repeat handled below
        for cond, store in (("naive", en), ("common", ec), ("binary", eb)):
            for r in rps:
                for _ in range(n_query):
                    q = query(t, r)
                    if q is None:
                        continue
                    if cond == "naive":
                        d = euclid(q, db)
                    elif cond == "common":
                        d = common_dist(q, db)
                    else:
                        d = binary_dist(q, db)
                    idx = knn_idx(d, knn)
                    store.append(np.linalg.norm(loc_from_idx(idx, d, rp_arr) - xy[r]))
        res["naive"].append(float(np.median(en)))
        res["common"].append(float(np.median(ec)))
        res["binary"].append(float(np.median(eb)))

        # per-cell anchor calibration (euclid) + common+per-cell
        for K, store, storec in ((5, e5, None), (10, e10, ecp)):
            errs, errsc = [], []
            for _ in range(6):
                anc = list(rng.choice(rps, min(K, len(rps)), replace=False))
                # per-cell delta averaged over anchors
                dsum = np.zeros(db.shape[1]); dcnt = np.zeros(db.shape[1])
                for a in anc:
                    q, d = full[(t, a)], db[ridx[a]]
                    both = (q > NOT_HEARD + 1e-6) & (d > NOT_HEARD + 1e-6)
                    dsum[both] += (q - d)[both]; dcnt[both] += 1
                delta_cell = np.where(dcnt > 0, dsum / np.maximum(dcnt, 1), 0.0)
                for r in rps:
                    if r in anc:
                        continue
                    for _ in range(max(1, n_query // 2)):
                        q = query(t, r)
                        if q is None:
                            continue
                        qc = q.copy()
                        h = qc > NOT_HEARD + 1e-6
                        qc[h] = qc[h] - delta_cell[h]
                        d = euclid(qc, db)
                        idx = knn_idx(d, knn)
                        errs.append(np.linalg.norm(loc_from_idx(idx, d, rp_arr) - xy[r]))
                        if storec is not None:
                            d2 = common_dist(qc, db)
                            i2 = knn_idx(d2, knn)
                            errsc.append(np.linalg.norm(loc_from_idx(i2, d2, rp_arr) - xy[r]))
            store.append(float(np.median(errs)) if errs else float("nan"))
            if storec is not None:
                storec.append(float(np.median(errsc)) if errsc else float("nan"))
        res["percell5"].append(e5[-1]); res["percell10"].append(e10[-1])
        res["common_pc10"].append(ecp[-1])
    res["_jaccard"] = float(np.mean(jacc)) if jacc else float("nan")
    res["_percell_offset_std"] = float(np.mean(off_consistency)) if off_consistency else float("nan")
    return res


def summ(vals):
    a = np.array([x for x in vals if np.isfinite(x)], float)
    return round(float(a.mean()), 2) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "localization_diag.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    agg = {c: [] for c in ("naive", "common", "binary", "percell5", "percell10", "common_pc10")}
    jaccs, offstds = [], []
    for site, fl in SF:
        loaded = load_floor(site, fl)
        if loaded is None:
            continue
        v, coords = loaded
        cells = sorted(v[TX].unique()); cidx = {c: i for i, c in enumerate(cells)}
        phones = sorted(v[PHONE].unique()); rps = sorted(coords.keys())
        mats = build_mats(v, phones, rps, cidx)
        r = run_floor(mats, phones, rps, coords, rng)
        for c in agg:
            agg[c] += r[c]
        jaccs.append(r["_jaccard"]); offstds.append(r["_percell_offset_std"])
        print(f"[{site}_f{fl}] naive {summ(r['naive'])} | common {summ(r['common'])} | "
              f"binary {summ(r['binary'])} | pc5 {summ(r['percell5'])} | pc10 {summ(r['percell10'])} | "
              f"common+pc10 {summ(r['common_pc10'])} | Jaccard {r['_jaccard']:.2f} | "
              f"per-cell off std {r['_percell_offset_std']:.1f} dB")
    out = {c: summ(agg[c]) for c in agg}
    out["heardset_jaccard_crossdevice"] = round(float(np.nanmean(jaccs)), 3)
    out["percell_offset_std_db"] = round(float(np.nanmean(offstds)), 2)
    Path(args.results).write_text(json.dumps(out, indent=2))
    print("\n===== POOLED (per-phone median position error, m) =====")
    for c in ("naive", "common", "binary", "percell5", "percell10", "common_pc10"):
        print(f"  {c:14s}: {out[c]}")
    print(f"  cross-device heard-set Jaccard : {out['heardset_jaccard_crossdevice']} "
          f"(1.0 = phones hear identical cell sets)")
    print(f"  per-cell offset std across RPs : {out['percell_offset_std_db']} dB "
          f"(low = consistent per-cell bias -> calibratable)")
    print(f"\n[diag] -> {args.results}")


if __name__ == "__main__":
    main()

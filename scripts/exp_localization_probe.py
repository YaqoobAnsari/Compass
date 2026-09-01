#!/usr/bin/env python
"""
GATE EXPERIMENT: cross-device indoor localization headroom on UniCellular.

Fingerprint kNN localization, leave-one-phone-out (map built from other phones,
localize the held-out phone). Answers three questions:
  1) device penalty  = naive cross-device error - same-device ceiling (meters)
  2) is it trivially fixed by mean-normalization (offset-invariant features)?
  3) how many known anchors does per-device self-calibration need?

Per-phone median position error is the reported unit (the honest unit is the device,
not the query), aggregated across phones; plus a same-device ceiling and a random
baseline. Aggregation is fully vectorized (scan x cell matrices, numpy nanmean).

  bash scripts/submit.sh 1g.18gb python scripts/exp_localization_probe.py   (CPU ok, no --gres)
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
M_PER_PX = 0.032
NOT_HEARD = -110.0
RSS = getattr(U, "RSS_COL", "transmitter_rss")
TX = getattr(U, "TX_COL", "transmitter_id")
PHONE = getattr(U, "PHONE_COL", "phoneName")
SITES_FLOORS = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1)]


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
    """(phone,rp) -> ndarray (n_scans, n_cells) of RSS with NaN where a cell is unheard."""
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


def meannorm(fp):
    heard = fp > NOT_HEARD + 1e-6
    out = fp.copy()
    if heard.sum():
        out[heard] = out[heard] - out[heard].mean()
    return out


def localize(q, db_fps, db_xy, knn=3):
    d = np.linalg.norm(db_fps - q, axis=1)
    idx = np.argsort(d)[:knn]
    w = 1.0 / (d[idx] + 1e-6)
    return (w[:, None] * db_xy[idx]).sum(0) / w.sum()


def run_floor(mats, phones, rps, coords, rng, k=12, n_query=20, knn=3,
              anchors=(1, 3, 5, 10), n_rep=8):
    xy = {r: np.array(coords[r], float) * M_PER_PX for r in rps}
    rp_arr = np.array([xy[r] for r in rps])
    ridx = {r: i for i, r in enumerate(rps)}
    full_fp = {(p, r): fp_rows(mats[(p, r)]) for p in phones for r in rps}

    def query(p, r):
        n = mats[(p, r)].shape[0]
        if n == 0:
            return None
        return fp_rows(mats[(p, r)], rng.choice(n, size=min(k, n), replace=False))

    out = {"phones": phones, "n_rp": len(rps), "same_device": [], "naive": [],
           "meannorm": [], "random": [], "anchor": {int(a): {"naive": [], "cal": []} for a in anchors}}

    for t in phones:
        ref = [p for p in phones if p != t]
        db = np.array([fp_rows(np.vstack([mats[(p, r)] for p in ref])) for r in rps])
        db_mn = np.array([meannorm(f) for f in db])

        en, em, erand = [], [], []
        for r in rps:
            for _ in range(n_query):
                q = query(t, r)
                if q is None:
                    continue
                en.append(np.linalg.norm(localize(q, db, rp_arr, knn) - xy[r]))
                em.append(np.linalg.norm(localize(meannorm(q), db_mn, rp_arr, knn) - xy[r]))
                erand.append(np.linalg.norm(rp_arr[rng.integers(len(rps))] - xy[r]))
        out["naive"].append(float(np.median(en)))
        out["meannorm"].append(float(np.median(em)))
        out["random"].append(float(np.median(erand)))

        # same-device ceiling: DB from first half of t's own scans, query the second half
        db_self = np.array([fp_rows(mats[(t, rr)][: max(1, mats[(t, rr)].shape[0] // 2)]) for rr in rps])
        es = []
        for r in rps:
            n = mats[(t, r)].shape[0]
            if n < 2:
                continue
            qrows = np.arange(n // 2, n)
            for _ in range(n_query):
                q = fp_rows(mats[(t, r)], rng.choice(qrows, size=min(k, len(qrows)), replace=False))
                es.append(np.linalg.norm(localize(q, db_self, rp_arr, knn) - xy[r]))
        out["same_device"].append(float(np.median(es)) if es else float("nan"))

        # anchor self-calibration (eval on non-anchor rps, paired naive)
        for K in anchors:
            enn, ecc = [], []
            for _ in range(n_rep):
                anc = list(rng.choice(rps, size=min(K, len(rps)), replace=False))
                deltas = []
                for a in anc:
                    q, dbf = full_fp[(t, a)], db[ridx[a]]
                    both = (q > NOT_HEARD + 1e-6) & (dbf > NOT_HEARD + 1e-6)
                    if both.sum():
                        deltas.append(float((q[both] - dbf[both]).mean()))
                delta = float(np.mean(deltas)) if deltas else 0.0
                for r in rps:
                    if r in anc:
                        continue
                    for _ in range(max(1, n_query // 2)):
                        q = query(t, r)
                        if q is None:
                            continue
                        enn.append(np.linalg.norm(localize(q, db, rp_arr, knn) - xy[r]))
                        qc = q.copy()
                        h = qc > NOT_HEARD + 1e-6
                        qc[h] = qc[h] - delta
                        ecc.append(np.linalg.norm(localize(qc, db, rp_arr, knn) - xy[r]))
            out["anchor"][int(K)]["naive"].append(float(np.median(enn)) if enn else float("nan"))
            out["anchor"][int(K)]["cal"].append(float(np.median(ecc)) if ecc else float("nan"))
    return out


def summ(vals):
    a = np.array([x for x in vals if np.isfinite(x)], float)
    if not len(a):
        return {}
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
            "median": round(float(np.median(a)), 2), "n": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "localization_probe.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    report = {"per_floor": {}}
    agg = {m: [] for m in ("same_device", "naive", "meannorm", "random")}
    agg["anchor"] = {a: {"naive": [], "cal": []} for a in (1, 3, 5, 10)}

    for site, fl in SITES_FLOORS:
        loaded = load_floor(site, fl)
        if loaded is None:
            print(f"[{site} f{fl}] skip"); continue
        v, coords = loaded
        cells = sorted(v[TX].unique())
        cidx = {c: i for i, c in enumerate(cells)}
        phones = sorted(v[PHONE].unique())
        rps = sorted(coords.keys())
        mats = build_mats(v, phones, rps, cidx)
        r = run_floor(mats, phones, rps, coords, rng)
        key = f"{site}_f{fl}"
        report["per_floor"][key] = {
            "n_phones": len(phones), "n_rp": len(rps),
            "same_device_m": summ(r["same_device"]), "naive_m": summ(r["naive"]),
            "meannorm_m": summ(r["meannorm"]), "random_m": summ(r["random"]),
            "anchor_m": {k: {"naive": summ(r["anchor"][k]["naive"]),
                             "cal": summ(r["anchor"][k]["cal"])} for k in r["anchor"]}}
        for m in agg:
            if m == "anchor":
                continue
            agg[m] += r[m]
        for k in r["anchor"]:
            agg["anchor"][k]["naive"] += r["anchor"][k]["naive"]
            agg["anchor"][k]["cal"] += r["anchor"][k]["cal"]
        print(f"[{key}] phones={len(phones)} RP={len(rps)} | same-dev "
              f"{summ(r['same_device']).get('mean')} | naive {summ(r['naive']).get('mean')} | "
              f"meannorm {summ(r['meannorm']).get('mean')} m")

    p = {"same_device_m": summ(agg["same_device"]), "naive_m": summ(agg["naive"]),
         "meannorm_m": summ(agg["meannorm"]), "random_m": summ(agg["random"]),
         "anchor_m": {k: {"naive": summ(agg["anchor"][k]["naive"]),
                          "cal": summ(agg["anchor"][k]["cal"])} for k in agg["anchor"]}}
    report["pooled"] = p
    report["device_penalty_m"] = round(p["naive_m"].get("mean", float("nan"))
                                       - p["same_device_m"].get("mean", float("nan")), 2)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(report, indent=2))

    print("\n===== POOLED (per-phone median position error, meters) =====")
    print(f"  random baseline      : {p['random_m'].get('mean')}")
    print(f"  naive cross-device   : {p['naive_m'].get('mean')} ± {p['naive_m'].get('std')}")
    print(f"  same-device ceiling  : {p['same_device_m'].get('mean')} ± {p['same_device_m'].get('std')}")
    print(f"  DEVICE PENALTY       : {report['device_penalty_m']} m  (naive - ceiling)")
    print(f"  mean-normalized      : {p['meannorm_m'].get('mean')} (offset-invariant, no anchors)")
    for k in (1, 3, 5, 10):
        a = p["anchor_m"][k]
        print(f"  anchor K={k:<2d} cal      : {a['cal'].get('mean')}  (paired naive {a['naive'].get('mean')})")
    print(f"\n[probe] -> {args.results}")


if __name__ == "__main__":
    main()

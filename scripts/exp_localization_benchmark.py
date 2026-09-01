#!/usr/bin/env python
"""
Cross-device indoor localization benchmark (UniCellular), leave-one-phone-out.

Baselines (RSS-offset toolbox + heard-set) vs our detection-aware matchers, against a
same-device oracle ceiling. Reports per-phone median position error (device = the unit),
bootstrap 95% CI over phones, and paired Wilcoxon of the best method vs baselines.

  bash scripts/submit.sh 1g.18gb python scripts/exp_localization_benchmark.py   (CPU ok)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compass.eval.stats import paired_wilcoxon
from compass.realdata.localize import METHODS, evaluate_floor, learned_eval_floor, load_floor

REPO = Path(__file__).resolve().parents[1]
SF = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1)]
ORDER = ["naive", "meannorm", "rank", "binary", "wjaccard", "hybrid", "learned", "ceiling"]
LABEL = {"naive": "naive RSS (Euclidean)", "meannorm": "mean-normalized RSS",
         "rank": "rank / SSD (offset-invariant)", "binary": "heard-set Jaccard",
         "wjaccard": "reliability-weighted set (ours)", "sensmatch": "sensitivity-matched (ours)",
         "hybrid": "detection-aware hybrid (ours)", "learned": "device-invariant embedding (ours)",
         "ceiling": "same-device oracle"}


def boot_ci(vals, n=2000, seed=0):
    a = np.array([x for x in vals if np.isfinite(x)], float)
    if len(a) < 2:
        return {"mean": round(float(a.mean()), 2) if len(a) else None}
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(n)])
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "lo": round(float(np.percentile(b, 2.5)), 2), "hi": round(float(np.percentile(b, 97.5)), 2),
            "n_phones": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "localization_benchmark.json"))
    ap.add_argument("--figdir", default=str(REPO / "figures" / "pub"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    methods = list(METHODS)
    agg = {m: [] for m in methods}
    agg["ceiling"] = []
    agg["learned"] = []
    per_floor = {}
    for site, fl in SF:
        fd = load_floor(site, fl)
        if fd is None:
            continue
        r = evaluate_floor(fd, rng, methods=methods)
        lr, lc = learned_eval_floor(fd, rng)
        r["learned"] = lr
        r["ceiling"] = lc                       # use the embedding same-device ceiling
        key = f"{site}_f{fl}"
        per_floor[key] = {m: round(float(np.nanmean(r[m])), 2) for m in r}
        for m in r:
            agg[m] += r[m]
        print(f"[{key}] " + " ".join(f"{m}={per_floor[key][m]}" for m in ORDER if m in per_floor[key]))

    out = {"per_floor": per_floor, "pooled": {}, "wilcoxon": {}, "n_phones_total": len(agg["naive"])}
    for m in agg:
        out["pooled"][m] = boot_ci(agg[m])

    # learned (headline method) vs every baseline, paired over phones
    best = "learned"
    out["best_method"] = best
    for base in ("naive", "binary", "rank", "meannorm", "hybrid"):
        a, b = np.array(agg[best]), np.array(agg[base])
        out["wilcoxon"][f"{best}_vs_{base}"] = paired_wilcoxon(a, b)
    # gap closed toward ceiling
    p = out["pooled"]
    cl, nv, bs = p["ceiling"]["mean"], p["naive"]["mean"], p[best]["mean"]
    out["gap_closed_pct"] = round(100.0 * (nv - bs) / (nv - cl), 1) if nv > cl else None

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(out, indent=2))

    print("\n===== POOLED cross-device localization (per-phone median error, m) =====")
    for m in ORDER:
        c = p.get(m, {})
        tag = "  <- OURS" if m in ("wjaccard", "sensmatch", "hybrid") else ""
        star = "  *BEST*" if m == best else ""
        print(f"  {LABEL[m]:34s}: {c.get('mean')}  [{c.get('lo','?')}, {c.get('hi','?')}]{tag}{star}")
    print(f"\n  best method: {best} ({p[best]['mean']} m) vs naive {p['naive']['mean']} / "
          f"binary {p['binary']['mean']} / ceiling {p['ceiling']['mean']}")
    print(f"  gap closed toward ceiling: {out['gap_closed_pct']}%")
    for k, w in out["wilcoxon"].items():
        print(f"  {k}: p={w.get('p_value'):.2g} (median_diff {w.get('median_diff'):.2f} m)")

    # figure
    try:
        _figure(out, Path(args.figdir))
    except Exception as e:  # noqa: BLE001
        print(f"  [fig skip] {e}")
    print(f"\n[loc-bench] -> {args.results}")


def _figure(out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.titlesize": 16, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    p = out["pooled"]
    show = ["naive", "meannorm", "rank", "binary", "hybrid", "learned", "ceiling"]
    vals = [p[m]["mean"] for m in show]
    los = [p[m].get("lo", vals[i]) for i, m in enumerate(show)]
    his = [p[m].get("hi", vals[i]) for i, m in enumerate(show)]
    err = [[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]]
    OURS, GREY, BLUE, CEIL = "#1e7d4f", "#9a9a9a", "#2c6fbf", "#c9a227"
    cols = [GREY, GREY, BLUE, BLUE, "#7aa8e0", OURS, CEIL]
    labels = ["naive\nRSS", "mean-norm\nRSS", "rank\n(SSD)", "heard-set\nJaccard",
              "detection-aware\nhybrid (ours)", "device-invariant\nembedding (ours)", "same-device\nceiling"]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    x = np.arange(len(show))
    ax.bar(x, vals, 0.62, yerr=err, capsize=5, color=cols)
    for xi, v, h in zip(x, vals, his):
        ax.text(xi, h + 0.15, f"{v:.1f}", ha="center", fontsize=13, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Median position error (m), lower is better")
    ax.set_ylim(0, max(his) * 1.15)
    ax.set_title(f"Cross-device indoor localization ({out['n_phones_total']} device-floors, LOPO, 95% CI)\n"
                 f"Detection-aware matching closes {out['gap_closed_pct']}% of the device gap")
    fig.savefig(figdir / "localization.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {figdir / 'localization.png'}")


if __name__ == "__main__":
    main()

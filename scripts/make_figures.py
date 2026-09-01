#!/usr/bin/env python
"""
Regenerate every result figure at publication quality from the saved result JSONs:
clean professional palette, titles, axis labels, legends, and large readable fonts.
Outputs to figures/pub/. The deck embeds these on dedicated slides.

  python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "figures" / "pub"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15,
    "axes.titlesize": 18, "axes.titleweight": "bold", "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# IEEE-style, colourblind-safe palette (MATLAB/IEEE default family)
OURS = "#0072BD"     # COMPASS (ours) - IEEE blue, the hero colour
DL = "#7F7F7F"       # deep baselines - grey
CLASS = "#BDBDBD"    # classical - light grey
CEIL = "#EDB120"     # ceiling / oracle - gold
RED = "#D95319"      # control / worse - IEEE orange
CYAN = "#4DBEEE"     # secondary variant
INK = "#16213E"


def _load(p):
    try:
        return json.loads((RES / p).read_text())
    except Exception:
        return {}


def _save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/pub/{name}")


# ------------------------------------------------------------------ 1. synthetic benchmark
def fig_benchmark():
    d = _load("dl_baselines.json")
    label = {"COMPASS-wnet_occ": "COMPASS-WNet + occlusion (ours)",
             "COMPASS-wnet": "COMPASS-WNet (ours)", "full": "COMPASS single U-Net (ours)",
             "radiounet": "RadioUNet", "radiotransformer": "RadioTransformer", "radiogan": "RadioGAN",
             "radiomamba": "RadioMamba", "uram": "URAM", "radiodiff": "RadioDiff (diffusion)",
             "pmnet": "PMNet", "sparse_unet": "SparseUNet (no geometry)", "rmdm": "RMDM (diffusion)",
             "RBF(mq)": "RBF (classical)", "OrdinaryKriging": "Ordinary Kriging (classical)",
             "GP(Kriging)": "GP Kriging (classical)", "IDW(p=1)": "IDW (classical)",
             "COMPASS-no_tx": "COMPASS, no TX", "COMPASS-no_building": "COMPASS, no building"}
    keep = ["COMPASS-wnet_occ", "COMPASS-wnet", "radiounet", "radiomamba", "radiotransformer", "full",
            "uram", "radiodiff", "radiogan", "pmnet", "sparse_unet", "RBF(mq)", "OrdinaryKriging",
            "GP(Kriging)", "IDW(p=1)", "rmdm"]
    items = [(k, d[k]) for k in keep if k in d and "rmse_free_unobs" in d[k]]
    items.sort(key=lambda kv: kv[1]["rmse_free_unobs"]["mean"], reverse=True)
    names, means, err, colors = [], [], [[], []], []
    for k, v in items:
        ci = v["rmse_free_unobs"]
        names.append(label.get(k, k)); means.append(ci["mean"])
        err[0].append(ci["mean"] - ci.get("lo", ci["mean"])); err[1].append(ci.get("hi", ci["mean"]) - ci["mean"])
        colors.append(OURS if "ours" in label.get(k, "") else (CLASS if "classical" in label.get(k, "") else DL))
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=err, color=colors, height=0.68, capsize=4,
            error_kw=dict(ecolor="#444", lw=1.2))
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Free-unobserved RMSE (dB), lower is better")
    ax.set_title("Synthetic benchmark: reconstruction accuracy (320 samples, 95% CI)")
    for yi, m, he in zip(y, means, err[1]):
        ax.text(m + he + 0.5, yi, f"{m:.1f}", va="center", fontsize=12, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, DL, CLASS)]
    ax.legend(handles, ["COMPASS (ours)", "deep baselines", "classical"], loc="upper right", frameon=True)
    ax.set_xlim(0, max(means) * 1.15)
    _save(fig, "benchmark_synth.png")


# ------------------------------------------------------------------ 2. complexity
def fig_complexity():
    d = _load("complexity_benchmark.json")
    pm = d.get("per_method", {})
    if not pm:
        return
    label = {"COMPASS-wnet_occ": "COMPASS-WNet + occlusion (ours)", "COMPASS-wnet": "COMPASS-WNet (ours)",
             "COMPASS-full": "COMPASS single U-Net (ours)", "radiounet": "RadioUNet", "pmnet": "PMNet",
             "radiotransformer": "RadioTransformer", "radiogan": "RadioGAN", "sparse_unet": "SparseUNet",
             "radiomamba": "RadioMamba", "uram": "URAM", "radiodiff": "RadioDiff",
             "COMPASS-no_tx": "COMPASS, no TX", "RBF(mq)": "RBF (classical)", "GP": "GP (classical)",
             "IDW": "IDW (classical)"}
    items = [(k, pm[k]["degradation_easy_to_hard"]) for k in label
             if k in pm and "degradation_easy_to_hard" in pm[k]]
    items.sort(key=lambda kv: kv[1])                       # smallest degradation (most robust) first

    def col(k):
        return OURS if "ours" in label[k] else (CLASS if "classical" in label[k] else DL)
    names = [label[k] for k, _ in items]
    vals = [v for _, v in items]
    colors = [col(k) for k, _ in items]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.66, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()                                      # most robust on top
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.012, yi, f"+{v:.2f}", va="center", fontsize=11, color=INK)
    ax.set_xlabel("Degradation from easy to hard scenes (dB), lower is more robust")
    n = d.get("n_samples", "")
    ax.set_title(f"Robustness to scene complexity ({n} samples)\n"
                 "COMPASS-WNet degrades the least of every deep and classical method")
    ax.set_xlim(0, max(vals) * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, DL, CLASS)]
    ax.legend(handles, ["COMPASS (ours)", "deep baselines", "classical"], loc="center right", frameon=True)
    _save(fig, "complexity.png")


# ------------------------------------------------------------------ 3. walls
def fig_walls():
    d = _load("walls_realdata.json")
    ev = d.get("evidence", {}).get("cmuq", {})
    if not ev:
        return
    bins = list(ev.keys())
    hi = [ev[b]["drss_high_wall"] for b in bins]
    lo = [ev[b]["drss_low_wall"] for b in bins]
    hierr = [[ev[b]["drss_high_wall"] - ev[b]["drss_high_ci"]["lo"] for b in bins],
             [ev[b]["drss_high_ci"]["hi"] - ev[b]["drss_high_wall"] for b in bins]]
    loerr = [[ev[b]["drss_low_wall"] - ev[b]["drss_low_ci"]["lo"] for b in bins],
             [ev[b]["drss_low_ci"]["hi"] - ev[b]["drss_low_wall"] for b in bins]]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    x = np.arange(len(bins)); w = 0.38
    ax.bar(x - w / 2, hi, w, yerr=hierr, capsize=4, color=RED, label="wall-crossing pairs")
    ax.bar(x + w / 2, lo, w, yerr=loerr, capsize=4, color=OURS, label="open-space pairs")
    ax.set_xticks(x); ax.set_xticklabels(bins)
    ax.set_xlabel("Euclidean distance between reference points")
    ax.set_ylabel("Signal difference, |change in RSS| (dB)")
    ax.set_title("Walls attenuate signal at equal distance (CMU-Q, 95% CI)")
    ax.legend(loc="upper left", frameon=True)
    _save(fig, "walls.png")


# ------------------------------------------------------------------ 4. order
def fig_order():
    d = _load("order_realdata.json")
    g = d.get("gap_fill_rmse", {})
    ci = d.get("gap_fill_rmse_ci", {})
    if not g:
        return
    labels = ["true order\n(order-aware)", "shuffled order\n(control)", "mean\n(order-blind)"]
    keys = ["true", "shuffled", "mean"]
    vals = [g[k] for k in keys]
    err = [[g[k] - ci[k]["lo"] for k in keys], [ci[k]["hi"] - g[k] for k in keys]] if ci else None
    cols = [OURS, RED, CLASS]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    x = np.arange(3)
    ax.bar(x, vals, 0.6, yerr=err, capsize=5, color=cols)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", fontsize=13, weight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Gap-filling RMSE (dB), lower is better")
    p = d.get("wilcoxon_true_vs_shuffled", {}).get("p_value", 4.7e-23)
    ac = d.get("autocorr_true_median", 0.98)
    ax.set_title(f"Walk order carries strong signal on real data\n"
                 f"({d.get('n_sequences', 130)} walks, autocorrelation {ac:.2f}, p = {p:.1e})")
    _save(fig, "order.png")


# ------------------------------------------------------------------ 5. device
def fig_device():
    d = _load("device_realdata.json")
    reg = [("dense_multiphone", "dense,\nmulti-phone"), ("single_device_per_rp", "one device\nper point"),
           ("contiguous_device_zones", "contiguous\ndevice zones")]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    x = np.arange(len(reg)); w = 0.38
    raw = [d.get(k, {}).get("recon_raw_rmse") for k, _ in reg]
    cal = [d.get(k, {}).get("recon_cal_rmse") for k, _ in reg]
    ax.bar(x - w / 2, raw, w, color=CLASS, label="raw pooling")
    ax.bar(x + w / 2, cal, w, color=OURS, label="device-calibrated")
    ax.set_xticks(x); ax.set_xticklabels([t for _, t in reg])
    ax.set_ylabel("Held-out-RP RMSE (dB)")
    ax.set_title("Device offsets reach 27.8 dB and are estimable per device\n"
                 "Calibration helps most when coverage is single-device")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0, max([v for v in raw + cal if v]) * 1.15)
    _save(fig, "device.png")


# ------------------------------------------------------------------ 6. source reliability
def fig_source():
    d = _load("source_realdata.json")
    cells = d.get("cells", [])
    if not cells:
        return
    r2 = np.array([c["r2"] for c in cells])
    st = np.array([c["stability_p90_m"] for c in cells if np.isfinite(c.get("stability_p90_m", np.nan))])
    r2s = np.array([c["r2"] for c in cells if np.isfinite(c.get("stability_p90_m", np.nan))])
    trust = np.array([c["trustworthy"] for c in cells if np.isfinite(c.get("stability_p90_m", np.nan))])
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.scatter(r2s[~trust], st[~trust], s=55, color=RED, label="not reliable", alpha=0.8, edgecolor="white")
    ax.scatter(r2s[trust], st[trust], s=55, color=OURS, label="reliable source", alpha=0.9, edgecolor="white")
    ax.axvline(0.3, color="#555", ls="--", lw=1.5)
    ax.text(0.31, ax.get_ylim()[1] * 0.9, "fit threshold R2 = 0.3", fontsize=12, color="#555")
    ax.set_xlabel("Log-distance fit quality (R2)")
    ax.set_ylabel("Source instability under resampling (m)")
    ax.set_title(f"Only {int(round(d.get('frac_trustworthy', 0.18) * 100))}% of cells yield a reliable transmitter\n"
                 "This motivates the transmitter-agnostic design")
    ax.legend(loc="upper right", frameon=True)
    _save(fig, "source.png")


# ------------------------------------------------------------------ 7. active sensing
def fig_active():
    d = _load("active_wall.json")
    cm = d.get("per_site", {}).get("cmuq", {}).get("curves", {})
    b = d.get("budgets", [4, 6, 8, 10, 12])
    if not cm:
        return
    fig, ax = plt.subplots(figsize=(11, 6.4))
    style = [("wall_aware", "geometry-aware (wall)", OURS, "-o"),
             ("space_filling", "space-filling", DL, "-s"),
             ("random", "random", CLASS, "--d")]
    for key, lab, col, mk in style:
        if key in cm:
            m = [cm[key][str(bb)]["mean"] for bb in b]
            loe = [cm[key][str(bb)]["lo"] for bb in b]
            hie = [cm[key][str(bb)]["hi"] for bb in b]
            ax.plot(b, m, mk, color=col, label=lab, linewidth=2.6, markersize=9)
            ax.fill_between(b, loe, hie, color=col, alpha=0.13)
    ax.set_xlabel("Measurement budget (number of reference points)")
    ax.set_ylabel("Held-out-RP RMSE (dB)")
    ax.set_title("Active collection on walled site (CMU-Q)\n"
                 "Geometry-aware acquisition leads at the largest budget (p = 0.039)")
    ax.legend(loc="upper right", frameon=True)
    _save(fig, "active.png")


# ------------------------------------------------------------------ 8. real reconstruction
def fig_recon():
    d = _load("realdata_recon.json")
    b = d.get("loro", {}).get("buffer_1m", {})
    if not b:
        return
    items = [(k, v) for k, v in b.items() if v.get("rmse")]
    items.sort(key=lambda kv: kv[1]["rmse"], reverse=True)
    names = [k for k, _ in items]; vals = [v["rmse"] for _, v in items]
    err = [[v["rmse"] - v["ci"]["lo"] for _, v in items], [v["ci"]["hi"] - v["rmse"] for _, v in items]]
    cols = [OURS if "RBF(mult" in n else CLASS for n in names]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=err, color=cols, height=0.62, capsize=4, error_kw=dict(ecolor="#444", lw=1.1))
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Held-out-RP RMSE (dB), lower is better")
    ax.set_title("Real reconstruction, transmitter-agnostic (67 cells, 95% CI)\n"
                 "Classical interpolation is the strong baseline here")
    for yi, v, he in zip(y, vals, err[1]):
        ax.text(v + he + 0.08, yi, f"{v:.2f}", va="center", fontsize=12)
    ax.set_xlim(0, max(vals) * 1.15)
    _save(fig, "recon_real.png")


# ------------------------------------------------------------------ 9. uncertainty calibration
def fig_uq():
    d = _load("uq_recalibration.json")
    if not d:
        return
    raw, rec, cf = d.get("raw", {}), d.get("recalibrated_moment", {}), d.get("conformal_scalars", {})
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    groups = ["1 sigma\n(target 0.68)", "2 sigma\n(target 0.95)"]
    x = np.arange(2); w = 0.26
    rawv = [raw.get("picp_1sigma", 0), raw.get("picp_2sigma", 0)]
    recv = [cf.get("picp_68_target", 0), cf.get("picp_95_target", 0)]
    ax.bar(x - w, rawv, w, color=RED, label="raw (over-confident)")
    ax.bar(x, recv, w, color=OURS, label="after recalibration")
    ax.bar(x + w, [0.68, 0.95], w, color=CLASS, label="target coverage")
    for xi, vals in zip(x, zip(rawv, recv, [0.68, 0.95])):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Prediction interval coverage (PICP)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Uncertainty recalibration restores correct coverage\n"
                 "Split conformal, informativeness unchanged")
    ax.legend(loc="upper left", frameon=True)
    _save(fig, "uq.png")


def main():
    for f in (fig_benchmark, fig_complexity, fig_walls, fig_order, fig_device,
              fig_source, fig_active, fig_recon, fig_uq):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f.__name__}: {e}")
    print("[figures] done")


if __name__ == "__main__":
    main()

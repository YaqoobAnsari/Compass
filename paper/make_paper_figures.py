#!/usr/bin/env python
"""Figures for the CVPR paper. Writes vector PDFs into paper/fig/.

Every number is read from results/*.json or recomputed from the dataset. Nothing
is hard-coded, so a regenerated result file changes the figure.

  PYTHONPATH=src python paper/make_paper_figures.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper" / "fig"
FIG.mkdir(parents=True, exist_ok=True)
R = lambda n: json.loads((REPO / "results" / n).read_text().replace("NaN", "null"))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK = "#1a1a1a"; ACC = "#1f4e79"; HOT = "#b3202c"; MUT = "#8a8a8a"; GRN = "#1e6b2e"


def fig_plateau():
    """Every deep family plateaus in the same NLoS band while LoS varies widely."""
    d = R("dl_baselines.json")
    # names match tab/main.tex: a dagger marks a family representative we built,
    # not a reimplementation of the cited paper
    label = {"COMPASS-wnet_occ": "Ours, WNet+occ", "COMPASS-unet_occ": "Ours, UNet+occ",
             "COMPASS-wnet_base": "WNet", "full": "U-Net",
             "radiounet": "RadioUNet", "radiomamba": "Mamba-UNet$^\\dagger$",
             "radiotransformer": "CNN-Transf.$^\\dagger$", "uram": "Bayes. U-Net$^\\dagger$",
             "radiodiff": "RadioDiff", "radiogan": "cGAN$^\\dagger$", "pmnet": "PMNet"}
    ours = {"COMPASS-wnet_occ", "COMPASS-unet_occ"}
    rows = [(n, d[k]["rmse_los"], d[k]["rmse_nlos"], k in ours)
            for k, n in label.items() if k in d and d[k].get("rmse_nlos") is not None]
    rows.sort(key=lambda r: r[2], reverse=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    y = np.arange(len(rows))
    for i, (n, los, nlos, is_ours) in enumerate(rows):
        c = HOT if is_ours else ACC
        ax.plot([los, nlos], [i, i], color=c, lw=1.0, alpha=0.5, zorder=2)
        ax.scatter([los], [i], s=22, marker="o", facecolor="white",
                   edgecolor=c, linewidths=1.0, zorder=3)
        ax.scatter([nlos], [i], s=24, marker="o", color=c, linewidths=0, zorder=3)
    band = [r[2] for r in rows if not r[3]]
    lo, hi = min(band), max(band)
    ax.axvspan(lo, hi, color=ACC, alpha=0.08, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[3]:
            t.set_color(HOT); t.set_weight("bold")
    ax.set_xlabel("RMSE (dB)")
    ax.set_ylim(-1.5, len(rows) - 0.2)
    ax.scatter([], [], s=22, marker="o", facecolor="white", edgecolor=INK,
               linewidths=1.0, label="line of sight")
    ax.scatter([], [], s=24, marker="o", color=INK, linewidths=0,
               label="non line of sight")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.13),
              ncol=2, handletextpad=0.3, columnspacing=1.0)
    # "to" rather than an en dash, matching the prose convention in the paper
    ax.text((lo + hi) / 2, -1.05, f"error floor {lo:.1f} to {hi:.1f} dB",
            fontsize=6, color=ACC, ha="center", va="center")
    fig.savefig(FIG / "plateau.pdf"); plt.close(fig)
    print(f"  plateau.pdf  NLoS band {lo:.2f}-{hi:.2f} dB over {len(band)} non-occlusion methods; "
          f"LoS spread {min(r[1] for r in rows):.2f}-{max(r[1] for r in rows):.2f}")


def fig_interaction():
    """Occlusion x cascade: the two gains overlap rather than compound."""
    d = R("dl_baselines.json")
    cells = {("UNet", "no occ."): "full", ("UNet", "occ."): "COMPASS-unet_occ",
             ("WNet", "no occ."): "COMPASS-wnet_base", ("WNet", "occ."): "COMPASS-wnet_occ"}
    v, e = {}, {}
    for k, name in cells.items():
        ci = d[name]["rmse_free_unobs"]
        v[k] = ci["mean"]; e[k] = (ci["mean"] - ci["lo"], ci["hi"] - ci["mean"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.35, 2.2),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    x = np.arange(2); w = 0.36
    for i, occ in enumerate(("no occ.", "occ.")):
        ys = [v[("UNet", occ)], v[("WNet", occ)]]
        er = np.array([e[("UNet", occ)], e[("WNet", occ)]]).T
        ax.bar(x + (i - 0.5) * w, ys, w, yerr=er, capsize=2, linewidth=0,
               color=(MUT if i == 0 else HOT), label=("without" if i == 0 else "with") + " occlusion")
        for xi, y in zip(x + (i - 0.5) * w, ys):
            ax.text(xi, y + 0.45, f"{y:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["single U-Net", "WNet cascade"])
    ax.set_ylabel("free-unobs. RMSE (dB)")
    ax.set_ylim(0, max(v.values()) * 1.34)
    ax.legend(frameon=False, loc="upper center", ncol=1, handlelength=1.1,
              handletextpad=0.4, borderaxespad=0.1)

    # gains are reductions, so work in positive "improvement" units throughout
    obs = v[("UNet", "no occ.")] - v[("WNet", "occ.")]
    add = ((v[("UNet", "no occ.")] - v[("UNet", "occ.")])
           + (v[("UNet", "no occ.")] - v[("WNet", "no occ.")]))
    ax2.bar([0, 1], [add, obs], 0.55, color=[MUT, HOT], linewidth=0)
    for xi, yv in zip([0, 1], [add, obs]):
        ax2.text(xi, yv + 0.08, f"{yv:.2f}", ha="center", fontsize=6)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["parts\nsummed", "observed\njointly"])
    ax2.set_ylabel("improvement (dB)")
    ax2.annotate("", (0.5, obs), (0.5, add), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK))
    ax2.text(0.5, add + 0.30, f"{add - obs:.2f} dB shortfall", fontsize=6,
             ha="center", color=INK)
    ax2.set_ylim(0, add * 1.34)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIG / "interaction.pdf"); plt.close(fig)
    print(f"  interaction.pdf  summed {add:.3f}  observed {obs:.3f}  shortfall {add-obs:+.3f}")


def fig_walls():
    """Real walls attenuate, and an open-plan control site shows no effect."""
    d = R("walls_realdata.json")["evidence"]
    bins = ["1-2m", "2-3m", "3-4m", "4-6m"]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.1), sharey=True)
    for ax, site, title in zip(axes, ("cmuq", "ec_parking"),
                               ("CMU-Q (walled)", "EC Parking (open, control)")):
        ev = d.get(site, {})
        bs = [b for b in bins if b in ev]
        x = np.arange(len(bs)); w = 0.36
        for i, (key, cikey, c, lab) in enumerate(
                [("drss_low_wall", "drss_low_ci", MUT, "few walls"),
                 ("drss_high_wall", "drss_high_ci", HOT, "many walls")]):
            ys = [ev[b][key] for b in bs]
            er = np.array([[ev[b][cikey]["mean"] - ev[b][cikey]["lo"],
                            ev[b][cikey]["hi"] - ev[b][cikey]["mean"]] for b in bs]).T
            ax.bar(x + (i - 0.5) * w, ys, w, yerr=er, capsize=1.6, linewidth=0,
                   color=c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([b.replace("m", "") for b in bs])
        ax.set_xlabel("separation (m)"); ax.set_title(title, fontsize=7)
    axes[0].set_ylabel(r"excess $|\Delta$RSS$|$ (dB)")
    # headroom so the legend never sits on a bar or its error cap
    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, top * 1.30)
    axes[0].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.0,
                   handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "walls.pdf"); plt.close(fig)
    print("  walls.pdf")


def fig_concept():
    """The occlusion field, and the transmitter-free variant that fails."""
    try:
        import torch
        sys.path.insert(0, str(REPO / "src"))
        from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids
        from compass.models.compass_net import occlusion_field, measurement_occlusion_field
    except Exception as ex:
        print(f"  concept.pdf SKIPPED ({ex})"); return
    root = REPO / "data" / "raw"
    mid = split_map_ids(list_map_ids(str(root)))["test"][0]
    ds = RadioMapSeerDataset(str(root), map_ids=[mid], variant="IRT2")
    s = ds[3]
    b = s["building_map"][None]
    txrc = torch.tensor([[s["tx_rowcol"][0], s["tx_rowcol"][1]]], dtype=torch.float32)
    occ = occlusion_field(b, txrc)[0, 0].numpy()
    rng = np.random.default_rng(0)
    free = s["free_mask"][0].numpy() > 0.5
    idx = np.argwhere(free); sel = idx[rng.choice(len(idx), 300, replace=False)]
    mask = torch.zeros_like(b); mask[0, 0, sel[:, 0], sel[:, 1]] = 1.0
    mocc = measurement_occlusion_field(b, mask)[0, 0].numpy()
    rm = s["radio_map_dbm"][0].numpy() if "radio_map_dbm" in s else s["radio_map"][0].numpy()

    # cividis is perceptually uniform, colour-vision safe and far less loud than
    # magma beside the line figures. The two occlusion panels are the same quantity
    # on the same scale, so they share one colourbar instead of carrying two.
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
    for ax, (im, t, cm) in zip(axes, [(b[0, 0].numpy(), "building layout $+$ Tx", "gray_r"),
                                      (occ, r"occlusion $O(p)$, Tx-anchored", "cividis"),
                                      (mocc, r"$\tilde{O}(p)$, Tx-free", "cividis"),
                                      (rm, "ground-truth field", "viridis")]):
        kw = dict(vmin=0, vmax=1) if cm == "cividis" else {}
        h = ax.imshow(im, cmap=cm, **kw)
        ax.set_title(t, fontsize=7.2); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if t.startswith("building"):
            ax.plot(txrc[0, 1], txrc[0, 0], marker="*", ms=9, color=HOT, mew=0)
        if t.startswith("ground"):
            cb = plt.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=5); cb.set_label("dBm", fontsize=5.5)
        if t.startswith(r"$\tilde"):
            shared = h
    cb = fig.colorbar(shared, ax=axes[1:3].tolist(), fraction=0.023, pad=0.012)
    cb.ax.tick_params(labelsize=5)
    cb.set_ticks([0, 0.5, 1.0])
    fig.savefig(FIG / "concept.pdf"); plt.close(fig)
    print(f"  concept.pdf  map {mid}  Tx-free field mean {mocc.mean():.3f} vs Tx-anchored {occ.mean():.3f}")


# ---------------------------------------------------------------- tables ----
def _f(x, n=2):
    return "--" if x is None else f"{x:.{n}f}"


def table_main():
    """Table 1, generated from results/dl_baselines.json so it cannot drift."""
    d = R("dl_baselines.json")
    groups = [
        ("\\textit{Ours}", [("COMPASS-wnet_occ", "WNet $+$ occlusion"),
                            ("COMPASS-unet_occ", "U-Net $+$ occlusion")]),
        ("\\textit{Learned baselines}", [
            ("radiounet", "RadioUNet~\\cite{levie2021radiounet}"),
            ("COMPASS-wnet_base", "WNet (no occlusion)"),
            ("radiomamba", "Mamba-UNet$^\\dagger$~\\cite{jia2025radiomamba}"),
            ("radiotransformer", "CNN-Transformer$^\\dagger$~\\cite{li2025rmtransformer}"),
            ("uram", "Bayesian U-Net$^\\dagger$~\\cite{lu2025uram}"),
            ("radiodiff", "RadioDiff~\\cite{wang2025radiodiff}"),
            ("radiogan", "cGAN$^\\dagger$~\\cite{zhang2023rmegan}"),
            ("full", "U-Net (no occlusion)"),
            ("pmnet", "PMNet~\\cite{lee2024pmnet}"),
            ("sparse_unet", "SparseUNet$^\\dagger$ (no geometry)"), ("rmdm", "RMDM~\\cite{jia2025rmdm}")]),
        ("\\textit{Input ablations}", [
            ("COMPASS-wnet_occ_meas", "Tx-free occlusion"),
            ("COMPASS-no_tx", "no Tx heatmap"),
            ("COMPASS-no_building", "no buildings")]),
        ("\\textit{Classical}", [("RBF(mq)", "RBF"),
                                 ("OrdinaryKriging", "Ord. Kriging"),
                                 ("IDW(p=1)", "IDW ($p{=}1$)"), ("GP(Kriging)", "GP")]),
    ]
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"& Method & RMSE$\downarrow$ & SSIM$\uparrow$ & LoS$\downarrow$ & NLoS$\downarrow$ \\",
         r"\midrule"]
    for gname, items in groups:
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{{gname}}} \\\\")
        for key, name in items:
            if key not in d:
                continue
            v = d[key]; ci = v["rmse_free_unobs"]
            rm = f"{ci['mean']:.2f}\\,\\tiny[{ci['lo']:.2f},{ci['hi']:.2f}]"
            if key == "COMPASS-wnet_occ":
                rm = "\\textbf{" + f"{ci['mean']:.2f}" + "}\\,\\tiny[" + f"{ci['lo']:.2f},{ci['hi']:.2f}" + "]"
            L.append(f"& {name} & {rm} & {_f(v.get('ssim_free'),3)} & "
                     f"{_f(v.get('rmse_los'))} & {_f(v.get('rmse_nlos'))} \\\\")
        L.append(r"\addlinespace[1pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "paper" / "tab" / "main.tex").write_text("\n".join(L))
    print(f"  tab/main.tex  {sum(len(i) for _, i in groups)} rows")


def table_real():
    """Table 2, real-data reconstruction, from results/learned_real.json."""
    d = R("learned_real.json")
    order = [("RBF(mq)", "RBF (multiquadric)"), ("wall-aware IDW", "wall-aware IDW"),
             ("IDW(p=2)", "IDW ($p{=}2$)"), ("GP/Kriging", "GP / Kriging"),
             ("learned+walls", "learned, with geometry"),
             ("learned-nowalls", "learned, no geometry")]
    L = [r"\begin{tabular}{@{}lrl@{}}", r"\toprule",
         r"Method & RMSE (dB)$\downarrow$ & vs.\ learned \\", r"\midrule"]
    for k, name in order:
        if k not in d["rmse"]:
            continue
        ci = d["rmse_ci"][k]
        w = d.get("wilcoxon_learned_vs", {}).get(k, {})
        p = w.get("p_value")
        ptxt = "--" if p is None else ("ref." if k == "learned+walls" else f"$p{{=}}${p:.0e}")
        best = "\\textbf{" if k == "RBF(mq)" else ""
        end = "}" if best else ""
        L.append(f"{name} & {best}{d['rmse'][k]:.2f}{end}\\,\\tiny[{ci['lo']:.2f},{ci['hi']:.2f}] & {ptxt} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "paper" / "tab" / "real.tex").write_text("\n".join(L))
    print("  tab/real.tex")



def fig_architecture():
    """Where the occlusion channel enters, and how the two backbones differ.

    Coordinates are laid out on an explicit 105x82 grid with reserved bands so no
    label can land on a box: boxes occupy y in [14,64], captions sit above y=66 or
    below y=12, and the output annotations live right of x=92.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon

    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax.set_xlim(0, 105); ax.set_ylim(0, 82); ax.axis("off")

    # ---- conditioning inputs, the studied channel picked out --------------
    chans = [("sparse RSS", False), ("mask", False), ("coverage", False),
             ("building", False), ("Tx heatmap", False), (r"occlusion $O(p)$", True)]
    x0, w, h, gap = 1.0, 27.0, 6.4, 1.6
    top = 64.0
    for i, (name, studied) in enumerate(chans):
        y = top - i * (h + gap)
        c = HOT if studied else MUT
        ax.add_patch(Rectangle((x0, y - h), w, h,
                               facecolor="#f6dee0" if studied else "#f0f0f0",
                               edgecolor=c, linewidth=1.1 if studied else 0.7, zorder=2))
        # the studied row is bold and carries math, so it needs a smaller size to
        # stay clear of the box border it sits inside
        ax.text(x0 + w / 2, y - h / 2, name, ha="center", va="center",
                fontsize=5.6 if studied else 6.0, color=HOT if studied else INK,
                weight="bold" if studied else "normal", zorder=3)
    ax.text(x0 + w / 2, 68.5, "conditioning inputs", ha="center", fontsize=6.3,
            color=INK, style="italic")
    ax.text(x0 + w / 2, 6.0, "$+432$ parameters", ha="center", fontsize=5.8, color=HOT)

    # ---- the two stages ---------------------------------------------------
    def unet(cx, label):
        for pts in ([(cx - 9, 52), (cx - 1.8, 45), (cx - 1.8, 33), (cx - 9, 26)],
                    [(cx + 1.8, 45), (cx + 9, 52), (cx + 9, 26), (cx + 1.8, 33)]):
            ax.add_patch(Polygon(pts, closed=True, facecolor="#e7edf3",
                                 edgecolor=ACC, linewidth=0.8, zorder=2))
        ax.plot([cx - 1.8, cx + 1.8], [39, 39], color=ACC, lw=0.8, zorder=2)
        ax.text(cx, 54.4, label, ha="center", fontsize=6.0, color=ACC, zorder=3)
        return cx - 9, cx + 9

    l1, r1 = unet(46, "stage 1")
    l2, r2 = unet(76, "stage 2")

    def arrow(xa, xb, y, c=INK, ls="-"):
        ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=c, linestyle=ls, shrinkA=0, shrinkB=0,
                                     zorder=4))

    arrow(x0 + w + 1.2, l1 - 1.0, 39)
    arrow(r1 + 1.0, l2 - 1.0, 39)
    arrow(r2 + 1.0, 91.0, 39)

    # stage 1's estimate re-entering stage 2 is what makes it a cascade
    ax.add_patch(FancyArrowPatch((r1 - 2.0, 25.0), (l2 + 2.0, 25.0), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.7, color=ACC,
                                 linestyle=(0, (2.5, 1.5)),
                                 connectionstyle="arc3,rad=0.45", zorder=3))
    ax.text((r1 + l2) / 2, 15.5, "coarse map", ha="center", fontsize=5.6, color=ACC)

    # ---- the two backbones under study ------------------------------------
    ax.add_patch(Rectangle((l1 - 3.0, 20.0), (r1 + 3.0) - (l1 - 3.0), 38.0, fill=False,
                           edgecolor=GRN, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1))
    ax.text(46, 60.0, "single U-Net", ha="center", fontsize=6.1, color=GRN)
    ax.add_patch(Rectangle((l1 - 5.5, 13.0), (r2 + 5.5) - (l1 - 5.5), 52.0, fill=False,
                           edgecolor=INK, linewidth=0.9, zorder=1))
    ax.text(61, 68.5, "WNet cascade", ha="center", fontsize=6.4, color=INK)

    # ---- outputs, clear of every box --------------------------------------
    ax.text(92.5, 39, r"$\hat{R}$", ha="left", va="center", fontsize=8, color=INK)
    ax.text(92.5, 30.5, r"$\hat{\sigma}$", ha="left", va="center", fontsize=8, color=MUT)
    ax.text(92.5, 23.5, "MC\ndropout", ha="left", va="center", fontsize=5.4, color=MUT,
            linespacing=1.15)

    fig.savefig(FIG / "architecture.pdf"); plt.close(fig)
    print("  architecture.pdf")


if __name__ == "__main__":
    print("[figures] ->", FIG)
    (REPO / "paper" / "tab").mkdir(parents=True, exist_ok=True)
    for fn in (fig_architecture, fig_plateau, fig_interaction, fig_walls,
               fig_concept, table_main, table_real):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            print(f"  {fn.__name__} FAILED: {type(ex).__name__}: {ex}")

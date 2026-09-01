#!/usr/bin/env python
"""
COMPASS-WNet architecture, publication grade, IEEE-style palette. A clean left-to-right
dataflow with an explicit colour code: inputs, backbone, our contributions, and the NEW
occlusion layer highlighted. Also renders the real-data residual pipeline.

  python scripts/make_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# ---- IEEE-style palette ----
INK = "#16213E"
BLUE = "#0072BD"          # backbone
BLUE_FILL = "#DCE6F2"
INPUT_FILL = "#EAF0F7"
INPUT_EDGE = "#5A7CA8"
GREEN = "#1B7A3D"         # COMPASS contributions
GREEN_FILL = "#D8EADD"
ORANGE = "#D95319"        # NEW in this work (occlusion)
ORANGE_FILL = "#FBE1D3"
GOLD = "#C8960C"          # outputs / uncertainty
GOLD_FILL = "#FBEFCB"
WHITE = "#FFFFFF"
GREY = "#6B7686"
W, H = 16.0, 9.0          # inches; canvas 160 x 90 (equal aspect)


def _canvas():
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc, tc=INK, fs=9.0, ec=INPUT_EDGE, lw=1.4,
        weight="bold", rounding=1.4, z=4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                 facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
                weight=weight, zorder=z + 2)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.8, style="-|>", ms=12, ls="-", z=3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                 color=color, linewidth=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def unet_glyph(ax, cx, cy, label):
    """A clean U-Net encoder-decoder icon with skip connections."""
    fill, edge = BLUE_FILL, BLUE
    enc_y = [cy + 7.5, cy + 1.5, cy - 4.5]; enc_w = [10.0, 7.2, 4.8]
    dec_w = [4.8, 7.2, 10.0]; dec_y = [cy - 4.5, cy + 1.5, cy + 7.5]
    for w, y in zip(enc_w, enc_y):
        box(ax, cx - 7.6, y, w, 3.0, "", fill, ec=edge, lw=1.2, rounding=0.6)
    box(ax, cx, cy - 10.2, 4.6, 3.0, "", fill, ec=edge, lw=1.2, rounding=0.6)
    for w, y in zip(dec_w, dec_y):
        box(ax, cx + 7.6, y, w, 3.0, "", fill, ec=edge, lw=1.2, rounding=0.6)
    for ew, ey, dw in zip(enc_w, enc_y, dec_w[::-1]):
        arrow(ax, cx - 7.6 + ew / 2, ey, cx + 7.6 - dw / 2, ey,
              color="#9DB4CE", lw=1.0, style="-", ls=(0, (3, 2)), z=2)
    arrow(ax, cx - 7.6, enc_y[0] - 1.5, cx - 7.6, enc_y[1] + 1.5, color=edge, lw=1.1, ms=7)
    arrow(ax, cx - 7.6, enc_y[1] - 1.5, cx - 7.6, enc_y[2] + 1.5, color=edge, lw=1.1, ms=7)
    arrow(ax, cx - 7.6 + 1.0, enc_y[2] - 1.5, cx - 1.6, cy - 10.2, color=edge, lw=1.1, ms=7)
    arrow(ax, cx + 1.6, cy - 10.2, cx + 7.6 - 1.0, dec_y[0] - 1.5, color=edge, lw=1.1, ms=7)
    arrow(ax, cx + 7.6, dec_y[0] + 1.5, cx + 7.6, dec_y[1] - 1.5, color=edge, lw=1.1, ms=7)
    arrow(ax, cx + 7.6, dec_y[1] + 1.5, cx + 7.6, dec_y[2] - 1.5, color=edge, lw=1.1, ms=7)
    ax.text(cx, cy + 12.4, label, ha="center", fontsize=9.5, weight="bold", color=INK)


def legend(ax, y, items):
    for kind, label, x in items:
        fc = {"input": INPUT_FILL, "backbone": BLUE_FILL, "ours": GREEN_FILL,
              "new": ORANGE_FILL, "out": GOLD_FILL}.get(kind)
        ec = {"input": INPUT_EDGE, "backbone": BLUE, "ours": GREEN,
              "new": ORANGE, "out": GOLD}.get(kind)
        box(ax, x, y, 4.4, 3.0, "", fc, ec=ec, lw=1.4, rounding=0.7)
        ax.text(x + 3.6, y, label, ha="left", va="center", fontsize=8.8, color=INK)


# ---------------------------------------------------------------- COMPASS-WNet
def build_wnet():
    fig, ax = _canvas()
    ax.text(80, 86, "COMPASS-WNet architecture", ha="center", fontsize=19, weight="bold", color=INK)
    ax.text(80, 81, "A geometry, order, and device conditioned cascade of two U-Nets, with an explicit "
            "occlusion channel and calibrated uncertainty",
            ha="center", fontsize=9.5, color=GREY)

    # ---- input channel stack (left) ----
    ax.text(19, 74, "Conditioning inputs", ha="center", fontsize=10.5, weight="bold", color=INK)
    inputs = [
        ("Sparse RSS", INPUT_FILL, INPUT_EDGE, ""),
        ("Observed mask", INPUT_FILL, INPUT_EDGE, ""),
        ("Coverage density", INPUT_FILL, INPUT_EDGE, ""),
        ("Building map", GREEN_FILL, GREEN, ""),
        ("TX heatmap (multi-scale)", GREEN_FILL, GREEN, ""),
        ("Ray-occlusion field", ORANGE_FILL, ORANGE, "NEW"),
    ]
    ys = [69, 62.5, 56, 49.5, 43, 36.5]
    for (t, fc, ec, tag), y in zip(inputs, ys):
        box(ax, 19, y, 30, 5.4, t, fc, ec=ec, fs=9.0, lw=1.5)
        arrow(ax, 34, y, 39.5, 52.5, color=GREY, lw=1.0, ms=8)
        if tag:
            ax.text(35.6, y + 2.9, tag, ha="left", va="center", fontsize=7.6,
                    weight="bold", color=ORANGE)
    box(ax, 42, 52.5, 5.5, 40, "", INPUT_FILL, ec=INPUT_EDGE, rounding=1.2)
    ax.text(42, 52.5, "concat", ha="center", va="center", rotation=90, fontsize=9.0,
            weight="bold", color=INK, zorder=7)

    # ---- WNet cascade ----
    ax.add_patch(FancyBboxPatch((50, 30), 74, 45, boxstyle="round,pad=0,rounding_size=2.0",
                 facecolor="none", edgecolor=BLUE, linewidth=1.6, linestyle=(0, (6, 4)), zorder=1))
    ax.text(87, 72.5, "WNet cascade backbone", ha="center", fontsize=9.5, weight="bold", color=BLUE)
    unet_glyph(ax, 66, 52.5, "U-Net 1  (coarse)")
    box(ax, 89, 53.5, 12, 6.5, "coarse\nmap", INPUT_FILL, ec=INPUT_EDGE, fs=8.6, rounding=1.0)
    unet_glyph(ax, 111, 52.5, "U-Net 2  (refine)")
    arrow(ax, 45, 52.5, 54.5, 52.5, lw=2.0)
    arrow(ax, 78, 53.2, 83, 53.4, lw=1.8)
    arrow(ax, 95, 53.4, 100.5, 53.2, lw=1.8)

    # ---- order + device heads (bottom) ----
    box(ax, 60, 17, 28, 7, "Trajectory sequence", INPUT_FILL, ec=INPUT_EDGE, fs=9.0)
    box(ax, 99, 17, 28, 7, "GRU order encoder", GREEN_FILL, ec=GREEN, fs=9.0)
    arrow(ax, 74, 17, 84.5, 17, color=GREEN, lw=1.7)
    arrow(ax, 92, 20.6, 69, 40, color=GREEN, lw=1.6)
    arrow(ax, 105, 20.6, 111, 40, color=GREEN, lw=1.6)
    ax.text(95, 33, "FiLM", ha="center", fontsize=8.8, color=GREEN, weight="bold")
    box(ax, 140, 17, 26, 7, "Device-offset head", GREEN_FILL, ec=GREEN, fs=9.0)

    # ---- outputs (right) ----
    box(ax, 145, 60, 26, 7.2, "Dense radio map", GOLD_FILL, ec=GOLD, fs=9.0)
    box(ax, 145, 50, 26, 7.0, "MC-dropout ensemble", GOLD_FILL, ec=GOLD, fs=9.0)
    box(ax, 145, 39.5, 26, 7.6, "Mean + calibrated\nuncertainty", GOLD, tc=WHITE, ec=GOLD, fs=9.0)
    arrow(ax, 122, 53.0, 132, 59.0, lw=1.8)
    arrow(ax, 145, 56.4, 145, 53.6, lw=1.6)
    arrow(ax, 145, 46.5, 145, 43.4, lw=1.6)
    arrow(ax, 140, 20.6, 143, 35.7, color=GREEN, lw=1.6)

    ax.text(80, 6.6, "6.4 M parameters", ha="center", fontsize=8.6, color=GREY, style="italic")
    legend(ax, 3.0, [("input", "input", 8), ("backbone", "backbone", 33),
                     ("ours", "COMPASS contribution", 62), ("new", "new in this work", 108),
                     ("out", "output", 140)])

    out = REPO / "figures" / "architecture"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "compass_wnet.png", dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"[arch] wrote {out / 'compass_wnet.png'}")


# ---------------------------------------------------------------- residual pipeline
def build_residual():
    fig, ax = _canvas()
    ax.text(80, 84, "Real-data reconstruction: learned residual on the classical field",
            ha="center", fontsize=15, weight="bold", color=INK)
    ax.text(80, 78.5, "Keep the strong classical estimate, learn only the geometry-driven correction it misses",
            ha="center", fontsize=9.5, color=GREY)

    box(ax, 22, 58, 28, 10, "Sparse RSS\nreference points", INPUT_FILL, ec=INPUT_EDGE, fs=9.5)
    box(ax, 62, 58, 30, 10, "Classical field\nRBF multiquadric", BLUE_FILL, ec=BLUE, fs=9.5)
    ax.add_patch(Circle((100, 58), 3.2, facecolor=WHITE, edgecolor=INK, linewidth=1.7, zorder=5))
    ax.text(100, 58, "+", ha="center", va="center", fontsize=15, weight="bold", color=INK, zorder=6)
    box(ax, 136, 58, 30, 10, "Corrected\nradio map", GOLD, tc=WHITE, ec=GOLD, fs=10)

    box(ax, 55, 28, 34, 10, "Wall geometry\n+ distance features", GREEN_FILL, ec=GREEN, fs=9.5)
    box(ax, 100, 28, 30, 10, "Learned residual\ngradient boosting", GREEN_FILL, ec=GREEN, fs=9.5)

    arrow(ax, 36, 58, 47, 58)
    arrow(ax, 77, 58, 96.6, 58)
    ax.text(87, 60.5, "classical estimate", ha="center", fontsize=8.2, color=GREY)
    arrow(ax, 103.4, 58, 121, 58)
    arrow(ax, 22, 53, 40, 33, color=GREEN, lw=1.5)
    arrow(ax, 66, 53, 78, 33, color=GREEN, lw=1.5)
    arrow(ax, 72, 28, 85, 28, color=GREEN, lw=1.6)
    arrow(ax, 100, 33, 100, 54.4, color=GREEN, lw=1.6)
    ax.text(103, 45, "residual", ha="left", fontsize=8.2, color=GREEN, weight="bold")

    ax.text(80, 13.5, "Trained leave-one-cell-out on real cells (no leakage). Beats the best classical "
            "reconstructor 5.49 to 5.14 dB, paired Wilcoxon p = 2e-4.",
            ha="center", fontsize=9.2, color=INK)
    legend(ax, 5.5, [("backbone", "classical backbone", 8), ("ours", "learned (COMPASS)", 64)])

    out = REPO / "figures" / "architecture"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "residual_pipeline.png", dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"[arch] wrote {out / 'residual_pipeline.png'}")


if __name__ == "__main__":
    build_wnet()
    build_residual()

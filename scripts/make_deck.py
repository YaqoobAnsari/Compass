#!/usr/bin/env python
"""
Generate the COMPASS presentation as a PDF, structured as a PAPER in slide form
(problem, related work, data, formulation, method, baselines, results, discussion,
conclusion), matching make_pptx.py. Formal, black text with green and red reserved for
key results and honest caveats, no em dashes.

  python scripts/make_deck.py --out /data1/yansari/Compass/COMPASS_results.pdf
"""

from __future__ import annotations

import argparse
import textwrap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

REPO = Path("/data1/yansari/Compass")
FIG = REPO / "figures"
BLACK = "#000000"
GREEN = "#1e7d4f"
RED = "#c0392b"
HDR = "#e9edf2"
GREY = "#888888"
W, H = 13.333, 7.5

import re


def _clean(s):
    """No semicolons, no em dashes (colons allowed for lists)."""
    if not isinstance(s, str):
        return s
    s = s.replace(" -- ", ", ").replace("\u2014", ", ").replace("\u2013", ", ")
    s = re.sub(r";\s+([a-z])", lambda m: ". " + m.group(1).upper(), s)
    s = re.sub(r";\s+", ", ", s)
    return s.replace(";", ".")


def new_slide(title):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.035, 0.93, _clean(title), color=BLACK, fontsize=20, weight="bold", va="center")
    ax.plot([0.035, 0.965], [0.885, 0.885], color=GREY, linewidth=1.2)
    ax.text(0.035, 0.03, "COMPASS: reconstructing indoor radio maps from real crowdsensed measurements",
            color=GREY, fontsize=8, va="center")
    return fig


def _count_lines(lines, width):
    """Wrapped-line count (plus inter-bullet gaps) a block would occupy at `width`."""
    n = 0.0
    for ln in lines:
        if ln == "":
            n += 0.5
            continue
        t = ln[3:] if ln[:3] in ("G| ", "R| ") else ln
        t = _clean(t.replace("**", ""))
        n += len(textwrap.fill(t.strip(), width=width).split("\n")) + 0.2
    return n


def bullets(fig, lines, x=0.035, y=0.82, width=112, fs=12, dy=0.045, bottom=0.075):
    """Lay out bullets, shrinking type and leading just enough to clear the footer.

    `dy` is per WRAPPED line, so long paragraphs can otherwise run off the page. We
    search downward from the requested size for the first that fits; a smaller font
    also fits more characters per line, so the wrap width grows with the shrink.
    """
    fs0, dy0, w0 = float(fs), float(dy), int(width)
    for k in range(26):
        f = max(7.5, fs0 * (1.0 - 0.03 * k))
        w = max(60, int(round(w0 * fs0 / f)))
        d = dy0 * f / fs0
        if _count_lines(lines, w) * d <= (y - bottom) or f <= 7.5:
            fs, width, dy = f, w, d
            break
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cy = y
    for ln in lines:
        if ln == "":
            cy -= dy * 0.5
            continue
        color, force_bold = BLACK, False
        if ln.startswith("G| "):
            color, force_bold, ln = GREEN, True, ln[3:]
        elif ln.startswith("R| "):
            color, force_bold, ln = RED, True, ln[3:]
        topic = ln.lstrip().startswith("**")          # black topic-sentence-led bullet
        sub = ln.startswith("  ")
        ln = _clean(ln.replace("**", ""))
        marker, indent = ("•  ", x) if not sub else ("   -  ", x + 0.025)
        wrapped = textwrap.fill(ln.strip(), width=width)
        segs = wrapped.split("\n")
        for i, seg in enumerate(segs):
            if force_bold:
                wt = "bold"                            # G|/R| result/caveat lines
            elif topic and i == 0:
                wt = "bold"                            # bold only the topic-sentence lead line
            else:
                wt = "normal"
            ax.text(indent, cy, (marker if i == 0 else "     ") + seg, fontsize=fs - (1 if sub else 0),
                    va="top", color=color, weight=wt)
            cy -= dy
        cy -= dy * 0.2
    return ax


def table(fig, headers, rows, rect, col_w=None, fs=11, hi_rows=None):
    hi_rows = hi_rows or {}
    rows = [[_clean(str(c)) for c in r] for r in rows]
    headers = [_clean(h) for h in headers]
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colWidths=col_w)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.5)
    for j in range(len(headers)):
        c = t[0, j]; c.set_facecolor(HDR); c.set_text_props(color=BLACK, weight="bold")
    for i in range(len(rows)):
        col = hi_rows.get(i, BLACK)
        for j in range(len(headers)):
            cell = t[i + 1, j]; cell.set_facecolor("white")
            cell.set_text_props(color=col, weight="bold" if i in hi_rows else "normal")
    return ax


def image(fig, path, rect):
    ax = fig.add_axes(rect); ax.axis("off")
    p = Path(path)
    if p.exists():
        ax.imshow(plt.imread(p))
    else:
        ax.text(0.5, 0.5, f"[figure: {p.name}]", ha="center", color=GREY)
    return ax


def fig_slide(figpath, caption=None):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    if caption:
        image(fig, figpath, [0.03, 0.12, 0.94, 0.84])
        fig.text(0.5, 0.06, _clean(caption), ha="center", va="center", fontsize=11, color=BLACK, wrap=True)
    else:
        image(fig, figpath, [0.02, 0.02, 0.96, 0.96])
    return fig


def build(out):
    from deck_content import slides

    pdf = PdfPages(out)

    def save(fig):
        pdf.savefig(fig); plt.close(fig)

    # ============================== TITLE ==============================
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.62, "COMPASS", color=BLACK, fontsize=58, weight="bold", ha="center")
    ax.text(0.5, 0.51, "Reconstructing indoor radio maps from real crowdsensed measurements",
            color=BLACK, fontsize=18, ha="center")
    ax.text(0.5, 0.40, "A geometry, order, and device conditioned reconstructor with explicit occlusion\n"
            "conditioning and calibrated uncertainty, validated on a large synthetic benchmark\n"
            "and a real indoor cellular crowdsensing dataset.",
            color=BLACK, fontsize=13, ha="center")
    save(fig)

    for spec in slides():
        if "fig" in spec:
            save(fig_slide(FIG / spec["fig"], spec.get("caption")))
            continue

        fig = new_slide(spec["title"])
        if "table" in spec:
            headers, rows = spec["table"]
            lead, after = spec["bullets"], spec.get("after", [])
            # lead block sits between the rule (0.885) and the table
            # Budget the slide explicitly: give the table only what its rows need, then
            # split the remaining vertical space between the lead and trailing sentences
            # in proportion to their length, so neither is ever squeezed below 8pt.
            n_rows = len(rows) + 1
            t_h = min(0.030 * n_rows + 0.03, 0.46)
            lead_n = _count_lines(lead, 112)
            after_n = _count_lines(after, 112) if after else 0.0
            text_space = max(0.12, 0.77 - t_h - 0.05)
            tot = lead_n + after_n
            lead_h = text_space * (lead_n / tot) if tot else text_space
            t_top = 0.845 - lead_h - 0.015
            t_bottom = t_top - t_h
            bullets(fig, lead, y=0.845, fs=11, bottom=t_top + 0.005)
            fs_t = 10.5 if len(rows) <= 6 else (9.5 if len(rows) <= 9 else 8.5)
            table(fig, headers, rows, rect=[0.03, t_bottom, 0.94, t_h],
                  col_w=[0.34, 0.16, 0.20, 0.14, 0.16][:len(headers)] if len(headers) == 5 else None,
                  fs=fs_t, hi_rows={0: GREEN})
            if after:
                bullets(fig, after, y=t_bottom - 0.025, fs=10.5, bottom=0.072)
        else:
            bullets(fig, spec["bullets"], y=0.82, fs=12)
        save(fig)

    pdf.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data1/yansari/Compass/COMPASS_results.pdf")
    args = ap.parse_args()
    print(f"[deck] wrote {build(args.out)}")


if __name__ == "__main__":
    main()

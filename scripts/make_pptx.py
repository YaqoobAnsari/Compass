#!/usr/bin/env python
"""
Generate the COMPASS presentation (.pptx) as a PAPER in slide form: it argues from the
problem (radio maps, crowdsensing, the ignored gap) through related work, data, problem
formulation, method, baselines, and every result, to discussion and conclusion. Formal,
self-contained, black text with green and red reserved for key results and honest
caveats. No em dashes.

  python scripts/make_pptx.py --out /data1/yansari/Compass/COMPASS_results.pptx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path("/data1/yansari/Compass")
RES = REPO / "results"
FIG = REPO / "figures"

BLACK = RGBColor(0x00, 0x00, 0x00)
GREEN = RGBColor(0x1E, 0x7D, 0x4F)
RED = RGBColor(0xC0, 0x39, 0x2B)
HDR = RGBColor(0xE9, 0xED, 0xF2)     # neutral grey table header fill
RULE = RGBColor(0x88, 0x88, 0x88)


import re


def _clean(s):
    """Enforce the writing rules: no semicolons, no em dashes. A semicolon becomes a
    period plus a capitalised next word (or a comma when followed by a digit); em/en
    dashes become commas. Colons are left untouched (allowed for lists)."""
    if not isinstance(s, str):
        return s
    s = s.replace(" -- ", ", ").replace("—", ", ").replace("–", ", ")
    s = re.sub(r";\s+([a-z])", lambda m: ". " + m.group(1).upper(), s)
    s = re.sub(r";\s+", ", ", s)
    return s.replace(";", ".")


def _load(p):
    try:
        return json.loads((RES / p).read_text())
    except Exception:
        return {}


def slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12.3), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = _clean(title)
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = BLACK
    ln = s.shapes.add_shape(1, Inches(0.5), Inches(1.0), Inches(12.33), Pt(1.4))
    ln.fill.solid(); ln.fill.fore_color.rgb = RULE
    ln.line.fill.background()
    return s


def bullets(s, lines, left=0.55, top=1.25, width=12.2, size=14, height=5.9):
    # never let a text box extend past the bottom of the slide (7.5 in)
    height = max(0.4, min(height, 7.5 - top - 0.08))
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for ln in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if ln == "":
            p.text = ""
            p.font.size = Pt(6)
            continue
        base, whole_bold = BLACK, False
        if ln.startswith("G| "):
            base, whole_bold, ln = GREEN, True, ln[3:]
        elif ln.startswith("R| "):
            base, whole_bold, ln = RED, True, ln[3:]
        sub = ln.startswith("  ")
        if sub:
            ln = ln.strip()
        ln = _clean(ln)
        marker = "    -  " if sub else "•  "
        r0 = p.add_run(); r0.text = marker
        r0.font.size = Pt(size - (1 if sub else 0)); r0.font.color.rgb = base; r0.font.bold = whole_bold
        for idx, seg in enumerate(ln.split("**")):
            if seg == "":
                continue
            r = p.add_run(); r.text = seg
            r.font.size = Pt(size - (1 if sub else 0)); r.font.color.rgb = base
            r.font.bold = whole_bold or (idx % 2 == 1)
        p.space_after = Pt(6)
    return tb


def table(s, headers, rows, left, top, width, height, col_w=None, hi_rows=None, fs=11):
    hi_rows = hi_rows or {}
    shp = s.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                             Inches(width), Inches(height))
    t = shp.table
    t.first_row = False
    t.horz_banding = False
    if col_w:
        for j, w in enumerate(col_w):
            t.columns[j].width = Inches(width * w)
    for j, h in enumerate(headers):
        c = t.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = HDR
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_top = Pt(2); c.margin_bottom = Pt(2)
        p = c.text_frame.paragraphs[0]
        p.text = _clean(h); p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(fs); p.font.bold = True; p.font.color.rgb = BLACK
    for i, row in enumerate(rows):
        color = hi_rows.get(i, BLACK)
        for j, val in enumerate(row):
            c = t.cell(i + 1, j)
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = Pt(1); c.margin_bottom = Pt(1)
            p = c.text_frame.paragraphs[0]
            p.text = _clean(str(val))
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            p.font.size = Pt(fs); p.font.bold = i in hi_rows; p.font.color.rgb = color
    return shp


def image(s, path, left, top, max_w, max_h):
    p = Path(path)
    if not p.exists():
        return
    iw, ih = Image.open(p).size
    ar = iw / ih
    w, h = max_w, max_w / ar
    if h > max_h:
        h, w = max_h, max_h * ar
    s.shapes.add_picture(str(p), Inches(left + (max_w - w) / 2), Inches(top + (max_h - h) / 2),
                         width=Inches(w), height=Inches(h))


def fig_slide(prs, figpath, caption=None):
    """A dedicated slide holding one large, publication-quality figure (its own
    title is baked into the image), with an optional one-line caption."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if caption:
        image(s, figpath, 0.25, 0.2, 12.83, 6.35)
        tb = s.shapes.add_textbox(Inches(0.6), Inches(6.75), Inches(12.13), Inches(0.55))
        tb.text_frame.word_wrap = True
        pr = tb.text_frame.paragraphs[0]
        pr.text = _clean(caption)
        pr.font.size = Pt(12); pr.font.color.rgb = BLACK; pr.alignment = PP_ALIGN.CENTER
    else:
        image(s, figpath, 0.25, 0.2, 12.83, 7.1)
    return s


def _fit_size(lines, width_in, height_in, start=13.0, floor=8.5):
    """Largest font (down to `floor`) at which these bullets fit the given box.
    Mirrors the estimator in qa_layout.py so the two agree."""
    for k in range(40):
        sz = max(floor, start - 0.25 * k)
        total = 0.0
        for ln in lines:
            t = ln[3:] if ln[:3] in ("G| ", "R| ") else ln
            t = _clean(t.replace("**", ""))
            cpl = max(int(width_in * 96 / (sz * 0.52)), 10)
            total += max(1, -(-len(t) // cpl)) * sz * 1.22 + 6
        if total <= height_in * 72 * 0.98 or sz <= floor:
            return sz
    return floor


def build(out):
    from deck_content import slides

    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)

    # ============================== TITLE ==============================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(3.3))
    tf = tb.text_frame; tf.word_wrap = True
    p0 = tf.paragraphs[0]; p0.text = "COMPASS"; p0.font.size = Pt(52)
    p0.font.bold = True; p0.font.color.rgb = BLACK
    p2 = tf.add_paragraph()
    p2.text = "Reconstructing indoor radio maps from real crowdsensed measurements"
    p2.font.size = Pt(22); p2.font.color.rgb = BLACK; p2.space_before = Pt(12)
    p3 = tf.add_paragraph()
    p3.text = ("A geometry, order, and device conditioned reconstructor with explicit occlusion "
               "conditioning and calibrated uncertainty, validated on a large synthetic benchmark "
               "and a real indoor cellular crowdsensing dataset.")
    p3.font.size = Pt(14); p3.font.color.rgb = BLACK; p3.space_before = Pt(16)

    for spec in slides():
        if "fig" in spec:
            fig_slide(prs, FIG / spec["fig"], spec.get("caption"))
            continue

        sl = slide(prs, spec["title"])
        if "table" in spec:
            headers, rows = spec["table"]
            lead, after = spec["bullets"], spec.get("after", [])
            lead_h = 1.05
            lead_sz = _fit_size(lead, 12.2, lead_h, start=12.0)
            bullets(sl, lead, size=lead_sz, height=lead_h)
            t_top = 1.25 + lead_h + 0.05
            t_h = min(0.285 * (len(rows) + 1) + 0.12, 7.4 - t_top - (1.05 if after else 0.1))
            t_fs = 10.5 if len(rows) <= 6 else (9.5 if len(rows) <= 9 else 8.5)
            table(sl, headers, rows, left=0.4, top=t_top, width=12.5, height=t_h,
                  col_w=[0.36, 0.15, 0.19, 0.14, 0.16][:len(headers)] if len(headers) == 5 else None,
                  fs=t_fs, hi_rows={0: GREEN})
            if after:
                a_top = t_top + t_h + 0.10
                a_h = max(0.5, 7.42 - a_top)
                bullets(sl, after, top=a_top, size=_fit_size(after, 12.2, a_h, start=11.5), height=a_h)
        else:
            h = 6.1
            bullets(sl, spec["bullets"], size=_fit_size(spec["bullets"], 12.2, h), height=h)

    prs.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data1/yansari/Compass/COMPASS_results.pptx")
    args = ap.parse_args()
    print(f"[pptx] wrote {build(args.out)}")


if __name__ == "__main__":
    main()

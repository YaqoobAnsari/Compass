#!/usr/bin/env python
"""Layout QA: detect text overflow off the page, overlapping text blocks, and
unreadably small type, in both the PDF and PPTX decks."""
import sys
import pymupdf
from pptx import Presentation
from pptx.util import Emu

PDF = "/data1/yansari/Compass/COMPASS_results.pdf"
PPTX = "/data1/yansari/Compass/COMPASS_results.pptx"
MIN_PT = 8.0
problems = []

# ---------------- PDF ----------------
doc = pymupdf.open(PDF)
print(f"=== PDF: {doc.page_count} pages ===")
for pno, page in enumerate(doc, 1):
    W, H = page.rect.width, page.rect.height
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        txt = "".join(s["text"] for l in b.get("lines", []) for s in l.get("spans", []))
        if not txt.strip():
            continue
        sizes = [s["size"] for l in b.get("lines", []) for s in l.get("spans", [])]
        blocks.append((pymupdf.Rect(b["bbox"]), txt.strip(), min(sizes) if sizes else 99))
    # off-page
    for r, t, sz in blocks:
        if r.x0 < -1 or r.y0 < -1 or r.x1 > W + 1 or r.y1 > H + 1:
            problems.append(f"PDF p{pno}: OFF-PAGE text {r} vs page {W:.0f}x{H:.0f}: {t[:60]!r}")
        if sz < MIN_PT:
            problems.append(f"PDF p{pno}: TINY {sz:.1f}pt: {t[:60]!r}")
    # text crossing a drawn horizontal rule (title separators etc.)
    rules = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.height < 3 and r.width > W * 0.5:
            rules.append(r)
    for r, t, sz in blocks:
        for rule in rules:
            if r.y0 < rule.y0 - 1 and r.y1 > rule.y1 + 1 and r.x1 > rule.x0 and r.x0 < rule.x1:
                problems.append(f"PDF p{pno}: TEXT CROSSES RULE at y={rule.y0:.0f}: {t[:60]!r}")
    # text straddling the edge of a large drawn box (e.g. running behind a table)
    boxes = [dr["rect"] for dr in page.get_drawings()
             if dr["rect"].width * dr["rect"].height > 0.10 * W * H and dr["rect"].height > 20]
    for r, t_, sz in blocks:
        for bx in boxes:
            inter = r & bx
            if inter.is_empty:
                continue
            a = inter.width * inter.height
            ra = r.width * r.height
            if ra > 0 and 0.05 < a / ra < 0.95:
                problems.append(f"PDF p{pno}: TEXT STRADDLES TABLE EDGE ({a/ra:.0%} inside): {t_[:55]!r}")
    # overlaps (ignore tiny slivers)
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a, b = blocks[i][0], blocks[j][0]
            inter = a & b
            if inter.is_empty:
                continue
            area = inter.width * inter.height
            small = min(a.width * a.height, b.width * b.height)
            if small > 0 and area / small > 0.30 and area > 60:
                problems.append(f"PDF p{pno}: OVERLAP {area/small:.0%} between "
                                f"{blocks[i][1][:32]!r} and {blocks[j][1][:32]!r}")

# ---------------- PPTX ----------------
prs = Presentation(PPTX)
SW, SH = prs.slide_width, prs.slide_height
print(f"=== PPTX: {len(prs.slides)} slides ({SW/914400:.2f}x{SH/914400:.2f} in) ===")
for sno, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if sh.top is None or sh.left is None:
            continue
        r = sh.left + (sh.width or 0)
        b = sh.top + (sh.height or 0)
        if sh.left < -Emu(1) or sh.top < 0 or r > SW + Emu(1) or b > SH + Emu(1):
            nm = sh.name
            problems.append(f"PPTX s{sno}: shape past slide edge: {nm} "
                            f"right={r/914400:.2f}in bottom={b/914400:.2f}in")
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for run in p.runs:
                    if run.font.size and run.font.size.pt < MIN_PT:
                        problems.append(f"PPTX s{sno}: TINY {run.font.size.pt}pt: {run.text[:40]!r}")
    # estimate text overflow: rough line-count vs box height
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.height:
            continue
        total = 0.0
        for p in sh.text_frame.paragraphs:
            txt = "".join(r.text for r in p.runs)
            if not txt:
                continue
            sz = max([r.font.size.pt for r in p.runs if r.font.size] or [14])
            chars_per_line = max(int((sh.width / 914400) * 96 / (sz * 0.52)), 10)
            lines = max(1, -(-len(txt) // chars_per_line))
            total += lines * sz * 1.22 + 6
        if total > (sh.height / 914400) * 72 * 1.06:
            problems.append(f"PPTX s{sno}: TEXT OVERFLOW ~{total:.0f}pt of text in "
                            f"{(sh.height/914400)*72:.0f}pt box")

print()
if problems:
    print(f"!!! {len(problems)} LAYOUT PROBLEM(S):")
    for x in problems:
        print("  -", x)
    sys.exit(1)
print("LAYOUT CLEAN: no off-page text, no overlapping blocks, no type under 8pt.")

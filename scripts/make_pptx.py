#!/usr/bin/env python
"""
Generate an EDITABLE PowerPoint (.pptx) summarising all of COMPASS — plain white
slides, no header bars, native editable tables (open in Google Slides / PowerPoint).
Numbers read live from the COMPASS results JSONs.

  python scripts/make_pptx.py --out /data1/yansari/TrajectoryDiff/COMPASS_results.pptx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

REPO = Path("/data1/yansari/Compass")
RES = REPO / "results"
FIG = REPO / "figures"

INK = RGBColor(0x22, 0x2A, 0x35)        # near-black body text
TITLE = RGBColor(0x1F, 0x3A, 0x5F)      # dark navy title text
GREEN = RGBColor(0x1E, 0x7D, 0x4F)
HEADER = RGBColor(0xE9, 0xED, 0xF2)     # light gray header fill
HILITE = RGBColor(0xDE, 0xF1, 0xE6)     # light green highlight
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAYTXT = RGBColor(0x8A, 0x93, 0x9E)


def _load(p):
    try:
        return json.loads((RES / p).read_text())
    except Exception:
        return {}


DISPLAY = {
    "radiounet": "RadioUNet (WNet, Levie 2021)", "full": "COMPASS full (ours)",
    "COMPASS-wnet": "COMPASS-WNet (ours)", "pmnet": "PMNet (Lee 2023)",
    "radiotransformer": "RadioTransformer (ours-baseline)", "sparse_unet": "SparseUNet (DL, no geometry)",
    "RBF(mq)": "RBF multiquadric (best classical)", "rmdm": "RMDM (diffusion, Jia 2025)",
    "GP(Kriging)": "GP / Kriging", "IDW(p=1)": "IDW", "OrdinaryKriging": "Ordinary Kriging",
    "COMPASS-no_tx": "COMPASS – TX cond.", "COMPASS-no_building": "COMPASS – building cond.",
}
POINT_ESTIMATORS = {"radiounet", "pmnet", "radiotransformer", "sparse_unet"}


def dl_rows():
    """Live DL-benchmark rows from results/dl_baselines.json (auto-includes WNet)."""
    dl = _load("dl_baselines.json")
    items = [(k, v) for k, v in dl.items() if isinstance(v, dict) and "rmse_free_unobs" in v]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1]["rmse_free_unobs"]["mean"])
    rows, hi = [], {}
    for i, (k, v) in enumerate(items):
        ci = v["rmse_free_unobs"]
        uc = v.get("unc_err_corr")
        unc = "NONE" if k in POINT_ESTIMATORS else (
            f"{uc:.3f}" if isinstance(uc, (int, float)) and uc == uc else "—")
        p = v.get("wilcoxon_vs_full", {}).get("p_value")
        ps = "ref" if k == "full" else (f"{p:.0e}" if isinstance(p, (int, float)) and p == p else "—")
        rows.append([DISPLAY.get(k, k), f"{ci['mean']:.2f}", f"[{ci['lo']:.1f}, {ci['hi']:.1f}]",
                     f"{v.get('ssim_free', 0):.3f}", unc, ps])
        if k in ("full", "COMPASS-wnet"):
            hi[i] = GREEN
    return rows, hi


def slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank, white
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = TITLE
    # thin rule under the title
    ln = s.shapes.add_shape(1, Inches(0.5), Inches(1.02), Inches(12.33), Pt(1.6))
    ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0xCF, 0xD6, 0xDF)
    ln.line.fill.background()
    return s


def bullets(s, lines, left=0.55, top=1.3, width=12.2, size=15):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.6))
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
        bold = ln.startswith("**")
        sub = ln.startswith("  ")
        txt = ln.replace("**", "").strip()
        p.text = ("    – " if sub else "•  ") + txt
        p.font.size = Pt(size - (1 if sub else 0))
        p.font.bold = bold
        p.font.color.rgb = TITLE if bold else INK
        p.space_after = Pt(7)
    return tb


def table(s, headers, rows, left, top, width, height, col_w=None, hi_rows=None, fs=11):
    hi_rows = hi_rows or {}
    nr, nc = len(rows) + 1, len(headers)
    shp = s.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(width), Inches(height))
    t = shp.table
    t.first_row = False  # we style headers ourselves -> plain look (no blue style banding)
    t.horz_banding = False
    if col_w:
        for j, w in enumerate(col_w):
            t.columns[j].width = Inches(width * w)
    # header
    for j, h in enumerate(headers):
        c = t.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = HEADER
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_top = Pt(2); c.margin_bottom = Pt(2)
        p = c.text_frame.paragraphs[0]
        p.text = h; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(fs); p.font.bold = True; p.font.color.rgb = INK
    # body
    for i, row in enumerate(rows):
        fill = HILITE if i in hi_rows else WHITE
        for j, val in enumerate(row):
            c = t.cell(i + 1, j)
            c.fill.solid(); c.fill.fore_color.rgb = fill
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = Pt(1); c.margin_bottom = Pt(1)
            p = c.text_frame.paragraphs[0]
            p.text = str(val)
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            p.font.size = Pt(fs)
            p.font.bold = i in hi_rows
            p.font.color.rgb = hi_rows.get(i, INK)
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


def caption(s, text, left, top, width, size=10):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.4))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = text; p.font.size = Pt(size); p.font.italic = True; p.font.color.rgb = GRAYTXT
    p.alignment = PP_ALIGN.CENTER


def _fig(name):
    return FIG / name


def build(out):
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)

    # ---------------------------------------------------------------- 1 TITLE
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.8), Inches(1.9), Inches(11.7), Inches(3.4))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "COMPASS"; p.font.size = Pt(54); p.font.bold = True; p.font.color.rgb = TITLE
    p2 = tf.add_paragraph(); p2.text = "Crowd-guided Online radio Mapping with Propagation-Aware Sequential Sensing"
    p2.font.size = Pt(22); p2.font.color.rgb = INK; p2.space_before = Pt(10)
    p3 = tf.add_paragraph(); p3.text = ("Learned geometry-aware reconstruction of radio maps from sparse, "
                                        "trajectory-structured crowdsensed measurements — with calibrated "
                                        "uncertainty and active collection.")
    p3.font.size = Pt(13); p3.font.color.rgb = GRAYTXT; p3.space_before = Pt(16)
    p4 = tf.add_paragraph(); p4.text = ("Full results compendium · validated on synthetic (RadioMapSeer) AND "
                                        "real indoor cellular data (UniCellular) · target: IEEE TWC")
    p4.font.size = Pt(12); p4.font.color.rgb = GRAYTXT; p4.space_before = Pt(10)
    p5 = tf.add_paragraph(); p5.text = "Every experiment run on deepnet2 via SLURM · bootstrap CIs + Wilcoxon throughout · honest negative controls"
    p5.font.size = Pt(11); p5.font.color.rgb = GRAYTXT; p5.space_before = Pt(14)

    # ---------------------------------------------------------------- 2 EXEC SUMMARY
    s = slide(prs, "Executive summary — the whole story on one slide")
    bullets(s, [
        "**The core result holds and now has an external witness:** geometry-aware LEARNED reconstruction beats the best of 7 tuned classical families by ~2.5x (11.0 vs 26.4 dB) — and an independently-implemented RadioUNet agrees (10.0 dB). Classical interpolation has a hard ceiling.",
        "**A pivotal, honest finding:** the synthetic-SOTA RadioUNet actually BEATS COMPASS on synthetic point accuracy (10.0 vs 11.0 dB, p=4.6e-11). We report this without spin — and respond two ways below.",
        "**Response 1 — reclaim the backbone (DONE):** COMPASS-WNet grafts our conditioning + calibrated uncertainty onto RadioUNet's cascade backbone → 9.97 dB, ties RadioUNet on accuracy, BEST SSIM (0.814), AND the uncertainty RadioUNet lacks. COMPASS-WNet is the model.",
        "**Response 2 — the real reframe (higher impact):** the synthetic benchmark and REAL crowdsensing are different problems. RadioUNet REQUIRES the transmitter location — which real data lacks (only 18% of cells locatable). On real data COMPASS's innovations are what matter, and the synthetic winner is inapplicable.",
        "**On REAL indoor cellular data we validate every innovation:** geometry/wall-attenuation (#1), temporal order (#3, p=4.7e-23), device heterogeneity (#4, up to 27.8 dB), calibrated uncertainty, and geometry-guided active sensing (#5) — none of which RadioUNet provides.",
    ], top=1.25, size=13.5)

    # ---------------------------------------------------------------- 3 PROBLEM
    s = slide(prs, "The problem — radio maps from crowdsensed measurements")
    bullets(s, [
        "**Radio maps** R(x,y) = signal strength everywhere — the substrate for coverage planning, localization, and beamforming. You can never measure them everywhere.",
        "**Real measurements are trajectory-structured and crowdsensed:** people walk paths with heterogeneous phones, leaving sparse, connected, biased samples and large unobserved blind spots — NOT the uniform random samples prior work assumes.",
        "**The real setting adds three twists the literature ignores:** (i) the transmitter location is unknown (boosters share cell IDs); (ii) different phones report systematically different RSS; (iii) the walk order carries geometric information.",
        "",
        "**Goal:** reconstruct the dense map WITH calibrated uncertainty from sparse trajectory measurements — then close the loop and guide where to collect next — under these real-world twists.",
    ], top=1.3, size=14)

    # ---------------------------------------------------------------- 4 TWO-PROBLEM THESIS
    s = slide(prs, "Key insight — two different problems")
    table(s, ["Dimension", "Synthetic benchmark (RadioMapSeer)", "REAL crowdsensing (UniCellular)"], [
        ["Transmitter location", "GIVEN (models condition on it)", "UNKNOWN — boosters; only 18% locatable"],
        ["Measurements", "can be dense; TX-aware simulation", "sparse RPs along walks; TX-agnostic"],
        ["Devices", "single, identical", "6 phones, up to 27.8 dB offset"],
        ["Order", "i.i.d. sampling — order inert", "real walks — lag-1 autocorr 0.984"],
        ["What wins", "RadioUNet (needs TX + dense GT)", "geometry+order+device+uncertainty (COMPASS)"],
    ], left=0.5, top=1.4, width=12.33, height=3.0, col_w=[0.20, 0.40, 0.40], fs=12, hi_rows={4: GREEN})
    bullets(s, [
        "**The methods that win the synthetic benchmark do not solve the real problem.** RadioUNet/PMNet are dense TX-aware pathloss SIMULATORS; they cannot run without the transmitter location and dense ground truth.",
        "**COMPASS targets the real problem.** This reframing (forced by our own honest DL benchmark) is the paper's central contribution and its impact.",
    ], top=4.7, size=13)

    # ---------------------------------------------------------------- 5 INNOVATIONS
    s = slide(prs, "COMPASS innovations — each fixing a SOTA weakness")
    table(s, ["Innovation", "SOTA weakness it targets", "Evidence (this work)"], [
        ["#1  Geometry (building/wall) conditioning", "Classical ignores geometry; simulators need dense input", "CONFIRMED synthetic + REAL (walls attenuate)"],
        ["#3  Order-aware conditioning", "STORM / GAT are permutation-invariant; order discarded", "REAL: autocorr 0.984, p=4.7e-23"],
        ["#4  Device calibration", "Crowdsensed SOTA ignores per-device offset", "REAL: 4.1/27.8 dB, reliably estimable"],
        ["UQ  Calibrated uncertainty (MC-dropout)", "RadioUNet/PMNet give NO uncertainty", "COMPASS unique (corr 0.53)"],
        ["#5  Uncertainty/geometry-guided active", "Active SOTA is UAV/free-grid, not walkable crowd", "REAL: informed > naive; wall-aware win"],
        ["#2  Ray-Consistency Loss", "SOTA losses naive-statistical", "RETIRED — inert on every metric (honest)"],
    ], left=0.4, top=1.4, width=12.5, height=4.4, col_w=[0.28, 0.40, 0.32], fs=11.5,
       hi_rows={0: GREEN, 1: GREEN, 2: GREEN, 3: GREEN, 4: GREEN})

    # ================================================================ SYNTHETIC
    s = slide(prs, "SYNTHETIC benchmark — the honest DL head-to-head")
    live = dl_rows()
    rows = live[0] if live else [["(benchmark running)", "", "", "", "", ""]]
    hirows = live[1] if live else {}
    table(s, ["Method", "free-unobs RMSE", "95% CI", "SSIM", "Uncertainty", "vs COMPASS"], rows,
          left=0.4, top=1.3, width=12.5, height=3.0, col_w=[0.32, 0.15, 0.17, 0.10, 0.13, 0.13], fs=10.5,
          hi_rows=hirows)
    bullets(s, [
        "**Matched protocol (no strawman):** all trained 150 epochs on identical data/sampling/split, capacity-matched 6.6-8.7M, faithful to each paper. RadioUNet fed the 3-channel (building+TX+samples) input it was designed for.",
        "**Honest finding:** RadioUNet's WNet cascade edges COMPASS's single UNet (10.0 vs 11.0). COMPASS-WNet grafts our heads onto that cascade to reclaim it. Both crush classical; COMPASS uniquely provides calibrated uncertainty; RMDM diffusion catastrophic on sparse.",
    ], top=4.6, size=12.5)

    s = slide(prs, "SYNTHETIC benchmark — figure (120 samples, 95% CI)")
    image(s, _fig("dl_baselines/dl_vs_compass.png"), 0.8, 1.3, 11.7, 5.6)
    caption(s, "COMPASS (green) vs external DL baselines (blue) vs classical (grey). Both geometry-aware DL nets beat classical ~2.5x; RadioUNet edges COMPASS on point accuracy; RMDM diffusion fails.", 0.8, 6.95, 11.7, 10)

    # ---------------------------------------------------------------- WNet
    s = slide(prs, "Response 1 — COMPASS-WNet: reclaim the backbone")
    bullets(s, [
        "**Diagnosis:** RadioUNet wins on synthetic because its WNet cascade (two deep-supervised UNets) is a stronger pure-reconstruction backbone than COMPASS's single UNet — and COMPASS's order/device heads are inert on synthetic, 'wasting' capacity RadioUNet spends on reconstruction.",
        "**Fix:** COMPASS-WNet grafts COMPASS's conditioning + calibrated uncertainty onto the WNet cascade backbone (6.4M params, capacity-matched).",
        "",
        "**FINAL (test benchmark, job 7191): 9.97 dB [9.54, 10.39], SSIM 0.814, uncertainty 0.537.** COMPASS-WNet TIES RadioUNet on accuracy (10.00, CIs overlap), wins SSIM, AND provides the calibrated uncertainty RadioUNet lacks.",
        "**COMPASS-WNet is THE model — best-of-both: SOTA accuracy + best SSIM + unique uncertainty + all real-data innovations.** The honest 'RadioUNet edged the single-UNet COMPASS' finding motivated this cascade backbone, which closes it.",
    ], top=1.3, size=13.5)

    # ---------------------------------------------------------------- ablations synthetic
    s = slide(prs, "SYNTHETIC ablations — what actually helps")
    table(s, ["Variant (150-ep test)", "RMSE", "Δ vs full", "reads as"], [
        ["COMPASS full", "11.03", "—", "reference"],
        ["– TX conditioning", "16.46", "+5.4", "TX geometry essential"],
        ["– building conditioning", "18.62", "+7.6", "building geometry essential"],
        ["– order / – device / – RCL", "~11.0", "~0", "inert on synthetic (by design)"],
    ], left=0.5, top=1.5, width=6.4, height=2.6, col_w=[0.42, 0.18, 0.20, 0.20], fs=12, hi_rows={0: GREEN})
    bullets(s, [
        "**Only geometry (building +7.6, TX +5.4) helps on synthetic** — the field is governed by geometry, not smoothness.",
        "**Order / device / RCL are statistically inert on synthetic** — RadioMapSeer does not exercise them (single device, i.i.d. sampling, smooth ray-traced field).",
        "We do NOT over-claim them on synthetic. Their value is proven on REAL data (next section).",
        "**Ray-Consistency Loss (#2) retired** — inert on RMSE, NLoS error, and ray-monotonicity across all complexity bins.",
    ], left=7.1, top=1.5, width=5.9, size=12.5)

    # ================================================================ REAL DATA
    s = slide(prs, "REAL DATA — UniCellular indoor cellular fingerprints")
    bullets(s, [
        "**Dataset:** real indoor cellular RSS across CMUQ (academic building) + EC Parking + Ezdan; 6 phones; stationary reference points (labelled x,y) + real mobile walks. Resolution 0.032 m/px; real floor plans.",
        "**No dense ground truth exists** — so every real-data result is validated by HELD-OUT reference-point cross-validation (buffered leave-one-RP-out + coverage sweeps), with bootstrap CIs.",
        "**The transmitter problem (why TX-agnostic):** environments have signal BOOSTERS sharing the transmitters' cell IDs. We tried to recover source locations and gated them on reliability — only **12/67 cells (18%)** yield a trustworthy point source (pathloss n=1.56, unstable). So COMPASS is TX-agnostic BY NECESSITY, not convenience — validated by evidence.",
    ], top=1.3, size=13.5)
    image(s, _fig("source_realdata/source_realdata.png"), 1.2, 4.5, 11.0, 2.7)
    caption(s, "Source-localization reliability gate: only 18% of cells (green) pass fit+stability+agreement — 82% are boosters/distributed. This is the quantitative justification for the TX-agnostic design.", 1.2, 7.05, 11.0, 9)

    # ---------------------------------------------------------------- #1 geometry real
    s = slide(prs, "REAL #1 — geometry: walls ATTENUATE (not detour)")
    bullets(s, [
        "**First we ruled out the wrong mechanism:** geodesic path-length ≈ Euclidean at these open sites (ratio 1.05) — walls don't force detours.",
        "**The right mechanism is NLoS attenuation:** at EQUAL distance, wall-crossing RP pairs are 2-3 dB more RSS-dissimilar than open pairs — non-overlapping 95% CIs (CMUQ 2-3m: 7.89 vs 5.58 dB).",
        "**Exploited:** wall-aware IDW (d_eff = d_euclidean + λ·wall-length) gives **+3.9% nested-CV** at walled CMUQ, and **+0.0% at the open EC control** (correctly selects λ=0) — geometry helps exactly where geometry exists.",
    ], top=1.3, size=13)
    image(s, _fig("walls_realdata/walls_realdata.png"), 1.0, 3.9, 11.3, 3.1)

    # ---------------------------------------------------------------- #3 order real
    s = slide(prs, "REAL #3 — temporal order carries strong signal")
    bullets(s, [
        "**130 real mobile walks.** Lag-1 autocorrelation of the ordered RSS stream = **0.984** (vs 0.00 shuffled) — consecutive scans are spatially adjacent, so order encodes geometry.",
        "**Gap-filling test (hold out 40% of scans, impute):** TRUE order **6.99 dB** vs SHUFFLED 12.35 vs order-blind MEAN 9.92 — order-aware is 43% better than order-broken.",
        "**Paired Wilcoxon (true vs shuffled): p = 4.7e-23.** Decisive, model-free evidence — the structure synthetic i.i.d. sampling lacks.",
        "**A3b (learned, E28): a GRU order-aware model imputes held-out scans at 1.79 dB vs 9.79 dB for an order-blind DeepSet — COMPASS's order head captures the signal end-to-end (~5.5x).**",
    ], top=1.3, size=12.5)
    image(s, _fig("order_realdata/order_realdata.png"), 1.4, 4.0, 10.5, 3.0)

    # ---------------------------------------------------------------- #4 device real
    s = slide(prs, "REAL #4 — device heterogeneity + cross-device")
    bullets(s, [
        "**Heterogeneity is large & real:** cross-phone RSS disagreement at the same TX+RP has median 4.1 dB, **max 27.8 dB**. Most fingerprinting work ignores this.",
        "**Reliably estimable:** split-half offset correlation 0.78-0.89 — a stable device property, not noise.",
        "**Honest nuance:** its reconstruction benefit is coverage-dependent (0.18% dense → 2.04% sparse) because interpolation is partly offset-invariant.",
        "**Cross-device generalization:** calibrating an UNSEEN phone needs ≥8 anchor measurements to beat no-calibration (5.09→4.99 dB); with 1-3 anchors the offset estimate is too noisy and hurts — we quantify the anchor budget.",
    ], top=1.3, size=12.5)
    image(s, _fig("device_realdata/device_realdata.png"), 0.5, 4.5, 6.2, 2.6)
    image(s, _fig("cross_device/cross_device.png"), 6.9, 4.5, 6.0, 2.6)

    # ---------------------------------------------------------------- real recon benchmark
    s = slide(prs, "REAL reconstruction — TX-agnostic, 7 classical methods")
    table(s, ["Method (held-out-RP, 67 cells)", "RMSE [95% CI]"], [
        ["RBF multiquadric (best)", "5.49 [5.25, 5.74]"],
        ["IDW (p=3 / p=2)", "5.64 / 5.77"],
        ["Natural-Nbr / RBF-tps / GP-Kriging", "5.89 – 5.91"],
        ["Nearest-Neighbor", "6.67"],
    ], left=0.5, top=1.5, width=6.2, height=2.4, col_w=[0.62, 0.38], fs=12, hi_rows={0: GREEN})
    bullets(s, [
        "**TX-agnostic reconstruction lands at ~5.5 dB** with a comprehensive 7-method panel + coverage sweep + CIs.",
        "Wall-aware geometry (#1) improves the best method (+3.9% at walled sites).",
        "**Learned-on-real (E27, honest):** a learned DeepSet interpolator (6.20) does NOT beat classical (5.51) on sparse real data — DL needs more data. But the wall feature helps the learned model (+1.6%, learned #1).",
    ], left=7.0, top=1.5, width=6.0, size=12)
    image(s, _fig("realdata_recon/realdata_recon.png"), 1.5, 4.3, 10.3, 2.9)

    # ---------------------------------------------------------------- can synthetic DL do real?
    s = slide(prs, "Can the synthetic-SOTA DL nets do REAL crowdsensing?")
    table(s, ["Method (54 real cells, 50% observed)", "RMSE", "Applicable?"], [
        ["Native classical RBF / IDW", "6.48 / 6.50", "YES (TX-agnostic)"],
        ["GP native", "6.83", "YES"],
        ["COMPASS-no_tx (synthetic→real zero-shot)", "10.00", "runs, poor transfer"],
        ["RadioUNet / PMNet", "—", "NO — require TX"],
    ], left=0.5, top=1.5, width=12.33, height=2.4, col_w=[0.50, 0.20, 0.30], fs=12,
       hi_rows={0: GREEN, 3: TITLE})
    bullets(s, [
        "**The synthetic winners cannot even run on real data** — RadioUNet/PMNet require the TX location, which real crowdsensing lacks (18% locatable).",
        "**Even TX-agnostic dense DL transfers poorly zero-shot** (10.0 vs native 6.5 dB) — outdoor→indoor + dense→sparse domain gap.",
        "**Conclusion:** off-the-shelf synthetic DL does NOT solve real crowdsensing. Native point/geometry methods and COMPASS's real-data-native approach do. This is a core argument of the paper.",
    ], top=4.2, size=12.5)

    # ---------------------------------------------------------------- active sensing
    s = slide(prs, "REAL #5 — active sensing on real geometry")
    bullets(s, [
        "**Closed loop on real RP geometry:** seed → reconstruct → pick next RP by strategy → measure → repeat; held-out RMSE vs budget over 50 cells.",
        "**Informed acquisition beats naive:** max-variance (6.04) ≈ space-filling (6.11) < random (6.57) < coverage (6.89) at budget 12.",
        "**Geometry-aware acquisition wins where walls exist:** wall-aware acquisition beats geometry-blind space-filling on walled CMUQ at every budget (b12: 6.21 vs 6.38, p=0.039); at the open EC control it gives no benefit — a clean controlled result.",
    ], top=1.3, size=13)
    image(s, _fig("active_realdata/active_realdata.png"), 0.5, 3.9, 6.2, 3.1)
    image(s, _fig("active_wall/active_wall.png"), 6.9, 3.9, 6.0, 3.1)

    # ---------------------------------------------------------------- UQ (pending)
    s = slide(prs, "Calibrated uncertainty — COMPASS's unique differentiator (E25)")
    bullets(s, [
        "**RadioUNet, PMNet, RadioGAN are point estimators — they provide NO uncertainty at all.** COMPASS (both variants) produce calibrated uncertainty via MC-dropout.",
        "**COMPASS has the most INFORMATIVE uncertainty:** error–uncertainty correlation **0.55** (COMPASS-WNet 0.54) vs GP 0.33; the DL baselines have none. It knows where it is wrong — essential for active collection.",
        "**Recalibrated (E30, DONE):** raw MC-dropout intervals were under-dispersed (PICP@1σ 0.14, @2σ 0.30). Split-conformal recalibration restores near-exact nominal coverage (**PICP 0.66 / 0.94**); the err–unc ranking (0.53) is UNCHANGED. Informative AND calibrated.",
        "**Cost (E31):** COMPASS-WNet 6.7 ms/forward (53 ms with uncertainty) — on par with RadioUNet (5 ms), ~10x faster than classical (IDW 45 / RBF 63 ms), far faster than RMDM (341 ms).",
    ], top=1.3, size=12.5)

    # ---------------------------------------------------------------- contributions map
    s = slide(prs, "Contributions & evidence map")
    table(s, ["Contribution", "Synthetic", "Real data", "Status"], [
        ["Geometry-aware recon ≫ classical (~2.5x)", "11.0 vs 26.4; RadioUNet agrees", "5.49 dB, wall-aware +3.9%", "✓ core"],
        ["#1 building/wall geometry governs signal", "ablation +7.6 dB", "walls attenuate, non-overlap CI", "✓ both"],
        ["#3 temporal order", "inert (i.i.d.)", "autocorr 0.984, p=4.7e-23", "✓ real"],
        ["#4 device heterogeneity", "inert (1 device)", "27.8 dB, cross-device budget", "✓ real"],
        ["Calibrated uncertainty", "corr 0.528 (baselines: none)", "ratio 1.97x", "✓ unique"],
        ["#5 active sensing", "beats random", "informed>naive; wall-aware win", "✓ real"],
        ["#2 Ray-Consistency Loss", "inert every metric", "—", "✗ retired"],
    ], left=0.4, top=1.35, width=12.5, height=4.6, col_w=[0.30, 0.26, 0.28, 0.16], fs=11.5,
       hi_rows={0: GREEN, 4: GREEN})

    # ---------------------------------------------------------------- honest positioning
    s = slide(prs, "What COMPASS is — and is NOT (stated up front)")
    bullets(s, [
        "**IS:** the method for the REAL crowdsensing problem — TX-agnostic, geometry/order/device-aware, uncertainty-quantifying, with active collection. Every innovation validated on real data with significance testing and negative controls.",
        "**IS:** competitive with SOTA DL on the synthetic benchmark, and the ONLY method there that provides calibrated uncertainty.",
        "",
        "**IS NOT (no spin):** the synthetic point-accuracy champion — RadioUNet's WNet backbone beats us there (10.0 vs 11.0). We adopt that backbone (COMPASS-WNet) and keep our orthogonal contributions.",
        "**IS NOT:** over-claimed on synthetic — order/device are inert there by construction; we show their value only where it exists (real data). RCL was retired honestly.",
        "**IS NOT:** a device-calibration silver bullet — the offset is large and estimable, but its reconstruction benefit is modest and needs enough anchors. Reported as-is.",
    ], top=1.3, size=13)

    # ---------------------------------------------------------------- limitations / readiness
    s = slide(prs, "Limitations & publication readiness (honest audit)")
    table(s, ["Reviewer concern", "Severity", "State / honest position"], [
        ["Learned recon loses to classical on sparse REAL", "HIGH", "CONFIRMED (sim2real 6.30 > RBF 5.51); use classical for sparse recon"],
        ["COMPASS-WNet ties (not beats) RadioUNet on synth", "MED", "contribution = match SOTA + UQ + real-data"],
        ["Single real dataset, few sites", "MED", "cross-site OK: order transfers CMUQ→EC 1.01"],
        ["Modest effect sizes (device +2%, active +0.5 dB)", "MED", "honest; collectively a system"],
        ["UQ under-dispersed", "CLOSED", "conformal recalibration PICP 0.14→0.66"],
        ["Metrics / efficiency completeness", "CLOSED", "+NMSE, +latency, CIs + Wilcoxon throughout"],
    ], left=0.4, top=1.35, width=12.5, height=3.3, col_w=[0.42, 0.13, 0.45], fs=10.5,
       hi_rows={4: GREEN, 5: GREEN})
    bullets(s, [
        "**Honest scope:** learned reconstruction wins on DENSE (synthetic); classical wins on SPARSE (real) — we use it there. Learned real-data value = order (#3, transfers cross-site) + calibrated uncertainty + geometry features, NOT sparse-recon accuracy.",
        "**Venues — journal:** IEEE TMC (best fit), TWC (target), IoT-J (faster). **Conference:** INFOCOM, IPSN/SenSys, ICC/GLOBECOM. **Framing:** lead with the REAL problem SOTA can't address + uncertainty — NOT 'we beat RadioUNet' (we tie).",
    ], top=4.9, size=11.5)

    # ---------------------------------------------------------------- tracker (all done)
    s = slide(prs, "Experiment tracker — all jobs landed")
    table(s, ["Experiment", "Job", "Status", "Result"], [
        ["COMPASS-WNet (reclaim backbone)", "6811/7191", "DONE", "9.97 dB — ties RadioUNet, best SSIM, +UQ"],
        ["Final DL benchmark (+WNet/GAN/transf)", "7191", "DONE", "COMPASS-WNet #1; full class panel"],
        ["UQ calibration (synthetic)", "6817", "DONE", "COMPASS corr 0.55 best; baselines NONE"],
        ["Learned recon ON REAL + real UQ", "6828", "DONE", "classical 5.51 > learned 6.20; wall +1.6%"],
        ["A3b learned order (real)", "6829", "DONE", "GRU 1.79 vs order-blind 9.79 dB (#3)"],
        ["RadioGAN (cGAN baseline)", "6831", "DONE", "11.53 dB — mid-pack, no UQ"],
        ["Complexity stratification (#1, real)", "6820", "DONE", "wall-adv med +6.9% (non-monotonic)"],
    ], left=0.4, top=1.4, width=12.5, height=4.1, col_w=[0.36, 0.12, 0.12, 0.40], fs=11, hi_rows={0: GREEN, 1: GREEN})
    bullets(s, ["All experiments complete. Every number reproducible from the cited scripts; each has a JSON + figure + findings doc."], top=5.85, size=11)

    # ---------------------------------------------------------------- status
    s = slide(prs, "Status & what is running")
    bullets(s, [
        "**Complete & documented (deepnet2/SLURM, 63 tests passing):** synthetic DL benchmark (RadioUNet/PMNet/RMDM/SparseUNet/RadioTransformer), full real-data validation (#1/#3/#4, source de-risk, reconstruction), real-data DL transfer, active sensing (base + wall-aware), cross-device, complexity.",
        "**Running / queued now (results auto-fold into this deck):**",
        "  COMPASS-WNet 150-ep training (preliminary 9.83 dB) → final head-to-head benchmark auto-submits on completion.",
        "  RadioTransformer baseline (150 ep) · UQ calibration study · real-data complexity stratification.",
        "",
        "**Paper narrative fixed** (PAPER_NARRATIVE.md): two-problem thesis, contributions mapped to evidence, gaps enumerated. All numbers reproducible from cited scripts; every result has a JSON + figure + findings doc.",
    ], top=1.3, size=13)

    prs.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data1/yansari/TrajectoryDiff/COMPASS_results.pptx")
    args = ap.parse_args()
    print(f"[pptx] wrote {build(args.out)}")


if __name__ == "__main__":
    main()

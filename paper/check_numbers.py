#!/usr/bin/env python
"""Re-derive every number the prose asserts, straight from results/*.json.

Print this next to the .tex after any result file is regenerated. Anything that
disagrees is a stale number in the paper.

  python paper/check_numbers.py
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = lambda n: json.loads((REPO / "results" / n).read_text().replace("NaN", "null"))
d = R("dl_baselines.json")
m = lambda k: d[k]["rmse_free_unobs"]["mean"]

print("=" * 66)
print("SYNTHETIC  (results/dl_baselines.json)")
print("=" * 66)
cells = {"unet_noocc": "full", "unet_occ": "COMPASS-unet_occ",
         "wnet_noocc": "COMPASS-wnet_base", "wnet_occ": "COMPASS-wnet_occ",
         "wnet_meas": "COMPASS-wnet_occ_meas"}
for a, b in cells.items():
    c = d[b]["rmse_free_unobs"]
    print(f"  {a:11s} {c['mean']:7.3f} [{c['lo']:.3f},{c['hi']:.3f}]  "
          f"LoS {d[b]['rmse_los']:6.3f}  NLoS {d[b]['rmse_nlos']:6.3f}")
u0, u1, w0, w1 = (m(cells[k]) for k in ("unet_noocc", "unet_occ", "wnet_noocc", "wnet_occ"))
print(f"\n  occlusion in U-Net   {u0-u1:+.3f}   (paper: 2.04)")
print(f"  occlusion in WNet    {w0-w1:+.3f}   (paper: 1.35)")
print(f"  cascade, no occ      {u0-w0:+.3f}   (paper: 1.51)")
print(f"  joint                {u0-w1:+.3f}   (paper: 2.86)")
print(f"  sum of parts         {(u0-u1)+(u0-w0):+.3f}   (paper: 3.55)")
print(f"  INTERACTION          {((u0-u1)+(u0-w0))-(u0-w1):+.3f}   (paper: 0.69)")
print(f"\n  Tx-free vs Tx-anchored {m(cells['wnet_meas'])-w1:+.3f}  (paper: 1.25)")
print(f"  Tx-free vs no occ      {m(cells['wnet_meas'])-w0:+.3f}   (paper: -0.10)")
print(f"  radiounet              {m('radiounet'):.3f}")

panel = ["radiounet", "COMPASS-wnet_base", "radiomamba", "radiotransformer", "uram",
         "radiodiff", "radiogan", "full", "pmnet"]
los = np.array([d[k]["rmse_los"] for k in panel])
nl = np.array([d[k]["rmse_nlos"] for k in panel])
print(f"\n  panel LoS  {los.min():.2f}-{los.max():.2f}  CV {los.std()/los.mean()*100:.1f}%  (paper: 5.18-9.98, 20.1%)")
print(f"  panel NLoS {nl.min():.2f}-{nl.max():.2f}  CV {nl.std()/nl.mean()*100:.1f}%  (paper: 11.12-15.52, 9.6%)")
p2 = [k for k in panel if k != "pmnet"]
n2 = np.array([d[k]["rmse_nlos"] for k in p2])
print(f"  minus PMNet {n2.min():.2f}-{n2.max():.2f}  CV {n2.std()/n2.mean()*100:.1f}%  (paper: 11.12-12.84, 5.1%)")
sp = [m(k) for k in panel]
print(f"  panel RMSE span {max(sp)-min(sp):.2f} dB  (paper: 4.18)")
print(f"  no_tx  {m('COMPASS-no_tx')-u0:+.3f}  no_building {m('COMPASS-no_building')-u0:+.3f}  (paper: 4.76, 7.32)")
cl = [m(k) for k in ("RBF(mq)", "GP(Kriging)", "IDW(p=1)", "OrdinaryKriging")]
print(f"  classical span {min(cl):.2f}-{max(cl):.2f}  (paper: 25.6-28.1)")
print(f"  unc_err_corr wnet_occ {d['COMPASS-wnet_occ']['unc_err_corr']}  (paper: 0.55)")

pw = d.get("_pairwise_occlusion_cascade")
if pw:
    print("\n  PAIRED TESTS")
    for k, v in pw.items():
        print(f"    {k:34s} {v['delta_b_minus_a']:+6.3f}  p={v['p_value']:.2e}")
it = d.get("_interaction_occlusion_x_cascade")
if it:
    print(f"\n  INTERACTION CI  {it['mean']:+.3f} [{it['lo']:+.3f},{it['hi']:+.3f}]  "
          f"p={it['wilcoxon_vs_zero']['p_value']:.2e}")
else:
    print("\n  (no pairwise block yet in dl_baselines.json)")

print("\n" + "=" * 66)
print("REAL  (learned_real.json, deep_real.json, walls, source)")
print("=" * 66)
lr = R("learned_real.json")
for k in ("RBF(mq)", "learned+walls", "learned-nowalls"):
    print(f"  {k:16s} {lr['rmse'][k]:.3f}")
for k, v in lr["wilcoxon_learned_vs"].items():
    print(f"    vs {k:16s} p={v['p_value']:.2e}  n={v.get('n')}")
dr = R("deep_real.json")
print(f"  deep_real epochs={dr['epochs']}  residual-GBT {dr['overall']['residual-GBT']['mean']:.3f} "
      f"vs RBF {dr['overall']['RBF (base)']['mean']:.3f}  "
      f"p={dr['wilcoxon_vs_rbf']['residual-GBT']['p_value']:.2e}")
sr = R("source_realdata.json")
print(f"  locatable {sr['n_trustworthy']}/{sr['n_cells']} = {sr['frac_trustworthy']*100:.1f}%  r2_median {sr['r2_median']}")
ev = R("walls_realdata.json")["evidence"]
for site in ev:
    for b, x in ev[site].items():
        hi, lo = x["drss_high_ci"], x["drss_low_ci"]
        print(f"  {site:11s} {b:5s} delta {x['drss_high_wall']-x['drss_low_wall']:+.2f}  "
              f"disjoint={hi['lo']>lo['hi']}")
ps = R("walls_realdata.json")["per_site"]
for s, v in ps.items():
    print(f"  {s:11s} cv_gain {v['cv_gain_pct']}%  lambda picks {v['cv_lambda_picks']}")
print(f"  picp_1sigma {lr['real_uq']['picp_1sigma']}  (paper: 0.161)")

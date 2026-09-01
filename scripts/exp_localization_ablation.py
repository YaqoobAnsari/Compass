#!/usr/bin/env python
"""Ablation: does detection-dropout augmentation drive the device-invariant embedding win?"""
import json
from pathlib import Path
import numpy as np
from compass.eval.stats import paired_wilcoxon
from compass.realdata.localize import learned_eval_floor, load_floor

SF = [("cmuq", 1), ("cmuq", 2), ("cmuq", 3), ("ec_parking", 1)]
rng = np.random.default_rng(0)
full, noaug = [], []
for site, fl in SF:
    fd = load_floor(site, fl)
    if fd is None:
        continue
    f, _ = learned_eval_floor(fd, rng, aug=True)
    n, _ = learned_eval_floor(fd, rng, aug=False)
    full += f; noaug += n
    print(f"[{site}_f{fl}] aug={round(float(np.nanmean(f)),2)} no-aug={round(float(np.nanmean(n)),2)}")
a, b = np.array(full), np.array(noaug)
w = paired_wilcoxon(a, b)
out = {"aug_mean_m": round(float(a.mean()), 2), "noaug_mean_m": round(float(b.mean()), 2),
       "delta_m": round(float(b.mean() - a.mean()), 2), "wilcoxon_p": w.get("p_value"),
       "n_phones": len(a)}
Path("results/localization_ablation.json").write_text(json.dumps(out, indent=2))
print(f"\n  with detection-dropout aug : {out['aug_mean_m']} m")
print(f"  without aug                : {out['noaug_mean_m']} m  (delta +{out['delta_m']}, p={out['wilcoxon_p']:.3g})")

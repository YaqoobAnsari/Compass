"""Device-offset estimation recovers injected offsets and calibration is reliable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from compass.realdata.device import (
    calibration_reliability,
    cell_offset_spread,
    estimate_offsets,
    shared_cells,
)
from compass.realdata.unicellular import PHONE_COL, RSS_COL, TX_COL


def _synthetic(n_cells=40, seed=0):
    """(tx, rp) cells each seen by 4 phones with known per-phone offsets."""
    rng = np.random.default_rng(seed)
    phones = ["A", "B", "C", "D"]
    true_off = {"A": 0.0, "B": 4.0, "C": -3.0, "D": 1.5}
    rows = []
    for c in range(n_cells):
        base = -50 - rng.uniform(0, 40)  # cell's true field value
        for ph in phones:
            rows.append({TX_COL: c % 5, "rpNumber": c, PHONE_COL: ph,
                         RSS_COL: base + true_off[ph] + rng.normal(0, 0.5)})
    return pd.DataFrame(rows), true_off


def test_estimate_offsets_recovers_injected():
    g, true_off = _synthetic()
    sc = shared_cells(g, min_phones=2)
    est = estimate_offsets(sc)
    centred = {k: v - np.mean(list(true_off.values())) for k, v in true_off.items()}
    for ph, v in centred.items():
        assert abs(est[ph] - v) < 0.6, f"{ph}: est {est[ph]:.2f} vs true {v:.2f}"


def test_calibration_reliable():
    g, _ = _synthetic(n_cells=60)
    sc = shared_cells(g, min_phones=2)
    rel = calibration_reliability(sc, n_rep=10)
    assert rel["offset_corr_mean"] > 0.9  # offsets generalise across cell splits
    assert rel["var_reduction_mean"] > 0.7  # calibration removes most cross-device variance


def test_offset_spread_positive():
    g, _ = _synthetic()
    sc = shared_cells(g, min_phones=2)
    spread = cell_offset_spread(sc)
    assert spread.mean() > 3.0  # injected offsets span 7 dB

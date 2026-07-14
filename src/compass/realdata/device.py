"""
Device heterogeneity & cross-device calibration on real data (innovation #4).

Different phones report systematically different RSS for the *same* signal at the
*same* place (antenna, chipset, firmware). We (1) measure the offset magnitude at
reference points jointly observed by multiple phones, (2) estimate a per-device
offset by a one-way fixed-effect decomposition, (3) validate that the estimate is
RELIABLE (generalises to held-out shared cells — the user's "trends reliably" bar),
and (4) test whether calibrating before pooling improves held-out-RP reconstruction.

A "shared cell" is a (transmitter, reference-point) pair seen by >= 2 phones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .unicellular import PHONE_COL, RECON_COLS, RSS_COL, TX_COL, load_floor, valid_measurements


def device_long_table(site: str, floor: int, mode: str = "stationary") -> pd.DataFrame:
    """Per-(transmitter, reference-point, phone) mean RSS at labelled RPs."""
    df = load_floor(site, mode, floor, usecols=RECON_COLS)
    v = valid_measurements(df)
    v = v[v["rpNumber"] >= 1]
    g = v.groupby([TX_COL, "rpNumber", PHONE_COL])[RSS_COL].mean().reset_index()
    return g


def shared_cells(g: pd.DataFrame, min_phones: int = 2) -> pd.DataFrame:
    """Keep only (transmitter, RP) cells jointly observed by >= min_phones phones."""
    cnt = g.groupby([TX_COL, "rpNumber"])[PHONE_COL].nunique()
    keep = set(cnt[cnt >= min_phones].index)
    mask = g.set_index([TX_COL, "rpNumber"]).index.isin(keep)
    return g[mask.tolist() if hasattr(mask, "tolist") else mask].reset_index(drop=True)


def cell_offset_spread(shared: pd.DataFrame) -> np.ndarray:
    """Per shared-cell max-min RSS across phones (device disagreement, dB)."""
    return shared.groupby([TX_COL, "rpNumber"])[RSS_COL].agg(lambda s: s.max() - s.min()).to_numpy()


def estimate_offsets(shared: pd.DataFrame) -> pd.Series:
    """Per-device offset = mean residual after removing each shared cell's cross-phone mean
    (one-way fixed effect; centred so offsets sum-to-zero in expectation)."""
    s = shared.copy()
    s["cellmean"] = s.groupby([TX_COL, "rpNumber"])[RSS_COL].transform("mean")
    s["resid"] = s[RSS_COL] - s["cellmean"]
    off = s.groupby(PHONE_COL)["resid"].mean()
    return off - off.mean()  # re-centre to zero mean


def calibration_reliability(shared: pd.DataFrame, n_rep: int = 10, seed: int = 0) -> dict:
    """Split shared cells A/B; estimate offsets on A, test on B. Reports offset
    correlation across splits and the % of cross-device variance the A-offsets
    remove from B — i.e. does calibration generalise (trend reliably)?"""
    rng = np.random.default_rng(seed)
    cells = list(shared.groupby([TX_COL, "rpNumber"]).groups.keys())
    corrs, var_red = [], []
    for _ in range(n_rep):
        idx = rng.permutation(len(cells))
        a = set(cells[i] for i in idx[: len(cells) // 2])
        key = list(zip(shared[TX_COL], shared["rpNumber"]))
        ina = np.array([k in a for k in key])
        A, B = shared[ina], shared[~ina]
        if A[PHONE_COL].nunique() < 2 or B[PHONE_COL].nunique() < 2:
            continue
        oa, ob = estimate_offsets(A), estimate_offsets(B)
        common = oa.index.intersection(ob.index)
        if len(common) >= 3 and oa[common].std() > 1e-6 and ob[common].std() > 1e-6:
            corrs.append(float(np.corrcoef(oa[common], ob[common])[0, 1]))
        # variance reduction on B using A's offsets
        b = B.copy()
        b["off"] = b[PHONE_COL].map(oa).fillna(0.0)
        b["cal"] = b[RSS_COL] - b["off"]
        raw = b.groupby([TX_COL, "rpNumber"])[RSS_COL].var(ddof=0).mean()
        cal = b.groupby([TX_COL, "rpNumber"])["cal"].var(ddof=0).mean()
        if raw and np.isfinite(raw) and raw > 1e-9:
            var_red.append(float(1 - cal / raw))
    return {
        "offset_corr_mean": round(float(np.mean(corrs)), 3) if corrs else float("nan"),
        "offset_corr_std": round(float(np.std(corrs)), 3) if corrs else float("nan"),
        "var_reduction_mean": round(float(np.mean(var_red)), 3) if var_red else float("nan"),
        "n_splits": len(var_red),
    }

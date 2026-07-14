"""
Loader for the UniCellular indoor RSS fingerprint dataset (Eric / EricluvPython).

Real cellular RSS fingerprints across indoor sites in Qatar (CMUQ, EC Parking,
Ezdan Towers 1 & 4). Two modes:
  * STATIONARY — fixed reference points with pixel coords (x, y), 6 phones,
    ~150 scans each. Sparse but POSITION-LABELLED supervision.
  * MOBILE — continuous free walks; ordered by (phone, scanNumber/timeStamp);
    NO position labels (x=y=rpNumber=-1). Real TRAJECTORIES.

Key facts for COMPASS:
  * RSS in dBm (``transmitter_rss``); multiple transmitters per scan
    (``transmitter_id``); device id in ``phoneName`` (heterogeneity).
  * TX (cell) LOCATIONS are NOT provided — only cell IDs. No dense ground truth.
  * Geometry: floor-plan PNGs + RP-coordinate JSONs for some sites.

This module just reads/organises the CSVs; it does not impose RadioMapSeer
conventions (different problem: indoor, multi-cell, no TX coords).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATASET_ROOT = Path("/data1/yansari/Compass/external/unicellular-fingerprint-dataset")
SITES = ("cmuq", "ec_parking", "ezdan_tower1", "ezdan_tower4")
RSS_COL = "transmitter_rss"
PHONE_COL = "phoneName"
TX_COL = "transmitter_id"
INT_MAX = 2147483647  # Android "unavailable" sentinel; appears in transmitter_id too


def valid_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sentinel transmitters and missing/sentinel RSS — real cells only."""
    out = df[df[TX_COL] != INT_MAX]
    out = out[out[RSS_COL].notna() & (out[RSS_COL] > -200) & (out[RSS_COL] < 0)]
    return out


@dataclass
class UniCellularPaths:
    root: Path = DATASET_ROOT

    def data(self, site: str, mode: str) -> Path:
        return self.root / "data" / site / mode

    def floor_csv(self, site: str, mode: str, floor: int) -> Path:
        return self.data(site, mode) / f"floor{floor}.csv"

    def coords(self, site: str, floor: int) -> Path:
        return self.root / "coordinates" / site / f"floor{floor}.json"

    def floor_plan(self, site: str, floor: int) -> Path:
        return self.root / "floor_plans" / site / f"floor{floor}.png"

    def list_floors(self, site: str, mode: str) -> List[int]:
        d = self.data(site, mode)
        if not d.exists():
            return []
        floors = []
        for p in d.glob("floor*.csv"):
            if p.stat().st_size > 1000:  # skip unfetched LFS pointers (~131 B)
                try:
                    floors.append(int(p.stem.replace("floor", "")))
                except ValueError:
                    pass
        return sorted(floors)


def load_floor(site: str, mode: str, floor: int, root: Path = DATASET_ROOT,
               usecols=None) -> pd.DataFrame:
    """Load one floor CSV (optionally only ``usecols`` for speed on the large files).
    Raises a clear error if the LFS file was not fetched."""
    path = UniCellularPaths(root).floor_csv(site, mode, floor)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size < 1000:
        raise RuntimeError(
            f"{path} is a Git-LFS pointer ({path.stat().st_size} B), not data. "
            f"Run: git lfs pull --include='data/{site}/{mode}/**'"
        )
    return pd.read_csv(path, usecols=usecols)


# columns needed for reconstruction / interpolation (subset for speed on 128 MB files)
RECON_COLS = ["rpNumber", "x", "y", "transmitter_id", "transmitter_rss", "phoneName",
              "scanNumber", "timeStamp"]


def load_rp_coords(site: str, floor: int, root: Path = DATASET_ROOT) -> Dict[int, tuple]:
    """Reference-point pixel coordinates {rp_number: (x, y)} from the JSON, if present."""
    p = UniCellularPaths(root).coords(site, floor)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    # value is either [x, y] (CMUQ, EC Parking) or {"x":.., "y":..} (Ezdan)
    def xy(v):
        return (v["x"], v["y"]) if isinstance(v, dict) else (v[0], v[1])
    return {int(k): xy(v) for k, v in raw.items()}


def device_offsets_at_rp(df: pd.DataFrame, tx_id: int, rp: Optional[int] = None) -> pd.Series:
    """Mean RSS per phone for one transmitter (optionally at one RP) — device heterogeneity."""
    sub = valid_measurements(df)
    sub = sub[sub[TX_COL] == tx_id]
    if rp is not None and "rpNumber" in sub:
        sub = sub[sub["rpNumber"] == rp]
    return sub.groupby(PHONE_COL)[RSS_COL].mean().sort_values()


def best_shared_rp(df: pd.DataFrame, tx_id: int) -> Optional[int]:
    """RP where the most phones jointly observe ``tx_id`` (for a fair device comparison)."""
    sub = valid_measurements(df)
    sub = sub[(sub[TX_COL] == tx_id) & (sub.get("rpNumber", -1) >= 1)]
    if sub.empty:
        return None
    by_rp = sub.groupby("rpNumber")[PHONE_COL].nunique()
    return int(by_rp.idxmax())


def mobile_sequences(df: pd.DataFrame, tx_id: int) -> Dict[str, pd.DataFrame]:
    """Ordered RSS series for one transmitter along each phone's mobile walk."""
    out = {}
    sub = valid_measurements(df)
    sub = sub[sub[TX_COL] == tx_id]
    for phone, g in sub.groupby(PHONE_COL):
        g = g.sort_values(["scanNumber", "timeStamp"])
        out[phone] = g[["scanNumber", "timeStamp", RSS_COL]].reset_index(drop=True)
    return out


def top_transmitters(df: pd.DataFrame, n: int = 5, min_phones: int = 2) -> List[int]:
    """Most-observed REAL transmitter IDs (sentinel excluded, seen by >=min_phones)."""
    v = valid_measurements(df)
    by_phone = v.groupby(TX_COL)[PHONE_COL].nunique()
    eligible = by_phone[by_phone >= min_phones].index
    counts = v[v[TX_COL].isin(eligible)][TX_COL].value_counts()
    return counts.head(n).index.tolist()

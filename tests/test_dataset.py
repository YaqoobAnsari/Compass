"""Fast correctness tests for the COMPASS RadioMapSeer loader.

These lock in the two conventions v1 got wrong: street = PNG 0 (polarity) and
TX pixel = (H-1-y, x) (the y-flip). Uses a handful of maps so it runs on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compass.data import (
    RadioMapSeerDataset,
    list_map_ids,
    png_to_dbm,
    split_map_ids,
    tx_pixel_from_json,
)

ROOT = Path("/data1/yansari/Compass/data/raw")
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="RadioMapSeer data not available")


# --- conventions ------------------------------------------------------------
def test_png_to_dbm_endpoints():
    assert png_to_dbm(0) == pytest.approx(-186.0)
    assert png_to_dbm(255) == pytest.approx(-47.0)


def test_tx_pixel_flip():
    # antenna [x=108, y=68] on a 256 map -> (row=255-68, col=108)
    assert tx_pixel_from_json([108, 68]) == (187, 108)


# --- splits -----------------------------------------------------------------
def test_splits_disjoint_and_sized():
    s = split_map_ids(list_map_ids(ROOT))
    sets = [set(s[k]) for k in ("train", "val", "test")]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    total = sum(len(s[k]) for k in s)
    assert total == 701
    assert len(s["train"]) == 490 and len(s["val"]) == 105 and len(s["test"]) == 106


# --- dataset items ----------------------------------------------------------
@pytest.fixture(scope="module")
def ds():
    test_maps = split_map_ids(list_map_ids(ROOT))["test"][:2]
    return RadioMapSeerDataset(ROOT, map_ids=test_maps, variant="IRT2")


def test_sample_shapes_and_ranges(ds):
    s = ds[0]
    assert s["building_map"].shape == (1, 256, 256)
    assert s["radio_map"].shape == (1, 256, 256)
    assert float(s["radio_map"].min()) >= -1.0001 and float(s["radio_map"].max()) <= 1.0001
    dbm = s["radio_map_dbm"]
    assert float(dbm.min()) >= -186.01 and float(dbm.max()) <= -46.99
    assert set(np.unique(s["building_map"].numpy()).tolist()).issubset({0.0, 1.0})


def test_polarity_street_stronger_than_building(ds):
    # streets carry signal; building interiors sit at the floor
    for i in range(min(6, len(ds))):
        s = ds[i]
        free = s["free_mask"].numpy()[0].astype(bool)
        dbm = s["radio_map_dbm"].numpy()[0]
        assert dbm[free].mean() - dbm[~free].mean() > 10.0


def test_tx_pixel_is_gain_peak(ds):
    # the strongest-signal pixel must coincide with the computed TX pixel
    for i in range(min(6, len(ds))):
        s = ds[i]
        dbm = s["radio_map_dbm"].numpy()[0]
        tx_row, tx_col = (int(v) for v in s["tx_rowcol"].numpy())
        arg_row, arg_col = np.unravel_index(int(np.argmax(dbm)), dbm.shape)
        # IRT2 reflection hotspots can sit a few px off the antenna; transform is exact.
        assert max(abs(arg_row - tx_row), abs(arg_col - tx_col)) <= 3

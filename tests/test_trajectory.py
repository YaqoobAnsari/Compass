"""Trajectories must run ON STREETS, be ordered/timed, and yield real signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compass.data import RadioMapSeerDataset, list_map_ids, split_map_ids, street_mask
from compass.data.conditioning import SEQ_FEATURES, make_conditioning, sample_clean_dbm
from compass.data.noise import MeasurementNoise
from compass.data.trajectory import TrajectorySampler

ROOT = Path("/data1/yansari/Compass/data/raw")
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="RadioMapSeer data not available")


@pytest.fixture(scope="module")
def fixture():
    map_id = split_map_ids(list_map_ids(ROOT))["test"][0]
    ds = RadioMapSeerDataset(ROOT, map_ids=[map_id], variant="IRT2")
    building = ds.load_building(map_id)
    sample = ds[0]
    return building, sample


def test_trajectories_on_streets_and_ordered(fixture):
    building, _ = fixture
    free = street_mask(building)
    sampler = TrajectorySampler(seed=0)
    for tr in sampler.sample_many(building, k=3, n_points=80):
        assert len(tr) == 80
        assert np.all(np.diff(tr.t) >= -1e-6)  # time monotonic
        ri = np.clip(np.round(tr.rows).astype(int), 0, 255)
        ci = np.clip(np.round(tr.cols).astype(int), 0, 255)
        assert free[ri, ci].mean() >= 0.99  # on streets (the v1 fix)


def test_trajectories_sample_real_signal(fixture):
    building, sample = fixture
    free = street_mask(building)
    dbm = sample["radio_map_dbm"].numpy()[0]
    floor = dbm[~free].mean()
    sampler = TrajectorySampler(seed=1)
    tr = sampler.sample_many(building, k=1, n_points=100)[0]
    clean = sample_clean_dbm(dbm, tr.rows, tr.cols)
    assert clean.mean() - floor > 5.0  # real signal, not the floor


def test_conditioning_shapes_and_consistency(fixture):
    building, sample = fixture
    dbm = sample["radio_map_dbm"].numpy()[0]
    tx_rc = tuple(int(v) for v in sample["tx_rowcol"].numpy())
    cond = make_conditioning(building, dbm, tx_rc, TrajectorySampler(seed=2),
                             MeasurementNoise(), np.random.default_rng(0), k=3, n_points=60)
    assert cond.sparse_rss.shape == (256, 256)
    assert set(np.unique(cond.mask).tolist()).issubset({0.0, 1.0})
    # sparse RSS is non-zero only where observed
    assert np.all(cond.sparse_rss[cond.mask == 0] == 0)
    assert 0.0 < cond.coverage_fraction < 0.05
    for seq in cond.sequences:
        assert seq.shape[1] == len(SEQ_FEATURES)
        assert np.isfinite(seq).all()

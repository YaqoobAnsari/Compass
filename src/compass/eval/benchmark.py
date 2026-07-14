"""
Benchmark orchestrator: score every method on identical held-out test samples,
with full metrics + bootstrap CI + paired Wilcoxon. Classical and learned methods
see the SAME trajectories, so comparisons are fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..data.conditioning import SEQ_FEATURES, make_conditioning
from ..data.conventions import signed_unit_to_dbm
from ..data.noise import MeasurementNoise
from ..data.radiomapseer import RadioMapSeerDataset
from ..data.trajectory import TrajectorySampler
from ..recon.base import Observations
from .metrics import all_metrics
from .stats import bootstrap_ci, paired_wilcoxon


def _seq_tensor(sequences: List[np.ndarray], seq_len: int) -> torch.Tensor:
    seq = np.concatenate(sequences, axis=0) if sequences else np.zeros((1, len(SEQ_FEATURES)), np.float32)
    if len(seq) >= seq_len:
        seq = seq[:seq_len]
    else:
        seq = np.concatenate([seq, np.zeros((seq_len - len(seq), seq.shape[1]), np.float32)])
    return torch.from_numpy(seq.astype(np.float32))[None]


def build_eval_sample(base: RadioMapSeerDataset, sampler, noise, map_id, tx, rng,
                      k=3, n_points=100, seq_len=256):
    idx = base.samples.index((map_id, tx))
    s = base[idx]
    gt_dbm = s["radio_map_dbm"].numpy()[0]
    building_raw = base.load_building(map_id)
    tx_rc = tuple(int(v) for v in s["tx_rowcol"].numpy())
    shadow = noise.new_field(gt_dbm.shape, rng=rng) if noise is not None else None
    cond = make_conditioning(building_raw, gt_dbm, tx_rc, sampler, noise, rng,
                             k=k, n_points=n_points, shadow_field=shadow)
    # observations in dBm for classical methods
    rows = np.concatenate([np.clip(np.round(t.rows).astype(int), 0, 255) for t in cond.trajectories])
    cols = np.concatenate([np.clip(np.round(t.cols).astype(int), 0, 255) for t in cond.trajectories])
    vals = np.concatenate(cond.measured_dbm)
    obs = Observations(rows, cols, vals)
    batch = {
        "sparse_rss": torch.from_numpy(cond.sparse_rss[None, None]).float(),
        "mask": torch.from_numpy(cond.mask[None, None]).float(),
        "coverage": torch.from_numpy(cond.coverage[None, None]).float(),
        "building": s["building_map"][None].float(),
        "free_mask": s["free_mask"][None].float(),
        "tx_rowcol": s["tx_rowcol"][None],
        "sequence": _seq_tensor(cond.sequences, seq_len),
        "target": s["radio_map"][None].float(),
    }
    free = s["free_mask"].numpy()[0].astype(bool)
    building01 = s["building_map"].numpy()[0]
    obs_mask = np.zeros_like(free)
    obs_mask[rows, cols] = True
    obs_mask &= free
    return {"gt_dbm": gt_dbm, "free": free, "building01": building01, "tx_rc": tx_rc,
            "obs": obs, "obs_mask": obs_mask, "batch": batch}


def score_classical(reconstructor, sample) -> dict:
    rec = reconstructor.reconstruct(sample["obs"], sample["free"])
    return all_metrics(rec.mean, sample["gt_dbm"], rec.std, sample["building01"],
                       sample["free"], sample["obs_mask"], sample["tx_rc"])


def score_learned(learned, sample) -> dict:
    mean_dbm, std_dbm = learned.predict_batch(sample["batch"])
    return all_metrics(mean_dbm[0], sample["gt_dbm"], std_dbm[0], sample["building01"],
                       sample["free"], sample["obs_mask"], sample["tx_rc"])


def run_benchmark(classical: Dict, learned: Dict, base: RadioMapSeerDataset,
                  test_maps: List[int], tx_list: List[int], k=3, n_points=100, seed=0):
    sampler = TrajectorySampler(seed=seed)
    noise = MeasurementNoise()
    rng = np.random.default_rng(seed)
    per_method = {name: [] for name in list(classical) + list(learned)}

    for map_id in test_maps:
        for tx in tx_list:
            sample = build_eval_sample(base, sampler, noise, map_id, tx, rng, k=k, n_points=n_points)
            for name, rec in classical.items():
                per_method[name].append(score_classical(rec, sample))
            for name, lr in learned.items():
                per_method[name].append(score_learned(lr, sample))

    # aggregate
    key = "rmse_free_unobs"
    summary = {}
    for name, rows in per_method.items():
        arr = np.array([r[key] for r in rows])
        agg = {"n_samples": len(rows), key: bootstrap_ci(arr)}
        for mk in rows[0]:
            if mk == key:
                continue  # keep the CI dict; don't overwrite with a float
            agg[mk] = round(float(np.nanmean([r[mk] for r in rows])), 4)
        summary[name] = agg

    # Wilcoxon: every method vs the 'full' learned model if present
    ref = "full" if "full" in per_method else list(per_method)[0]
    ref_arr = np.array([r[key] for r in per_method[ref]])
    for name in per_method:
        if name == ref:
            continue
        summary[name][f"wilcoxon_vs_{ref}"] = paired_wilcoxon(
            np.array([r[key] for r in per_method[name]]), ref_arr)
    return summary, per_method

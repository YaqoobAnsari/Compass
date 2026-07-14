"""
Training dataset for the COMPASS learned reconstructor.

Wraps RadioMapSeerDataset + the trajectory sampler + realistic noise to produce,
per (map, tx): the conditioning channels, the ordered sequence tensor, the dense
target, and the masks. Ablation flags live here so every variant trains on an
identical pipeline except the dimension under test:
  * order_shuffle  -> destroy sequence order (innovation #3 control)
  * device_aug     -> add per-trajectory device offset (innovation #4 stress)
  * correlated_noise -> Gudmundson shadowing vs i.i.d.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.conditioning import SEQ_FEATURES, make_conditioning
from ..data.noise import MeasurementNoise
from ..data.radiomapseer import RadioMapSeerDataset
from ..data.trajectory import TrajectorySampler


class CompassTrainDataset(Dataset):
    def __init__(
        self,
        root,
        map_ids,
        variant: str = "IRT2",
        tx_per_map: int = 20,         # subsample TX for speed; 80 available
        k: int = 3,
        n_points: int = 100,
        seq_len: int = 256,
        order_shuffle: bool = False,
        use_noise: bool = True,
        correlated_noise: bool = True,
        tx_jitter_px: float = 0.0,
        seed: int = 0,
    ):
        self.base = RadioMapSeerDataset(root, map_ids=map_ids, variant=variant, tx_per_map=tx_per_map)
        self.sampler = TrajectorySampler(seed=seed)
        self.noise = MeasurementNoise() if use_noise else None
        self.correlated = correlated_noise
        self.k = k
        self.n_points = n_points
        self.seq_len = seq_len
        self.order_shuffle = order_shuffle
        self.tx_jitter_px = tx_jitter_px
        self.seed = seed

    def __len__(self):
        return len(self.base.samples)

    def _seq_tensor(self, sequences: List[np.ndarray], rng) -> torch.Tensor:
        if sequences:
            seq = np.concatenate(sequences, axis=0)
        else:
            seq = np.zeros((1, len(SEQ_FEATURES)), np.float32)
        if self.order_shuffle:
            seq = seq[rng.permutation(len(seq))]
        # pad / truncate to seq_len
        if len(seq) >= self.seq_len:
            seq = seq[: self.seq_len]
        else:
            seq = np.concatenate([seq, np.zeros((self.seq_len - len(seq), seq.shape[1]), np.float32)])
        return torch.from_numpy(seq.astype(np.float32))

    def __getitem__(self, idx: int) -> dict:
        map_id, tx = self.base.samples[idx]
        building_raw = self.base.load_building(map_id)               # {0,255}
        s = self.base[idx]
        gt_dbm = s["radio_map_dbm"].numpy()[0]
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)

        shadow = None
        if self.noise is not None and self.correlated:
            shadow = self.noise.new_field(gt_dbm.shape, rng=rng)
        cond = make_conditioning(
            building_raw, gt_dbm, tuple(int(v) for v in s["tx_rowcol"].numpy()),
            self.sampler, self.noise if self.correlated else None, rng,
            k=self.k, n_points=self.n_points, shadow_field=shadow,
        )
        # i.i.d. fallback noise already handled inside make_conditioning if noise passed

        tx_rc = s["tx_rowcol"]
        if self.tx_jitter_px > 0:  # teach the model to use a NOISY/estimated TX prior
            j = rng.normal(0, self.tx_jitter_px, 2)
            tx_rc = torch.tensor(
                [int(np.clip(tx_rc[0].item() + j[0], 0, 255)),
                 int(np.clip(tx_rc[1].item() + j[1], 0, 255))], dtype=torch.long)

        return {
            "sparse_rss": torch.from_numpy(cond.sparse_rss[None]).float(),
            "mask": torch.from_numpy(cond.mask[None]).float(),
            "coverage": torch.from_numpy(cond.coverage[None]).float(),
            "building": s["building_map"].float(),         # 1=building, 0=street
            "free_mask": s["free_mask"].float(),            # 1=street
            "tx_rowcol": tx_rc,
            "sequence": self._seq_tensor(cond.sequences, rng),
            "target": s["radio_map"].float(),               # normalised [-1,1]
            "map_id": map_id, "tx_id": tx,
        }

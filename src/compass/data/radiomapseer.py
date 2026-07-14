"""
RadioMapSeer dataset loader for COMPASS.

Returns, per (map, transmitter) sample, the ground-truth radio map plus the
scene geometry, with the *correct* conventions baked in
(:mod:`compass.data.conventions`):
  * free / walkable space is ``building_map == 0`` (street);
  * gain PNG -> dBm via ``(v/255)*139 - 186``;
  * TX pixel is ``(row, col) = (H-1-y, x)`` from the antenna JSON ``[x, y]``.

Trajectory-conditioning channels (sparse RSS, masks, coverage, realistic noise)
are added on top of this base dataset by the trajectory module — kept separate
so the ground-truth loader stays simple and verifiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import conventions as C
from .floor_plan import street_mask


@dataclass
class RadioMapSeerPaths:
    """Resolves the RadioMapSeer directory layout from a single root."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def buildings(self) -> Path:
        return self.root / "png" / "buildings_complete"

    @property
    def antenna_json(self) -> Path:
        return self.root / "antenna"

    def gain(self, variant: str) -> Path:
        return self.root / "gain" / variant

    def building_png(self, map_id: int) -> Path:
        return self.buildings / f"{map_id}.png"

    def gain_png(self, variant: str, map_id: int, tx_id: int) -> Path:
        return self.gain(variant) / f"{map_id}_{tx_id}.png"

    def antenna(self, map_id: int) -> Path:
        return self.antenna_json / f"{map_id}.json"


def list_map_ids(root: str | Path) -> List[int]:
    """All map IDs present under ``png/buildings_complete``."""
    paths = RadioMapSeerPaths(root)
    ids = sorted(int(p.stem) for p in paths.buildings.glob("*.png"))
    return ids


def split_map_ids(
    map_ids: Sequence[int],
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Deterministic split BY MAP (no geometry leaks across splits)."""
    ids = np.array(sorted(map_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": sorted(ids[:n_train].tolist()),
        "val": sorted(ids[n_train : n_train + n_val].tolist()),
        "test": sorted(ids[n_train + n_val :].tolist()),
    }


class RadioMapSeerDataset(Dataset):
    """Ground-truth + geometry per (map, TX).

    Each item is a dict of tensors:
      ``building_map``   (1,H,W) float {0,1}, 1 = building, 0 = street/free
      ``free_mask``      (1,H,W) float {0,1}, 1 = street/free space
      ``radio_map``      (1,H,W) float [-1,1], normalised gain (diffusion target)
      ``radio_map_dbm``  (1,H,W) float, physical dBm in [-186, -47]
      ``tx_position``    (2,)   float, normalised (x_norm, y_norm) in [0,1]
      ``tx_rowcol``      (2,)   long, TX pixel (row, col)
      ``map_id`` / ``tx_id`` ints
    """

    def __init__(
        self,
        root: str | Path,
        map_ids: Optional[Sequence[int]] = None,
        variant: str = C.DEFAULT_GAIN_VARIANT,
        tx_per_map: int = 80,
        cache_buildings: bool = True,
    ) -> None:
        if variant not in C.GAIN_VARIANTS:
            raise ValueError(f"variant must be one of {C.GAIN_VARIANTS}, got {variant!r}")
        self.paths = RadioMapSeerPaths(root)
        self.variant = variant
        self.tx_per_map = tx_per_map
        self.cache_buildings = cache_buildings

        self.map_ids = list(map_ids) if map_ids is not None else list_map_ids(root)
        if not self.map_ids:
            raise FileNotFoundError(f"No building maps under {self.paths.buildings}")

        self.samples = [(m, t) for m in self.map_ids for t in range(tx_per_map)]
        self._building_cache: Dict[int, np.ndarray] = {}
        self._antenna_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    # --- raw loaders --------------------------------------------------------
    def load_building(self, map_id: int) -> np.ndarray:
        if self.cache_buildings and map_id in self._building_cache:
            return self._building_cache[map_id]
        arr = np.array(Image.open(self.paths.building_png(map_id)))
        if arr.shape != (C.MAP_SIZE, C.MAP_SIZE):
            raise ValueError(f"map {map_id}: expected {C.MAP_SIZE}^2, got {arr.shape}")
        if self.cache_buildings:
            self._building_cache[map_id] = arr
        return arr

    def load_gain(self, map_id: int, tx_id: int) -> np.ndarray:
        return np.array(Image.open(self.paths.gain_png(self.variant, map_id, tx_id))).astype(np.float32)

    def load_antenna(self, map_id: int) -> np.ndarray:
        if map_id in self._antenna_cache:
            return self._antenna_cache[map_id]
        arr = np.array(json.load(open(self.paths.antenna(map_id))), dtype=np.float32)
        self._antenna_cache[map_id] = arr
        return arr

    # --- item ---------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        map_id, tx_id = self.samples[idx]

        building = self.load_building(map_id)                  # {0,255}
        gain_png = self.load_gain(map_id, tx_id)               # [0,255]
        antenna = self.load_antenna(map_id)                    # (80, 2) [x, y]

        free = street_mask(building)                           # bool, True = street
        building_bin = (~free).astype(np.float32)              # 1 = building

        radio_map = C.png_to_signed_unit(gain_png)             # [-1, 1]
        radio_map_dbm = C.png_to_dbm(gain_png)                 # dBm

        tx_xy = antenna[tx_id]
        tx_row, tx_col = C.tx_pixel_from_json(tx_xy)
        tx_pos = C.tx_position_normalised(tx_xy)

        return {
            "building_map": torch.from_numpy(building_bin[None]).float(),
            "free_mask": torch.from_numpy(free[None].astype(np.float32)).float(),
            "radio_map": torch.from_numpy(radio_map[None]).float(),
            "radio_map_dbm": torch.from_numpy(radio_map_dbm[None]).float(),
            "tx_position": torch.from_numpy(tx_pos).float(),
            "tx_rowcol": torch.tensor([tx_row, tx_col], dtype=torch.long),
            "map_id": int(map_id),
            "tx_id": int(tx_id),
        }

"""
Building-map geometry helpers for COMPASS.

CORRECT polarity (verified — see :mod:`compass.data.conventions`):
free / walkable space is ``building_map == 0`` (street); ``255`` is building.
All masks here key off that, so trajectories are sampled ON STREETS.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .conventions import BUILDING_VALUE, STREET_VALUE


def street_mask(building_map: np.ndarray, erosion_radius: int = 0) -> np.ndarray:
    """Boolean mask of free / walkable (street) pixels.

    Args:
        building_map: uint8 array with values in {0, 255}.
        erosion_radius: optionally erode the street region to keep trajectories
            off building walls (pixels).
    """
    mask = building_map == STREET_VALUE
    if erosion_radius > 0:
        struct = np.ones((2 * erosion_radius + 1, 2 * erosion_radius + 1), dtype=bool)
        mask = binary_erosion(mask, structure=struct)
    return mask


def building_mask(building_map: np.ndarray) -> np.ndarray:
    """Boolean mask of building-interior pixels."""
    return building_map == BUILDING_VALUE


def distance_to_walls(building_map: np.ndarray) -> np.ndarray:
    """For each street pixel, Euclidean distance (px) to the nearest building/wall.

    Zero inside buildings. Used for corridor-biased trajectory sampling.
    """
    return distance_transform_edt(street_mask(building_map)).astype(np.float32)


def corridor_mask(building_map: np.ndarray, min_distance: float = 3.0) -> np.ndarray:
    """Street pixels at least ``min_distance`` px from any wall (street centres)."""
    return distance_to_walls(building_map) >= min_distance


def street_fraction(building_map: np.ndarray) -> float:
    """Fraction of the map that is free / street space."""
    return float(street_mask(building_map).mean())

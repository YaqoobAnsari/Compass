"""
Authoritative conventions for the RadioMapSeer dataset as used by COMPASS.

This module is the SINGLE SOURCE OF TRUTH for how raw RadioMapSeer PNGs map to
physical quantities and pixel coordinates. Every fact below was verified
empirically against the raw files (map 0, multiple TX) on 2026-06-18; the checks
live in ``scripts/verify_dataset.py`` and ``tests/test_dataset.py``.

The predecessor project (TrajectoryDiff / v1) got TWO of these wrong. COMPASS
must not, so they are centralised here and asserted in tests.

Verified facts
--------------
1. Building map ``png/buildings_complete/{map}.png`` — uint8, values in {0, 255}:
     * 0   = STREET / free space   (~75 % of pixels; carries real signal)
     * 255 = BUILDING interior     (~25 %; signal sits at the noise floor)
   => free / walkable space is ``building_map == 0`` (v1 used 255 — inverted).

2. Gain map ``gain/{variant}/{map}_{tx}.png`` — uint8 in [0, 255], a linear
   encoding of negative pathloss:
       gain_dBm = (v / 255) * 139 - 186          # range [-186, -47], higher = stronger
   Street pixels average ~ -137 dBm; building interiors ~ -184 dBm (floor).

3. Antenna positions ``antenna/{map}.json`` — list of ``[x, y]`` per TX, with y
   measured from the BOTTOM of the image. The TX image-array pixel is therefore
       (row, col) = (H - 1 - y, x)
   Verified: the lit pixel of ``png/antennas/{map}_{tx}.png`` AND the gain-map
   argmax both sit at ``(H-1-y, x)``, not ``(y, x)``. v1 used ``(x, y)``
   unflipped, vertically mirroring the TX relative to the gain map.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# --- Geometry ---------------------------------------------------------------
MAP_SIZE = 256          # pixels per side
RESOLUTION_M = 1.0      # metres per pixel (256 m x 256 m scene)

# --- Building-map pixel semantics -------------------------------------------
STREET_VALUE = 0        # free space / walkable / carries signal
BUILDING_VALUE = 255    # obstacle interior / noise floor

# --- Pathloss (gain) encoding -----------------------------------------------
PATHLOSS_MIN_DBM = -186.0
PATHLOSS_MAX_DBM = -47.0
PATHLOSS_RANGE_DB = PATHLOSS_MAX_DBM - PATHLOSS_MIN_DBM   # 139.0

# Available ray-tracing variants under gain/
GAIN_VARIANTS = ("DPM", "IRT2", "IRT4")
DEFAULT_GAIN_VARIANT = "IRT2"


def png_to_dbm(v) -> np.ndarray:
    """uint8 gain-PNG value(s) -> dBm (negative pathloss)."""
    return (np.asarray(v, dtype=np.float32) / 255.0) * PATHLOSS_RANGE_DB + PATHLOSS_MIN_DBM


def dbm_to_png(dbm) -> np.ndarray:
    """dBm -> [0, 255] gain scale (inverse of :func:`png_to_dbm`, unclipped)."""
    return (np.asarray(dbm, dtype=np.float32) - PATHLOSS_MIN_DBM) / PATHLOSS_RANGE_DB * 255.0


def png_to_signed_unit(v) -> np.ndarray:
    """Map a [0, 255] PNG to [-1, 1] (diffusion signal convention)."""
    return np.asarray(v, dtype=np.float32) / 255.0 * 2.0 - 1.0


def signed_unit_to_dbm(x) -> np.ndarray:
    """Inverse of the gain pipeline: [-1, 1] normalised gain -> dBm."""
    v01 = (np.asarray(x, dtype=np.float32) + 1.0) / 2.0
    return v01 * PATHLOSS_RANGE_DB + PATHLOSS_MIN_DBM


def tx_pixel_from_json(xy: Sequence[float], map_size: int = MAP_SIZE) -> Tuple[int, int]:
    """RadioMapSeer antenna ``[x, y_from_bottom]`` -> ``(row, col)`` image index."""
    x, y = int(round(xy[0])), int(round(xy[1]))
    row = map_size - 1 - y
    col = x
    return row, col


def tx_position_normalised(xy: Sequence[float], map_size: int = MAP_SIZE) -> np.ndarray:
    """Antenna ``[x, y_from_bottom]`` -> normalised ``(x_norm, y_norm)`` in [0, 1],
    where (x_norm, y_norm) correspond to (col, row) in image space after the y-flip."""
    row, col = tx_pixel_from_json(xy, map_size)
    return np.array([col / map_size, row / map_size], dtype=np.float32)

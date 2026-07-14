"""
Order-preserving trajectory simulation for COMPASS — on STREETS (correct polarity).

Unlike v1 (which flattened walks into order-blind rasters and, due to the
polarity bug, walked *inside buildings*), this module:
  * samples trajectories on the free/street mask (``building_map == 0``);
  * keeps the ORDERED sequence of timed points (row, col, t), so direction,
    speed, and sequence information survive for the sequence-aware encoder.

Three motion styles (mixed by default): shortest-path (A*), momentum random
walk, and corridor-biased (prefers street centres). Timestamps come from a
configurable walking speed, so variable-speed / sampling-rate experiments are
possible later.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .conventions import RESOLUTION_M
from .floor_plan import distance_to_walls, street_mask


@dataclass
class Trajectory:
    """An ordered, timed walk in image coordinates."""

    rows: np.ndarray   # (N,) float
    cols: np.ndarray   # (N,) float
    t: np.ndarray      # (N,) seconds (monotonic)
    kind: str = "unknown"

    def __len__(self) -> int:
        return len(self.rows)

    def points_rc(self) -> np.ndarray:
        return np.stack([self.rows, self.cols], axis=1)


# --- A* on the street graph -------------------------------------------------
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _astar(free: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    H, W = free.shape

    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = [(h(start, goal), 0, start)]
    came: dict = {}
    g = {start: 0.0}
    counter = 0
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        cr, cc = cur
        for dr, dc in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc]:
                ng = g[cur] + np.hypot(dr, dc)
                nb = (nr, nc)
                if nb not in g or ng < g[nb]:
                    g[nb] = ng
                    came[nb] = cur
                    counter += 1
                    heapq.heappush(open_set, (ng + h(nb, goal), counter, nb))
    return None


def _resample_polyline(path_rc: np.ndarray, n_points: int, speed_px_s: float) -> Trajectory:
    """Resample an ordered pixel path at equal arc-length, assign timestamps from speed."""
    diffs = np.diff(path_rc, axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        rows = np.full(n_points, path_rc[0, 0], float)
        cols = np.full(n_points, path_rc[0, 1], float)
        return Trajectory(rows, cols, np.arange(n_points) / speed_px_s)
    targets = np.linspace(0.0, total, n_points)
    rows = np.interp(targets, cum, path_rc[:, 0])
    cols = np.interp(targets, cum, path_rc[:, 1])
    t = targets / speed_px_s
    return Trajectory(rows, cols, t)


class TrajectorySampler:
    """Generates on-street, ordered, timed trajectories from a building map."""

    def __init__(
        self,
        speed_mps: float = 1.4,
        resolution_m: float = RESOLUTION_M,
        erosion_radius: int = 1,
        min_path_len_px: float = 30.0,
        step_px: float = 2.0,
        momentum: float = 0.8,
        corridor_pref: float = 0.7,
        seed: Optional[int] = None,
    ):
        self.speed_px_s = speed_mps / resolution_m   # 1 m/px -> px/s == m/s
        self.erosion_radius = erosion_radius
        self.min_path_len_px = min_path_len_px
        self.step_px = step_px
        self.momentum = momentum
        self.corridor_pref = corridor_pref
        self.rng = np.random.default_rng(seed)

    def _free(self, building_map: np.ndarray) -> np.ndarray:
        return street_mask(building_map, erosion_radius=self.erosion_radius)

    # --- styles -------------------------------------------------------------
    def shortest_path(self, building_map: np.ndarray, n_points: int = 100, max_attempts: int = 40) -> Trajectory:
        free = self._free(building_map)
        coords = np.argwhere(free)
        for _ in range(max_attempts):
            a, b = coords[self.rng.choice(len(coords), 2, replace=False)]
            if np.hypot(*(a - b)) < self.min_path_len_px:
                continue
            path = _astar(free, tuple(a), tuple(b))
            if path is not None and len(path) >= self.min_path_len_px:
                tr = _resample_polyline(np.array(path, float), n_points, self.speed_px_s)
                tr.kind = "shortest_path"
                return tr
        return self.random_walk(building_map, n_points)  # fallback (rare on a connected street net)

    def random_walk(self, building_map: np.ndarray, n_points: int = 100) -> Trajectory:
        free = self._free(building_map)
        coords = np.argwhere(free)
        r, c = coords[self.rng.choice(len(coords))].astype(float)
        ang = self.rng.uniform(0, 2 * np.pi)
        rows, cols = [], []
        for _ in range(n_points):
            rows.append(r)
            cols.append(c)
            for _ in range(20):
                ang2 = ang + self.rng.normal(0, (1 - self.momentum) * np.pi / 2)
                nr, nc = r + self.step_px * np.sin(ang2), c + self.step_px * np.cos(ang2)
                if 0 <= int(round(nr)) < free.shape[0] and 0 <= int(round(nc)) < free.shape[1] and free[int(round(nr)), int(round(nc))]:
                    r, c, ang = nr, nc, ang2
                    break
            else:
                near = coords[np.linalg.norm(coords - [r, c], axis=1) < 20]
                if len(near):
                    r, c = near[self.rng.choice(len(near))].astype(float)
                    ang = self.rng.uniform(0, 2 * np.pi)
        dt = self.step_px / self.speed_px_s
        return Trajectory(np.array(rows), np.array(cols), np.arange(n_points) * dt, "random_walk")

    def corridor_biased(self, building_map: np.ndarray, n_points: int = 100) -> Trajectory:
        free = self._free(building_map)
        dist = distance_to_walls(building_map)
        corridor = np.argwhere(dist >= 3.0)
        if len(corridor) == 0:
            return self.random_walk(building_map, n_points)
        r, c = corridor[self.rng.choice(len(corridor))].astype(float)
        ang = self.rng.uniform(0, 2 * np.pi)
        rows, cols = [], []
        for _ in range(n_points):
            rows.append(r)
            cols.append(c)
            best, best_score = None, -np.inf
            for _ in range(10):
                ang2 = ang + self.rng.normal(0, np.pi / 4)
                nr, nc = r + self.step_px * np.sin(ang2), c + self.step_px * np.cos(ang2)
                ir, ic = int(round(nr)), int(round(nc))
                if 0 <= ir < free.shape[0] and 0 <= ic < free.shape[1] and free[ir, ic]:
                    score = dist[ir, ic] if self.rng.random() < self.corridor_pref else self.rng.random()
                    if score > best_score:
                        best, best_score = (nr, nc, ang2), score
            if best is not None:
                r, c, ang = best
        dt = self.step_px / self.speed_px_s
        return Trajectory(np.array(rows), np.array(cols), np.arange(n_points) * dt, "corridor_biased")

    # --- mixed --------------------------------------------------------------
    def sample(self, building_map: np.ndarray, n_points: int = 100, kind: Optional[str] = None) -> Trajectory:
        kind = kind or self.rng.choice(
            ["shortest_path", "random_walk", "corridor_biased"], p=[0.4, 0.3, 0.3]
        )
        return getattr(self, kind)(building_map, n_points)

    def sample_many(
        self, building_map: np.ndarray, k: int = 3, n_points: int = 100, kinds: Optional[Sequence[str]] = None
    ) -> List[Trajectory]:
        if kinds is not None:
            return [self.sample(building_map, n_points, kind=kinds[i % len(kinds)]) for i in range(k)]
        return [self.sample(building_map, n_points) for _ in range(k)]

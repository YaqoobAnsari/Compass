"""
Closed-loop active crowdsensing on a dense-GT field (RadioMapSeer now).

Protocol per round:
    reconstruct from observations so far
    -> acquisition strategy picks a target pixel
    -> route a walkable trajectory from the current position to the target
    -> "collect" GT (+ optional noise) along that walk
    -> append to observations
and record free-unobserved RMSE vs cumulative measurement budget.

Because RadioMapSeer has a dense ground truth, we can measure the blind-spot
RMSE-vs-budget curve exactly and compare strategies at MATCHED budget — the
headline evidence for uncertainty-guided collection. The same loop will run on
real data once a reconstructor/evaluator without dense GT is plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..recon.base import Observations, Reconstructor
from .acquisition import STRATEGIES
from .collect import collect_along_trajectories, observed_mask
from .graph import nearest_reachable, random_free_pixel, walk_to


@dataclass
class ActiveResult:
    strategy: str
    budgets: List[int] = field(default_factory=list)        # cumulative #observations
    rmse_free_unobs: List[float] = field(default_factory=list)
    rmse_free: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "budgets": self.budgets,
            "rmse_free_unobs": self.rmse_free_unobs,
            "rmse_free": self.rmse_free,
        }


def run_active_loop(
    reconstructor: Reconstructor,
    gt_dbm: np.ndarray,
    free: np.ndarray,
    strategy: str,
    n_rounds: int = 12,
    points_per_walk: int = 60,
    noise_std: float = 2.0,
    seed: int = 0,
    seed_obs: Optional[Observations] = None,
) -> ActiveResult:
    """Run one strategy on one map; return its RMSE-vs-budget trace."""
    rng = np.random.default_rng(seed)
    acq = STRATEGIES[strategy]
    res = ActiveResult(strategy=strategy)

    obs = seed_obs if seed_obs is not None else Observations.empty()
    # start position: a random free pixel (or last seed obs)
    if len(obs):
        pos = (int(obs.rows[-1]), int(obs.cols[-1]))
    else:
        pos = random_free_pixel(free, rng)

    for _ in range(n_rounds):
        rec = reconstructor.reconstruct(obs, free)
        obs_m = observed_mask(obs, gt_dbm.shape) & free
        m = rec.masked_rmse(gt_dbm, obs_m)
        res.budgets.append(len(obs))
        res.rmse_free_unobs.append(m["rmse_free_unobs"])
        res.rmse_free.append(m["rmse_free"])

        target = acq(rec.std, obs_m, free, rng)
        target = nearest_reachable(free, target, pos)
        tr = walk_to(free, pos, target, n_points=points_per_walk, rng=rng)
        if tr is None:  # unreachable; jump to a random free pixel
            pos = random_free_pixel(free, rng)
            continue
        new = collect_along_trajectories([tr], gt_dbm, rng=rng, noise_std=noise_std)
        obs = obs.add(new.rows, new.cols, new.values)
        pos = (int(tr.rows[-1]), int(tr.cols[-1]))

    # final point after last collection
    rec = reconstructor.reconstruct(obs, free)
    obs_m = observed_mask(obs, gt_dbm.shape) & free
    m = rec.masked_rmse(gt_dbm, obs_m)
    res.budgets.append(len(obs))
    res.rmse_free_unobs.append(m["rmse_free_unobs"])
    res.rmse_free.append(m["rmse_free"])
    return res

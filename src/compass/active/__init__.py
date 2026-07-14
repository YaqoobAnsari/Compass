"""COMPASS active crowdsensing: uncertainty -> walkable route -> collect -> repeat."""

from .acquisition import STRATEGIES
from .collect import collect_along_trajectories, observed_mask
from .graph import nearest_reachable, random_free_pixel, walk_to
from .loop import ActiveResult, run_active_loop

__all__ = [
    "STRATEGIES",
    "collect_along_trajectories",
    "observed_mask",
    "walk_to",
    "nearest_reachable",
    "random_free_pixel",
    "ActiveResult",
    "run_active_loop",
]

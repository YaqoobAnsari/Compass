"""COMPASS evaluation: metrics, significance testing, benchmark orchestration."""

from .metrics import (
    accuracy_metrics,
    all_metrics,
    calibration_metrics,
    plausibility_metrics,
    ray_monotonicity_violation,
)
from .stats import bootstrap_ci, paired_wilcoxon
from .benchmark import build_eval_sample, run_benchmark, score_classical, score_learned

__all__ = [
    "all_metrics", "accuracy_metrics", "calibration_metrics", "plausibility_metrics",
    "ray_monotonicity_violation",
    "bootstrap_ci", "paired_wilcoxon",
    "run_benchmark", "build_eval_sample", "score_classical", "score_learned",
]

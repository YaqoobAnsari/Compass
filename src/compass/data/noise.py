"""
Realistic measurement-noise model for COMPASS.

v1 added i.i.d. Gaussian noise (sigma=2 dB) — which averages out trivially and
is why the model looked "noise-robust" AND why trajectory order seemed useless.
Real RSS error is dominated by *spatially-correlated* shadow fading plus
per-device offsets. This module provides:

  * ``correlated_shadowing_field`` — a zero-mean Gaussian field with
    approximately exponential spatial autocorrelation (Gudmundson 1991;
    3GPP TR 38.901): consecutive samples along a walk share nearly the same
    shadowing, which a sequence model can exploit.
  * ``MeasurementNoise`` — applies, per trajectory, a shared correlated
    shadowing field + small fast-fade residual + a per-device affine
    (offset, gain) + 1 dB quantization + noise-floor clipping.

Default parameters are anchored to 3GPP UMa/UMi street-canyon ranges
(sigma_shadow 4-7.5 dB, decorrelation 10-50 m). See spikes/03.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

from .conventions import MAP_SIZE, PATHLOSS_MIN_DBM, RESOLUTION_M


def correlated_shadowing_field(
    shape: tuple[int, int] = (MAP_SIZE, MAP_SIZE),
    sigma_db: float = 6.0,
    decorr_m: float = 30.0,
    resolution_m: float = RESOLUTION_M,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Zero-mean Gaussian shadowing field, std ``sigma_db``, exponential spatial
    autocorrelation with length ~``decorr_m`` (Gudmundson). White noise convolved
    with an exponential kernel, rescaled to the target std."""
    rng = rng or np.random.default_rng()
    H, W = shape
    L = max(decorr_m / resolution_m, 1e-3)  # correlation length in pixels
    white = rng.standard_normal((H, W))
    rad = int(max(3 * L, 3))
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    kernel = np.exp(-np.hypot(xx, yy) / L)
    kernel /= np.sqrt((kernel ** 2).sum())  # preserve variance under convolution
    field = fftconvolve(white, kernel, mode="same")
    std = field.std()
    if std > 0:
        field = field / std * sigma_db
    return field.astype(np.float32)


def measured_decorrelation_px(field: np.ndarray, max_lag: int = 60) -> float:
    """Empirical decorrelation distance (px) where the radial autocorrelation drops to 1/e."""
    f = field - field.mean()
    var = (f ** 2).mean()
    if var <= 0:
        return 0.0
    lags = np.arange(1, max_lag)
    # average of horizontal and vertical lagged correlation
    corr = []
    for d in lags:
        ch = (f[:, :-d] * f[:, d:]).mean() / var
        cv = (f[:-d, :] * f[d:, :]).mean() / var
        corr.append(0.5 * (ch + cv))
    corr = np.asarray(corr)
    below = np.where(corr <= np.exp(-1))[0]
    return float(lags[below[0]]) if len(below) else float(max_lag)


@dataclass
class MeasurementNoise:
    """Layered, citation-anchored RSS measurement-noise model (all in dB)."""

    sigma_shadow_db: float = 6.0     # 3GPP UMa NLOS ~6; UMi NLOS ~7.5
    decorr_m: float = 30.0           # exponential correlation length
    fast_fade_db: float = 1.0        # small-scale residual (i.i.d.)
    device_offset_std_db: float = 4.0  # per-trajectory additive bias
    device_gain_std: float = 0.05    # per-trajectory multiplicative gain (around 1.0)
    quantization_db: float = 1.0     # RSSI reported at ~1 dB granularity
    floor_dbm: float = PATHLOSS_MIN_DBM

    def new_field(self, shape=(MAP_SIZE, MAP_SIZE), rng=None) -> np.ndarray:
        """Draw one shadowing field for a map (shared across that map's trajectories)."""
        return correlated_shadowing_field(shape, self.sigma_shadow_db, self.decorr_m, rng=rng)

    def apply(
        self,
        clean_dbm: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        shadow_field: np.ndarray,
        rng: np.random.Generator,
        ref_level_dbm: float = -130.0,
    ) -> np.ndarray:
        """Corrupt clean per-point RSS (dBm) for ONE trajectory/device.

        Args:
            clean_dbm: (N,) ground-truth gain (dBm) sampled along the path.
            rows, cols: (N,) integer-ish pixel coords (for the shadowing field).
            shadow_field: (H,W) correlated field from :meth:`new_field`.
            rng: per-call generator.
            ref_level_dbm: pivot for the device gain term.
        """
        ri = np.clip(np.round(rows).astype(int), 0, shadow_field.shape[0] - 1)
        ci = np.clip(np.round(cols).astype(int), 0, shadow_field.shape[1] - 1)
        shadow = shadow_field[ri, ci]                                   # correlated along path
        fast = rng.normal(0.0, self.fast_fade_db, size=clean_dbm.shape)  # i.i.d. residual
        offset = rng.normal(0.0, self.device_offset_std_db)             # per-device
        gain = 1.0 + rng.normal(0.0, self.device_gain_std)              # per-device
        meas = gain * (clean_dbm - ref_level_dbm) + ref_level_dbm + shadow + fast + offset
        if self.quantization_db > 0:
            meas = np.round(meas / self.quantization_db) * self.quantization_db
        return np.maximum(meas, self.floor_dbm).astype(np.float32)

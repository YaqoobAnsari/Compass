"""Plotting helpers for COMPASS — visual affirmation that the pipeline is correct.

Uses a headless (Agg) backend so figures render on the CPU login node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.conventions import PATHLOSS_MAX_DBM, PATHLOSS_MIN_DBM  # noqa: E402


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def plot_dataset_sample(sample: dict, out_path: str | Path, title: Optional[str] = None) -> Path:
    """4-panel sanity figure for one (map, tx): building, gain(dBm)+TX, free mask, dBm histogram."""
    building = _np(sample["building_map"])[0]
    free = _np(sample["free_mask"])[0].astype(bool)
    dbm = _np(sample["radio_map_dbm"])[0]
    row, col = (int(v) for v in _np(sample["tx_rowcol"]))

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.7))
    ax[0].imshow(building, cmap="gray_r", vmin=0, vmax=1)
    ax[0].plot(col, row, "r*", ms=15, mec="white")
    ax[0].set_title("building (1=bldg / 0=street)\nred ★ = TX")

    im = ax[1].imshow(dbm, cmap="viridis", vmin=PATHLOSS_MIN_DBM, vmax=PATHLOSS_MAX_DBM)
    ax[1].plot(col, row, "r*", ms=15, mec="white")
    ax[1].set_title("gain (dBm) + TX")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(free, cmap="Greens", vmin=0, vmax=1)
    ax[2].set_title(f"free / street mask ({free.mean() * 100:.0f}%)")

    ax[3].hist(dbm[free].ravel(), bins=60, alpha=0.6, color="seagreen", density=True, label="street")
    ax[3].hist(dbm[~free].ravel(), bins=60, alpha=0.6, color="dimgray", density=True, label="building")
    ax[3].axvline(dbm[free].mean(), color="seagreen", ls="--")
    ax[3].axvline(dbm[~free].mean(), color="dimgray", ls="--")
    ax[3].set_title("dBm distribution by region")
    ax[3].set_xlabel("dBm")
    ax[3].legend(fontsize=9)

    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectory_sample(sample: dict, cond, out_path: str | Path, title: Optional[str] = None) -> Path:
    """Trajectories on streets (coloured by order) + sparse RSS + coverage + per-point RSS."""
    free = _np(sample["free_mask"])[0].astype(bool)
    row, col = (int(v) for v in _np(sample["tx_rowcol"]))

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.7))
    ax[0].imshow(free, cmap="Greys", vmin=0, vmax=3)  # faint streets
    for tr in cond.trajectories:
        ax[0].scatter(tr.cols, tr.rows, c=np.arange(len(tr)), cmap="plasma", s=4)
    ax[0].plot(col, row, "r*", ms=15, mec="white")
    ax[0].set_title(f"trajectories on streets (order=colour)\n{len(cond.trajectories)} walks, cov={cond.coverage_fraction*100:.2f}%")

    sp = _np(cond.sparse_rss)
    m = _np(cond.mask).astype(bool)
    disp = np.where(m, sp, np.nan)
    im = ax[1].imshow(disp, cmap="viridis", vmin=-1, vmax=1)
    ax[1].set_title("measured sparse RSS (normalised)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(_np(cond.coverage), cmap="magma")
    ax[2].set_title("coverage density")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    for i, meas in enumerate(cond.measured_dbm):
        ax[3].plot(meas, lw=1.2, label=f"walk {i} ({cond.trajectories[i].kind})")
    ax[3].set_title("measured RSS along each walk")
    ax[3].set_xlabel("point index (time order)")
    ax[3].set_ylabel("dBm")
    ax[3].legend(fontsize=7)

    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_shadowing_field(field, out_path: str | Path, decorr_px: Optional[float] = None,
                         title: Optional[str] = None) -> Path:
    """Correlated shadowing field + its radial autocorrelation (vs i.i.d. reference)."""
    field = _np(field)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im = ax[0].imshow(field, cmap="RdBu_r", vmin=-3 * field.std(), vmax=3 * field.std())
    ax[0].set_title(f"correlated shadowing (std={field.std():.1f} dB)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    f = field - field.mean()
    var = (f ** 2).mean()
    lags = np.arange(0, 60)
    corr = [1.0] + [float((f[:, :-d] * f[:, d:]).mean() / var) for d in lags[1:]]
    ax[1].plot(lags, corr, "b-", label="shadowing field")
    ax[1].axhline(np.exp(-1), color="gray", ls=":", label="1/e")
    if decorr_px is not None:
        ax[1].axvline(decorr_px, color="r", ls="--", label=f"decorr≈{decorr_px:.0f} px")
    ax[1].set_xlabel("lag (px)")
    ax[1].set_ylabel("autocorrelation")
    ax[1].set_title("spatial autocorrelation")
    ax[1].legend(fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out_path)

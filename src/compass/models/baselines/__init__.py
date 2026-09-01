"""
External deep-learning radio-map baselines, re-implemented faithfully with the SAME
batch interface as COMPASS so the comparison is fair (identical data, trajectory
sampling, training budget, and evaluation).

Every baseline is an ``nn.Module`` whose ``forward(batch: dict) -> (B,1,H,W)``
returns a normalised radio map, exactly like ``CompassNet``. Each declares which
input channels it consumes via its designed architecture; TX can be dropped for the
TX-agnostic (real-data-relevant) track.

Baselines:
  * SparseUNet    — generic deep sparse-to-dense interpolation (samples only; no geometry).
  * RadioUNet     — Levie et al., IEEE TWC 2021 — WNet (two cascaded UNets), 3-channel
                    input (building + TX + sparse measurements). arXiv:1911.09002.
  * PMNet         — Lee et al., 2023 — ResNet encoder w/ dilated convs + ASPP bottleneck +
                    ConvTranspose decoder; won the 1st Pathloss Map Prediction Challenge.
                    arXiv:2211.10527.
  * RMDM          — Jia et al., 2025 — dual-UNet, coarse + diffusion refinement.
                    arXiv:2501.19160 (added separately).
  * RadioMamba    — hybrid Mamba-UNet; bidirectional selective-scan (S6/SSM) global
                    context at the bottleneck (state-space family, 2024-2025).
  * URAM          — uncertainty-aware Bayesian U-Net with MC-dropout + aleatoric head
                    (uncertainty-first reconstruction family, 2024-2025).
  * RadioDiff     — Wang et al., IEEE TCCN 2024 — decoupled, sampling-free diffusion
                    with adaptive FFT filtering. arXiv:2408.08593.
"""

from .pmnet import PMNet
from .radiodiff import RadioDiff
from .radiogan import RadioGAN
from .radiomamba import RadioMamba
from .radiotransformer import RadioTransformer
from .radiounet import RadioUNet
from .rmdm import RMDM
from .sparse_unet import SparseUNet
from .uram import URAM

# Feed-forward baselines trainable via the shared masked-recon trainer.
BASELINES = {
    "sparse_unet": SparseUNet,
    "radiounet": RadioUNet,
    "pmnet": PMNet,
    "radiotransformer": RadioTransformer,
    "radiomamba": RadioMamba,   # state-space (Mamba) family
    "uram": URAM,               # uncertainty-aware Bayesian UNet
    "radiodiff": RadioDiff,     # decoupled sampling-free diffusion (trained via train_rmdm)
    "rmdm": RMDM,        # diffusion — trained via train_rmdm, rebuilt from here for eval
    "radiogan": RadioGAN,  # cGAN — trained via train_gan, rebuilt from here for eval
}

# Baselines needing a dedicated (non-masked-recon) trainer.
DIFFUSION_BASELINES = {"rmdm", "radiodiff"}
GAN_BASELINES = {"radiogan"}

__all__ = ["SparseUNet", "RadioUNet", "PMNet", "RadioTransformer", "RadioMamba", "URAM",
           "RadioDiff", "RMDM", "RadioGAN", "BASELINES", "DIFFUSION_BASELINES", "GAN_BASELINES"]

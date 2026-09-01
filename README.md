<div align="center">

# COMPASS

**Crowd-guided Online radio Mapping with Propagation-Aware Sequential Sensing**

Geometry-aware reconstruction of radio maps from sparse, trajectory-structured
crowdsensed measurements, with calibrated uncertainty and active collection.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-64-brightgreen.svg)](tests/)
[![Data](https://img.shields.io/badge/data-RadioMapSeer%20%2B%20UniCellular-blue.svg)](#data)

<img src="figures/dl_baselines/dl_vs_compass.png" width="82%" alt="COMPASS compared against state-of-the-art deep-learning and classical baselines"/>

</div>

Validated on synthetic (RadioMapSeer) and real indoor cellular (UniCellular) data.
Every result is reported with bootstrap 95% confidence intervals and paired Wilcoxon
tests, alongside negative controls.

---

## Overview

Radio-map estimation has been driven by synthetic benchmarks on which dense
convolutional models such as RadioUNet and PMNet perform well, provided the
transmitter location is known. That setting diverges from real crowdsensing in four
respects. The transmitter is frequently unlocatable because boosters share cell
identifiers, measurements are sparse and trajectory-structured, devices are
heterogeneous by as much as 27.8 dB, and uncertainty is decision-critical rather than
cosmetic. COMPASS targets that regime through geometry-as-attenuation reconstruction,
order-aware and device-aware conditioning, calibrated uncertainty, and
uncertainty-guided active collection.

- COMPASS-WNet reaches 9.97 dB free-unobserved RMSE on the synthetic benchmark, which
  ties RadioUNet on accuracy, attains the best SSIM at 0.814, and is the only method in
  the comparison that also supplies calibrated uncertainty.
- Each innovation is validated independently on real indoor cellular data, covering
  geometry, temporal order, device heterogeneity, source de-risking, and active sensing.
- The evaluation includes six deep-learning baseline classes and seven classical
  families, with negative controls and an explicit statement of what is not claimed.

## Results

### Synthetic benchmark

RadioMapSeer, 120 held-out samples, 95% confidence intervals.

| Method | Free-unobs RMSE (dB) | SSIM | Uncertainty |
|:---|:---:|:---:|:---|
| **COMPASS-WNet (ours)** | **9.97** `[9.54, 10.39]` | **0.814** | Calibrated (0.537) |
| RadioUNet (Levie et al., TWC 2021) | 10.00 `[9.59, 10.42]` | 0.807 | None |
| RadioTransformer | 10.82 | 0.776 | Calibrated (0.451) |
| COMPASS (single U-Net) | 11.04 | 0.787 | Calibrated (0.534) |
| RadioGAN (conditional GAN) | 11.53 | 0.773 | None |
| PMNet (Lee et al., 2023) | 13.98 | 0.694 | None |
| SparseUNet (no geometry) | 21.26 | 0.641 | n/a |
| Classical (RBF, Kriging, GP, IDW) | 26.4 to 28.8 | 0.64 to 0.69 | n/a |
| RMDM (Jia et al., 2025, diffusion) | 39.62 | 0.674 | n/a |

Geometry-aware learned reconstruction outperforms both the best classical family and the
no-geometry deep-learning baseline by roughly a factor of 2.5. COMPASS-WNet is the only
model that matches state-of-the-art accuracy while also providing calibrated
uncertainty, recalibrated to near-exact coverage through split-conformal prediction,
which moves PICP at one standard deviation from 0.14 to 0.66 while preserving ranking.

### Real indoor cellular data

UniCellular. Each row corresponds to an independently validated design decision.

| Innovation | Finding | Figures |
|:---|:---|:---|
| Geometry | Walls attenuate rather than induce detours. Wall-crossing pairs show 2 to 3 dB higher absolute RSS difference at equal distance, and wall-aware IDW improves by 3.9% on the walled site against 0% on the open control. | [`walls_realdata`](figures/walls_realdata/) |
| Temporal order | A GRU order model reaches 1.79 dB against 9.79 dB for an order-blind model, with autocorrelation 0.984 and `p = 4.7e-23`, and it transfers across sites. | [`order_realdata`](figures/order_realdata/) |
| Device heterogeneity | Per-phone offsets have a median of 4.1 dB and a maximum of 27.8 dB, and are reliably estimable with correlation 0.78 to 0.89. | [`device_realdata`](figures/device_realdata/) |
| Source de-risking | Only 18% of cells yield a reliable point source, which supports the transmitter-agnostic design. | [`source_realdata`](figures/source_realdata/) |
| Active sensing | Geometry-aware acquisition outperforms space-filling acquisition on walled sites, with `p = 0.039`. | [`active_wall`](figures/active_wall/) |

### Scope and limitations

On sparse real data, classical interpolation remains stronger than the learned
reconstructor. RBF achieves 5.51 dB against 6.30 dB for the learned model, and this
holds even with simulation-to-real fine-tuning. COMPASS therefore uses classical
interpolation for sparse reconstruction and locates the contribution of learning in
temporal order, uncertainty, and geometry.

## Installation

```bash
git clone https://github.com/YaqoobAnsari/Compass.git && cd Compass
conda create -n compass python=3.10 -y && conda activate compass
pip install -r requirements.txt
pip install -e .          # editable install of the `compass` package
pytest tests/ -q          # 64 unit tests
```

## Data

| Dataset | Role | Notes |
|:---|:---|:---|
| RadioMapSeer | Synthetic benchmark | 256x256 ray-traced pathloss. Place or symlink at `data/raw/`. |
| UniCellular | Real indoor cellular RSS | CMUQ, EC Parking and Ezdan sites. Place under `external/` (Git-LFS). |

Both datasets are external and are excluded from version control.

## Quick start

```bash
# Train the flagship model (SLURM or GPU)
python scripts/train_reconstructor.py --name full --arch compass_wnet --epochs 150

# Benchmark every method, deep-learning and classical, with CIs and Wilcoxon tests
python scripts/eval_dl_baselines.py                 # -> results/dl_baselines.json

# Real-data validation
python scripts/exp_walls_realdata.py                # geometry
python scripts/exp_order_realdata.py                # temporal order
python scripts/exp_uq_recalibrate.py                # calibrated uncertainty
```

## Pretrained weights

All 28 trained checkpoints, covering COMPASS, COMPASS-WNet and every deep-learning
baseline, total 943 MB and are published on the
[Releases](https://github.com/YaqoobAnsari/Compass/releases) page rather than tracked in
git. Download and unpack them under `experiments/`.

## Reproducing the results

Every experiment writes a JSON file to [`results/`](results/) and a plot to
[`figures/`](figures/), and the corresponding scripts live in [`scripts/`](scripts/) as
`exp_*.py`, `train_*.py` and `eval_*.py`. Each number in the tables above is reproducible
from its script. The synthetic benchmark, for example, is produced by
`scripts/eval_dl_baselines.py` and written to `results/dl_baselines.json`.

## Repository structure

```
src/compass/
├── data/        RadioMapSeer, trajectory, noise and conditioning pipeline
├── models/      COMPASS, COMPASS-WNet and deep-learning baselines (baselines/)
├── recon/       classical reconstructors (IDW, Kriging, RBF, GP and others)
├── training/    training loops for reconstruction, diffusion and GAN
├── eval/        metrics (RMSE, MAE, SSIM, PSNR, NMSE, calibration, plausibility)
├── active/      uncertainty-guided active sensing
└── realdata/    UniCellular loaders, walls, order, device, source, learned interpolator
scripts/         experiments (exp_*.py), training and evaluation entry points
results/         per-experiment JSON output
figures/         per-experiment plots
tests/           64 unit tests
METHODS.md       architecture and training specification
```

[`METHODS.md`](METHODS.md) documents every shape, constant and layer exactly as
implemented, together with a section reconciling the code against earlier descriptions of
the system. It is the appropriate starting point for the method in full detail.

## Citation

```bibtex
@misc{ansari2026compass,
  title  = {COMPASS: Crowd-guided Online Radio Mapping with Propagation-Aware
            Sequential Sensing},
  author = {Ansari, Yaqoob},
  year   = {2026},
  note   = {https://github.com/YaqoobAnsari/Compass}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Built on [RadioMapSeer](https://radiomapseer.github.io/) and the UniCellular indoor
fingerprint dataset. Baselines were re-implemented from RadioUNet (Levie et al., TWC
2021), PMNet (Lee et al., 2023) and RMDM (Jia et al., 2025).

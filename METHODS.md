# COMPASS: Methods

**Conditioned Occlusion-aware Multi-Path Aware Sparse Sensing** for indoor radio-map
reconstruction from crowdsensed measurements.

This document describes the architecture and training procedure exactly as implemented,
at the level of detail expected in a conference submission. Every shape, constant, and
layer below was read from the source rather than from prior summaries, and the
provenance of each is given in the final section. Where the implementation differs from
how the system has previously been described, this is stated explicitly in §12.

---

## 1. Problem formulation

Let a scene be a regular grid of size `H × W = 256 × 256` at a resolution of 1 m per
pixel. Three quantities define an instance:

| Symbol | Meaning | Domain |
|---|---|---|
| `y ∈ R^{H×W}` | the true radio map (received signal strength at every pixel) | dBm |
| `B ∈ {0,1}^{H×W}` | building occupancy, 1 = building interior, 0 = free space | binary |
| `Ω ⊂ [H]×[W]` | the set of measured pixels, produced by walking trajectories | sparse |

We observe `{(p, ỹ_p) : p ∈ Ω}` where `ỹ_p` is a noisy measurement, together with `B`
and, when it is known, the transmitter pixel `τ = (τ_r, τ_c)`. The task is to predict
`y` at every free-space pixel that was **not** measured:

```
ŷ = f_θ(sparse measurements, B, τ, walk order, device statistics)
```

and the reported metric is RMSE over `F \ Ω`, where `F = {p : B_p = 0}` is free space.
Error over observed pixels or through building interiors is excluded, because including
either rewards a model for copying its own input or for predicting a constant noise
floor.

### 1.1 Signal normalisation

All signal quantities are mapped from dBm to a signed unit interval before entering the
network, using the dataset's linear pathloss encoding:

```
u = clip( 2 · (d − d_min) / R − 1 ,  −1, 1 )       d_min = −186 dBm,  R = 139 dB
```

so `u ∈ [−1, 1]` and **one normalised unit corresponds to 69.5 dB**. Every RMSE reported
in dB is obtained by multiplying a normalised residual by 69.5. This single constant is
the bridge between the loss (computed in normalised units) and every reported metric.

---

## 2. Input tensor construction

The network consumes a single stacked tensor `X ∈ R^{B×C_in×H×W}`. Channels are appended
in a fixed order, and each group is individually switchable, which is what allows one
class to realise the full model and every ablation without changing any other code path.

| # | Channel | Shape | Construction |
|---|---|---|---|
| 1 | `sparse_rss` | `1×H×W` | normalised measurement at measured pixels, 0 elsewhere |
| 2 | `mask` | `1×H×W` | 1 at measured pixels, 0 elsewhere |
| 3 | `coverage` | `1×H×W` | Gaussian-blurred mask, `σ = 5 px`, rescaled to max 1 |
| 4 | `building` | `1×H×W` | 1 = building, 0 = free space |
| 5–8 | `tx_heatmap` | `4×H×W` | multi-scale Gaussian bumps at the transmitter (§3.1) |
| 9 | `occlusion` | `1×H×W` | fraction of the transmitter ray inside buildings (§3.2) |

With every group enabled, `C_in = 3 + 1 + 4 + 1 = 9`.

The `coverage` channel deserves a note: it is a smoothed density of measurement, not a
binary indicator, and it exists so the decoder can distinguish *"unmeasured but near
measurements"* from *"unmeasured and far from anything"*. These two situations warrant
very different confidence and the binary mask alone cannot express the difference.

---

## 3. Derived geometric channels

Two input channels are **computed analytically rather than learned**. Both are
deterministic functions of `B` and `τ`, so in principle a sufficiently deep network could
infer them; in practice it does not, and supplying them directly is the single largest
accuracy effect we measure.

### 3.1 Multi-scale transmitter heat map

For each scale `i ∈ {0,1,2,3}` with `σ_i = 4 · 2^i` (i.e. 4, 8, 16, 32 px):

```
T_i(p) = exp( −‖p − τ‖² / (2 σ_i²) )
```

stacked into 4 channels. A single Gaussian forces a choice between localising the source
precisely and communicating direction at long range; the multi-scale stack supplies both,
with the narrow scales acting as a point locator and the broad scales as a smooth
distance-and-bearing field that survives to the coarsest decoder resolution.

### 3.2 Ray-occlusion field

This is the channel that breaks the non-line-of-sight plateau. For every pixel `p` we
march the straight segment from the transmitter `τ` to `p` and integrate building
occupancy along it:

```
O(p) = (1 / N) · Σ_{i=1}^{N−1}  B( τ + (i/N)(p − τ) )        N = 32
```

`O(p) ∈ [0,1]` is the fraction of the transmitter-to-pixel path that lies inside
buildings, and it is a differentiable proxy for knife-edge shadow depth. It is
implemented with `grid_sample` in nearest-neighbour mode, evaluated at 31 interior
sample points and normalised by `N = 32`, so the whole field for a batch is produced by
31 vectorised gathers with no Python loop over pixels.

The motivation is diagnostic rather than aesthetic. Line-of-sight and non-line-of-sight
error diverge sharply for every architecture we tested, and unrelated backbones plateau
at the *same* non-line-of-sight value, which is the signature of a missing input rather
than of insufficient capacity. `O` supplies precisely the quantity that a convolutional
receptive field cannot construct: a long, thin, geometry-dependent path integral whose
support is a line rather than a neighbourhood.

---

## 4. Backbone: the Conditioning U-Net

The backbone is a deliberately standard encoder–decoder, so that ablations differ in
their *inputs and losses* rather than in their architecture. All numbers below are for
`base = 32`, `depth = 4`, which is the configuration used in the reported WNet model.

### 4.1 Convolutional block

Every stage is built from one block:

```
Conv3×3(cin → cout, pad 1) → GroupNorm(min(8,cout), cout) → SiLU
Conv3×3(cout → cout, pad 1) → GroupNorm(min(8,cout), cout) → SiLU
[ Dropout2d(p) ]                                    # only where p > 0
```

GroupNorm is used rather than BatchNorm because the effective batch is small (16) and
because MC-dropout inference (§7) runs the network in a stochastic mode where BatchNorm's
running statistics would be inconsistent between the training and sampling passes.

### 4.2 Encoder

Channel widths are `chs = [32, 64, 128, 256]`. Note that the input block runs at **full
resolution** and each subsequent stage pools *before* convolving:

| Stage | Operation | Output | Dropout |
|---|---|---|---|
| `inc` | block(9 → 32) | `32 × 256 × 256` → skip 1 | — |
| down 0 | maxpool 2 → block(32 → 64) | `64 × 128 × 128` → skip 2 | — |
| down 1 | maxpool 2 → block(64 → 128) | `128 × 64 × 64` → skip 3 | — |
| down 2 | maxpool 2 → block(128 → 256) | `256 × 32 × 32` → skip 4 | **0.15** |
| `mid` | block(256 → 256) | `256 × 32 × 32` | **0.15** |

Dropout is placed only at the two deepest stages. This is a deliberate choice for the
uncertainty mechanism: stochasticity at the bottleneck perturbs the *semantic* summary of
the scene, producing spatially coherent variation across MC samples, whereas dropout near
full resolution would produce high-frequency speckle that reflects sampling noise rather
than genuine predictive ambiguity.

### 4.3 FiLM conditioning at the bottleneck

The sequence vector `s ∈ R^{64}` (§5) enters exactly once, at the bottleneck, as a
feature-wise linear modulation:

```
[γ, β] = Linear(64 → 512)(s)              γ, β ∈ R^{256}
h ← h · (1 + γ)  +  β                      broadcast over the 32×32 spatial grid
```

The residual form `(1 + γ)` means an untrained or uninformative sequence encoder leaves
the bottleneck unchanged, so order conditioning starts as a no-op and must earn its
effect. This matters for the order ablation to be interpretable.

### 4.4 Decoder

Three upsampling stages, each a transposed convolution followed by concatenation with the
corresponding encoder skip:

| Stage | Operation | Output |
|---|---|---|
| up 0 | ConvT(256 → 128, k2 s2), cat skip 3 → block(256 → 128) | `128 × 64 × 64` |
| up 1 | ConvT(128 → 64, k2 s2), cat skip 2 → block(128 → 64) | `64 × 128 × 128` |
| up 2 | ConvT(64 → 32, k2 s2), cat skip 1 → block(64 → 32) | `32 × 256 × 256` |
| `outc` | Conv1×1(32 → 1) | `1 × 256 × 256` |

The output is a normalised radio map in the same `[−1,1]` convention as the target. No
output activation is applied, so the network is free to produce values outside the range
and is disciplined only by the loss.

---

## 5. Sequence encoder (walk order)

Crowdsensed measurements arrive as *ordered* walks, not as an unordered point cloud. The
raster channels of §2 discard that order entirely, so it is reintroduced through a
separate branch.

Each measurement contributes an 11-dimensional feature vector:

```
[ t, Δt, row/256, col/256, speed, sin(heading), cos(heading),
  ‖p − τ‖/256, v_radial, v_tangential, normalised RSS ]
```

`v_radial` is the velocity component along the bearing to the transmitter (positive when
approaching) and `v_tangential` its signed perpendicular component. These two are the
features that let the encoder relate a *change* in signal to a *direction* of motion,
which is the physical content that ordering carries.

Trajectories are concatenated and padded or truncated to `N = 256` points, then encoded:

```
GRU(11 → 64, 2 layers, bidirectional, batch_first)   →  (B, 256, 128)
mean over the time axis                              →  (B, 128)
Linear(128 → 64) → SiLU                              →  s ∈ R^{64}
```

The mean pool is applied to the *GRU outputs*, not to the raw features. This is the
crucial detail: mean pooling is permutation-invariant, but the recurrence that produces
those outputs is not, so `s` remains order-sensitive. This is what makes the
shuffled-order control a genuine falsification test rather than a formality — a model
that ignored order would score identically under shuffling, and ours does not.

---

## 6. Device head

Handsets disagree about the same location by a median of roughly 4 dB, which is a
first-order effect at the scale of the reconstruction error itself. The device branch
predicts a single additive offset per sample from permutation-invariant summary
statistics of the observed measurements:

```
stats = [ mean, std, min, max ]  of  sparse_rss[mask > 0.5]      ∈ R^4
Linear(4 → 32) → SiLU → Linear(32 → 1)                           → δ ∈ R
ŷ ← ŷ + δ                                                        broadcast over H×W
```

Two properties are deliberate. The head is **permutation-invariant** so that it captures
a device's characteristic level rather than anything about walk order, keeping it
orthogonal to §5. And the offset is **spatially constant**, which restricts it to exactly
the transformation a device miscalibration can produce and prevents it from absorbing
spatial error that the reconstruction network should be modelling.

---

## 7. Uncertainty via MC dropout

At inference the dropout layers of §4.2 are **kept active** and the network is evaluated
`n_MC = 8` times:

```
mean  = (1/n) Σ_k ŷ_k        →  converted to dBm
sigma = std_k(ŷ_k) × 69.5    →  predictive spread in dB
```

The mean also functions as a small ensemble, which is a modest accuracy benefit; the
standard deviation is the acquisition surface used for active collection. We treat the
*calibration* of this spread as a claim to be tested, not a property to assume, and
report both the raw miscalibration and the post-hoc correction rather than only the
corrected result.

---

## 8. The WNet cascade (the reported model)

The single U-Net of §4 is not the strongest configuration. The reported model,
**COMPASS-WNet**, cascades two conditioning U-Nets under deep supervision:

```
X                    = [measurements ‖ building ‖ tx heatmap ‖ occlusion]     (9 ch)
s                    = SequenceEncoder(sequence)                              (64-d)
coarse               = UNet₁(X, s)                                            (1 ch)
refined              = UNet₂([X ‖ coarse], s)                                 (1 ch)
ŷ                    = refined + δ                                            (device offset)
```

Three implementation points matter:

1. **Capacity is held constant, not doubled.** The per-network width is scaled to
   `b = max(16, ⌊round(0.72 · base)/8⌋ · 8) = 32` for `base = 48`, rounded to a multiple
   of 8 for GroupNorm. Two 32-wide U-Nets total **6.40 M parameters**, *fewer* than the
   single 48-wide U-Net at **7.16 M**. The cascade therefore wins on structure rather
   than on budget, which is what makes the comparison meaningful.
2. **Both stages see the full conditioning.** The second stage receives the original 9
   channels *plus* the coarse map (10 channels), so refinement is conditioned on the
   evidence rather than only on the first stage's opinion of it.
3. **The FiLM vector is shared** between stages, and the device offset is applied **once**
   to the refined output, so neither mechanism is silently applied twice.

The coarse map is exposed as `self.aux` for deep supervision (§9.4).

---

## 9. Training objective

The total loss combines three terms plus deep supervision. All terms operate in
normalised units.

### 9.1 Masked reconstruction (weight 1.0)

A weighted MSE that makes the metric-relevant region dominate the gradient:

```
w_p = 2.0   if p ∈ Ω          (observed)
      1.0   if p ∈ F \ Ω      (free space, unobserved)  ← the reported metric
      0.1   if B_p = 1        (building interior)

L_recon = Σ_p w_p (ŷ_p − y_p)² / Σ_p w_p
```

Building interiors are down-weighted rather than excluded, which keeps the noise floor
learnable without letting roughly a quarter of the pixels dominate a loss that is
evaluated on the other three quarters.

### 9.2 Trajectory consistency (weight 0.5)

```
L_traj = mean_{p ∈ Ω} (ŷ_p − ỹ_p)²
```

This anchors the prediction to the actual measurements at observed pixels. Without it the
smoothing pressure of `L_recon` lets the network denoise its own conditioning away, which
degrades the reconstruction everywhere downstream of an observation.

### 9.3 Ray consistency (weight 0.1)

A physics prior evaluated along `R = 64` bearings from the transmitter, at `S = 48` radial
steps out to 200 px, sampled bilinearly:

```
Δ_r,s   = ŷ(r, s+1) − ŷ(r, s)                     step-to-step change outward
occ_r,s = B(r, s+1)                                building occupancy of the outer step
L_rcl   = mean[ (1 + 3 · occ) · ReLU(Δ)² ]
```

The term penalises signal *increasing* with distance from the transmitter, and penalises
it roughly four times harder when the step enters a building. It is a per-ray, geometry-
grounded monotonicity constraint rather than a global inequality or a dense PDE residual.

> **Honest note.** This term is **active in every reported model** at weight 0.1. Its
> dedicated ablation moves validation RMSE from 11.190 to 11.174 dB, a difference of
> 0.016 dB, i.e. no measurable effect. It is retained for reproducibility of the released
> checkpoints, not because it is doing useful work. See §12.

### 9.4 Deep supervision (WNet only)

```
L = L_recon + 0.5·L_traj + 0.1·L_rcl  +  0.5 · L_recon(coarse, y)
```

The coarse stage is supervised directly so that stage 1 must produce a usable map on its
own rather than an arbitrary intermediate code, which keeps the cascade from collapsing
into one effective network.

---

## 10. Optimisation and data

| Setting | Value |
|---|---|
| Optimiser | AdamW, `lr = 2e-4`, weight decay `1e-2` |
| Schedule | Cosine annealing to the epoch budget |
| Gradient clipping | global norm 1.0 |
| Batch size | 16 |
| Epochs | 150 |
| Training data | 120 maps × 20 transmitters; 24 maps held out for validation |
| Variant | RadioMapSeer `IRT2` |
| Selection | best free-unobserved validation RMSE |

**Trajectory simulation.** Each sample draws `k = 3` walks of 100 points, mixing three
motion styles: A\* shortest path over the free-space graph, a momentum random walk
(momentum 0.8, step 2 px), and a corridor-biased walk preferring street centres
(preference 0.7). Mixing styles prevents the model from overfitting the statistics of any
single motion prior.

**Measurement noise.** Noise is spatially correlated rather than i.i.d., because i.i.d.
noise averages out trivially and would make the task unrealistically easy. Each map draws
one Gudmundson shadowing field with `σ = 6 dB` and a 30 m decorrelation length, on top of
which the pipeline applies a fast-fade residual, a per-device affine offset and gain, 1 dB
quantisation, and noise-floor clipping.

---

## 11. Configurations and complexity

Every ablation is the same class under a different flag set, so no comparison is confounded
by an incidental code difference.

| Configuration | Flags changed | Params | Val RMSE (dB) |
|---|---|---|---|
| **COMPASS-WNet + occlusion** | `arch=wnet, occlusion=on` | 6.40 M | **8.26** |
| COMPASS-WNet | `arch=wnet` | 6.40 M | 9.59 |
| COMPASS single U-Net | — | 7.16 M | 10.91 |
| no building | `use_building=off` | 7.16 M | 17.93 |
| no transmitter | `use_tx=off` | 7.16 M | 15.18 |
| order blind | `use_order=off` | 7.16 M | 11.25 |
| order shuffled | `order_shuffle=on` | 7.16 M | 11.17 |
| no device head | `use_device=off` | 7.16 M | 11.10 |
| no ray-consistency loss | `use_rcl=off` | 7.16 M | 11.17 |

Inference cost is 6.7 ms per map for a single deterministic pass and 53 ms with the 8-pass
MC-dropout uncertainty, on one GPU at 256 × 256.

---

## 12. Implementation notes and discrepancies

This section records where the code differs from previous descriptions of the system, so
that the two can be reconciled rather than silently diverging.

1. **The ray-consistency loss is active, not retired.** It has been described in prior
   summaries as "tested and retired because it moved no metric." The first half is
   accurate and the second is not: `use_rcl = True` with weight 0.1 in the configuration
   of *every* released checkpoint, including the reported `wnet_occ` model. Its ablation
   shows a 0.016 dB effect, which is why the "moved no metric" conclusion is sound, but
   the term was never removed from the objective. Any paper text should say *retained but
   ablated to no measurable effect*.

2. **The WNet reduces total capacity.** It is natural to assume that cascading two U-Nets
   doubles the parameter count. The width rescaling means the opposite is true here: 6.40 M
   against 7.16 M for the single network.

3. **Pooling precedes convolution** in the encoder, so the first block runs at full
   resolution and each later stage is computed after downsampling. The skip resolutions
   are therefore 256, 128, 64, 32, and the bottleneck is 32 × 32 rather than 16 × 16.

4. **Occlusion normalisation.** The path integral sums 31 interior samples but divides by
   `N = 32`, so `O` is very slightly biased low relative to a true mean. The effect is a
   constant factor of 31/32 and is absorbed by the first convolution.

5. **The device offset is applied after the cascade**, not inside either stage, so it
   cannot be double-counted and does not interact with deep supervision.

### Empirical verification of the stated properties

Several claims above are behavioural rather than structural, so they were checked by
executing the modules rather than by reading them:

| Claim | Check | Result |
|---|---|---|
| Dropout only at the two deepest stages (§4.2) | enumerate `Dropout2d` modules | `downs.2`, `mid` only |
| Occlusion is a shadow-depth proxy (§3.2) | synthetic building block, transmitter outside | `O = 0.000` unobstructed, `O = 0.438` in shadow |
| Sequence encoding is order-sensitive (§5) | encode a sequence and a permutation of it | `‖Δ‖ = 1.2e-2`, non-zero |
| Device head is permutation-invariant (§6) | flip the spatial layout of the same measurements | `|Δδ| = 0.0` exactly |
| Parameter counts (§8, §11) | instantiate both models | 6.40 M cascade, 7.16 M single |

### Source provenance

| Component | File |
|---|---|
| Config, occlusion field, TX heat map, sequence encoder, device head | `src/compass/models/compass_net.py` |
| WNet cascade | `src/compass/models/compass_wnet.py` |
| Conditioning U-Net, block, FiLM | `src/compass/models/unet.py` |
| Losses | `src/compass/training/losses.py` |
| Training loop and config | `src/compass/training/train.py` |
| Trajectory features, conditioning rasters | `src/compass/data/conditioning.py` |
| Normalisation constants | `src/compass/data/conventions.py` |
| Trajectory simulation | `src/compass/data/trajectory.py` |
| Noise model | `src/compass/data/noise.py` |
| MC-dropout inference | `src/compass/recon/learned.py` |

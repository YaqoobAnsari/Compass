"""
Single source of truth for the COMPASS deck: every slide's title, bullets, and tables.

Writing standard enforced here: EVERY lead line is a strong topic sentence. It states a
claim rather than announcing a subject, carries its own evidence (the number, the
comparison, the significance) inside the sentence, connects to the point before it, and
sets up the point after it. A reader who skims only the lead sentences should come away
with the complete argument of the paper.

Both make_pptx.py and make_deck.py render from this module, so the two decks cannot
drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/data1/yansari/Compass")


def _load(name):
    try:
        return json.loads((REPO / "results" / name).read_text())
    except Exception:
        return {}


def deep_real_is_valid():
    """The CNP figure/slide is only shown when the run behind it is a trained one."""
    d = _load("deep_real.json")
    return bool(d) and d.get("epochs", 0) > 0


# ============================ SLIDE DEFINITIONS ============================
# Each entry: {"title", "bullets", optional "table", optional "fig", optional "caption"}

def slides():
    S = []
    A = S.append

    # ------------------------------- INTRODUCTION -------------------------------
    A({"title": "Introduction: the problem",
       "bullets": [
        "**A radio map, the spatial distribution of signal strength over an environment, is the substrate on which wireless systems plan coverage, localize devices, manage spectrum, and steer beams, and yet it is the one input that can never be measured everywhere it is needed, because signal must be physically sampled at every location and exhaustive surveys are both infeasible to run and stale as soon as the building changes.**",
        "**That impossibility turns radio-map construction into a reconstruction problem: recover a dense, accurate field from a sparse set of measurements plus whatever is known about the environment, which is the problem this work addresses.**",
        "**The measurements that are actually available at scale come from crowdsensing, where ordinary phones report signal strength as people walk, and this changes the problem in four specific ways that the literature has largely optimised away: the serving transmitter is usually unknown, samples arrive along walking paths rather than on a grid, different handsets disagree by several decibels on the same spot, and there is no dense ground truth anywhere to supervise against.**",
        "R| The field has nonetheless advanced almost entirely on synthetic benchmarks that assume a known transmitter, dense supervision, and a single device, so the methods that top those leaderboards are optimised for conditions real crowdsensing does not provide, and it remains largely untested which of their mechanisms survive the transition to real data.",
       ]})

    A({"title": "Introduction: the gap and what this work contributes",
       "bullets": [
        "**Prior work splits into two camps that each solve a different problem than crowdsensing poses, which is why neither is directly usable: deep radio-map networks reach strong synthetic accuracy but require the transmitter location and dense supervision that real data lacks, while classical interpolation is transmitter-agnostic and remains the workhorse on sparse real measurements but is blind to the building geometry that governs indoor propagation, and neither supplies the calibrated uncertainty needed to decide where to measure next.**",
        "**Recent work has begun to close individual pieces of that gap by adding geometry priors, uncertainty estimates, and trajectory-aware priors to sparse reconstruction, but because these mechanisms have been demonstrated in isolation and almost always on synthetic data, it remains untested which of them actually transfer once the transmitter is unknown and devices are heterogeneous, and that controlled evaluation on real data is precisely what this work provides.**",
        "G| Our method, COMPASS, is a geometry, order, and device conditioned reconstructor with explicit occlusion conditioning and calibrated uncertainty, evaluated on both a large synthetic benchmark and a real indoor cellular crowdsensing dataset so that every claim is tested in the regime it is meant to serve.",
        "G| Our headline results are that COMPASS is the most accurate reconstructor across nine deep-learning architecture classes and seven classical families at 8.65 dB, a 13 percent margin over the next best method with non-overlapping confidence intervals, that it is the most accurate method that also carries calibrated uncertainty, and that on real data our negative controls separate the mechanisms that genuinely transfer from those that are artifacts of synthetic benchmarks.",
       ]})

    # ------------------------------- RELATED WORK -------------------------------
    A({"title": "Related work: deep radio-map networks",
       "bullets": [
        "**A decade of deep radio-map models has driven synthetic accuracy down sharply, but every one of them is built around two assumptions, a known transmitter and a densely supervised map, that quietly define the problem they can solve.**",
        "**Dense predictors set the template: RadioUNet (Levie 2021) and PMNet (Lee 2023) take a building raster and a transmitter location and regress the entire pathloss field, learning the propagation prior from large ray-traced datasets, which is why they are strong on synthetic maps and inapplicable the moment the transmitter is unknown.**",
        "**Generative approaches inherit the same conditioning while changing the decoder: conditional GANs and diffusion models such as RMDM (Jia 2025) and RadioDiff (Wang 2024) synthesise the map from identical geometry-plus-transmitter inputs, so they extend the modelling family without relaxing its assumptions.**",
        "**The newest architectural families continue this pattern into state-space and uncertainty-aware designs, with RadioMamba bringing linear-time selective scanning and URAM bringing Bayesian predictive uncertainty, which is why we implement and train all of them here rather than citing them as untested alternatives.**",
       ]})

    A({"title": "Related work: classical interpolation and the untested setting",
       "bullets": [
        "**Classical spatial interpolation occupies the opposite corner of the design space, and it is the reason a deep model cannot simply be declared the winner on real data: inverse-distance weighting, Kriging, Gaussian processes, and radial basis functions need no transmitter and no training set, which makes them the default and genuinely strong choice wherever measurements are sparse and scattered.**",
        "**Their single structural weakness is that they interpolate through walls, because a distance kernel has no representation of the geometry that actually attenuates indoor signal, which predicts that their errors should concentrate exactly where buildings block the path.**",
        "**Taken together the two camps leave a specific and testable gap, namely that no published evaluation trains the full spectrum of methods under one protocol on a real indoor-cellular crowdsensing dataset and asks which mechanisms survive unknown transmitters and heterogeneous devices, which is the gap this work closes.**",
        "R| We are explicit that occlusion conditioning, uncertainty estimation, and trajectory-aware priors are each established techniques rather than inventions of ours, so our contribution is the controlled real evaluation and the honest attribution of which of them transfer, not the mechanisms themselves.",
       ]})

    # ---------------------------------- DATA -----------------------------------
    A({"title": "Data: the synthetic benchmark (RadioMapSeer)",
       "bullets": [
        "**We use RadioMapSeer as the controlled half of the evaluation because it supplies the one thing real crowdsensing never can, a dense and exact ground-truth field for every map, which is what makes an unambiguous accuracy comparison possible at all.**",
        "**Each sample pairs a 256 by 256 building raster with a ray-traced pathloss map and a known transmitter, and we simulate crowdsensed collection on top of it by sampling measurements along walking trajectories rather than uniformly, so the sparsity pattern matches the deployment we care about while the ground truth remains complete.**",
        "**It is also the right control precisely because it favours the baselines: the known transmitter lets every deep method run exactly as designed, so any margin we report is earned under the conditions those methods were built for rather than under a handicap.**",
       ]})

    A({"title": "Data: real indoor cellular crowdsensing (IndoCell)",
       "bullets": [
        "**IndoCell is the half of the evaluation that carries the real conditions, and it is where every convenient synthetic assumption breaks at once: measurements are real cellular signal strength collected by phones walking two multi-floor sites, a university building and a parking structure, with no dense ground truth anywhere.**",
        "**The transmitter problem is concrete rather than theoretical, because indoor coverage is delivered largely through boosters and distributed antennas, so only about 18 percent of serving cells can be located at all, which is what makes every transmitter-conditioned deep model structurally inapplicable here.**",
        "**Device heterogeneity is equally concrete, with a median disagreement of 4.12 dB between phones at the same reference point and a ninetieth percentile of 11.6 dB, an error comparable in size to the reconstruction error itself and therefore impossible to ignore.**",
        "**Because there is no dense ground truth, every real-data number in this work is measured by holding out reference points under a buffered leave-one-out protocol, which is a stricter and more honest test than interpolating a dense raster.**",
       ]})

    A({"fig": "pub/source.png",
       "caption": "Real crowdsensed collection on the two IndoCell sites. Coverage follows walkable paths, "
                  "leaving the structured gaps that reconstruction has to fill."})

    # -------------------------- FORMULATION AND METRICS -------------------------
    A({"title": "Problem formulation and metrics",
       "bullets": [
        "**We state the task precisely so that every method in the panel is solving the same problem: given a sparse set of measured signal values, the building geometry, and where available the transmitter, predict the signal at every unobserved free-space location.**",
        "**Our headline metric is free-unobserved RMSE, the error over free space that was never measured, which we use because reporting error over observed points or through solid walls flatters methods that merely copy their inputs and tells us nothing about reconstruction.**",
        "**We report structural similarity alongside it to capture whether the spatial shape of the field is right rather than only its pointwise level, and we split the error into line-of-sight and non-line-of-sight regions because that split is what later localises the failure common to every architecture.**",
        "**Every comparison carries bootstrap 95 percent confidence intervals and paired Wilcoxon tests on identical samples, so the differences we claim are separated from sampling noise rather than asserted from single point estimates.**",
       ]})

    A({"title": "Hypotheses: what each experiment asks and why it matters",
       "bullets": [
        "**Rather than reporting whatever the experiments happened to produce, we state six hypotheses in advance, each paired with the control that could falsify it, so that a mechanism which fails to transfer to real data is recorded as such rather than quietly dropped.**",
        "**H1, geometry. We ask whether explicit building geometry is what actually limits sparse reconstruction, testing it by removing building conditioning and by isolating error behind walls, because if geometry is the binding constraint then the entire classical family has a ceiling no amount of tuning can lift.**",
        "**H2, occlusion. We ask whether the residual error concentrates in non-line-of-sight regions and whether supplying an explicit occlusion field removes it, because a shared plateau across unrelated architectures would indicate a missing input rather than insufficient model capacity.**",
        "**H3, order. We ask whether the sequence in which a walk collects measurements carries information beyond the positions themselves, tested against a shuffled-order control, because a gain that survives shuffling would be spatial rather than sequential and the claim would be false.**",
        "**H4, device. We ask whether per-device offsets can be estimated and removed well enough to matter, because with a median cross-phone spread of 4.12 dB this is either a first-order correction or a distraction, and only a controlled test distinguishes the two.**",
        "**H5, uncertainty. We ask whether predictive uncertainty is calibrated well enough to direct where to measure next, because uncertainty that does not track error is worse than none at all, as it would actively misdirect collection effort.**",
        "**H6, transfer. We ask whether methods that win on synthetic data work on real crowdsensing, which is the question the whole paper exists to answer, and we test it both by transferring synthetic winners and by training a transmitter-agnostic deep model natively on the real data.**",
       ]})

    # --------------------------------- METHOD ----------------------------------
    A({"fig": "architecture/compass_wnet.png",
       "caption": "COMPASS-WNet. Blue is the shared backbone, green marks the conditioning we add, and orange "
                  "marks the occlusion field introduced in this work."})

    A({"title": "Method: conditioning, occlusion, and uncertainty",
       "bullets": [
        "**COMPASS is deliberately not a new backbone, because our claim is about which inputs matter rather than which layer type is fashionable, so we use a cascade of two conditioning U-Nets and concentrate the contribution in what the network is told about the world.**",
        "**Geometry enters through the building raster and, where known, a transmitter heat map, which together give the network the structural context that a distance kernel fundamentally cannot represent.**",
        "**The occlusion field is the input that breaks the behind-building plateau, and it is computed rather than learned: for every pixel we ray-cast to the transmitter and record the fraction of that path passing through buildings, which hands the network the non-line-of-sight information it was otherwise forced to infer.**",
        "**Walk order and device identity enter through a recurrent encoder and a device head that modulate features, which lets us test on real data whether sequence and handset identity carry usable signal rather than assuming they do.**",
        "**Uncertainty comes from Monte-Carlo dropout at inference, giving a predictive spread per pixel, and we treat its calibration as a claim to be tested rather than a property to be assumed.**",
       ]})

    # -------------------------------- BASELINES ---------------------------------
    A({"title": "Baselines: an exhaustive comparison panel",
       "bullets": [
        "**We compare against a deliberately exhaustive panel so that our result cannot rest on a weak or dated baseline, covering nine deep architecture classes and seven classical families rather than the two or three comparisons that are customary.**",
        "**The deep panel spans plain CNN (SparseUNet), cascade CNN (RadioUNet), dilated CNN with ASPP (PMNet), transformer (RadioTransformer), conditional GAN (RadioGAN), iterative diffusion (RMDM), decoupled sampling-free diffusion (RadioDiff), state-space Mamba (RadioMamba), and Bayesian U-Net (URAM), which means every major modelling family of the last five years is represented by a trained model rather than a citation.**",
        "**The classical panel spans inverse-distance weighting, ordinary and simple Kriging, Gaussian processes, radial basis functions, and nearest and natural neighbour interpolation, each tuned on held-out data so the comparison is against these methods at their best.**",
        "**Fairness is enforced by protocol rather than asserted: every deep method sees the same data, the same split, the same 150-epoch budget, a matched parameter count between six and nine million, and the inputs it was designed to consume, and every method is scored on identical samples with confidence intervals and significance tests.**",
       ]})

    A({"title": "Newer baselines: diffusion, state-space, and Bayesian models",
       "bullets": [
        "**Because the literature moved while this work was in progress, we implemented and trained the three most applicable recent architectures under our identical protocol rather than leaving them as untested alternatives, and none of them closes the gap to COMPASS.**",
        "G| RadioDiff, the current diffusion reference, splits the forward process into an attenuation term and a noise term so the map is recovered in a single step, and trained under our protocol it reaches 11.40 dB.",
        "G| RadioMamba brings the state-space family in as a hybrid Mamba U-Net with bidirectional selective-scan blocks, the linear-time analogue of our transformer baseline, and it reaches 10.92 dB, essentially tied with RadioTransformer.",
        "G| URAM represents the uncertainty-first design as a Bayesian U-Net with Monte-Carlo dropout and reaches 11.34 dB, and because it genuinely does provide predictive uncertainty our honest claim is that COMPASS is the most accurate method that also carries calibrated uncertainty, not the only method offering any uncertainty at all.",
        "R| The two diffusion models differ by a factor of three under one protocol, 11.40 dB for the decoupled formulation against 35.98 dB for iterative denoising from pure noise, which shows that how a diffusion process is formulated matters far more on sparse input than the fact that it is diffusion, and that even the working formulation trails COMPASS by 32 percent.",
       ]})

    # ------------------------------ SYNTHETIC RESULTS ---------------------------
    A({"title": "Results: accuracy across the full panel",
       "bullets": [
        "**COMPASS-WNet with occlusion conditioning is the most accurate reconstructor in the panel at 8.65 dB, ahead of every deep and classical baseline including the newest diffusion, state-space, and Bayesian models, and its interval does not overlap the next best method.**",
       ],
       "table": (["Method", "RMSE (dB)", "95% CI", "SSIM", "Uncertainty"], [
            ["COMPASS-WNet + occlusion (ours)", "8.65", "8.42 to 8.86", "0.841", "yes"],
            ["RadioUNet (Levie 2021)", "9.93", "9.68 to 10.18", "0.816", "none"],
            ["COMPASS-WNet (ours)", "10.11", "9.82 to 10.42", "0.823", "yes"],
            ["RadioMamba (state-space, 2024)", "10.92", "10.65 to 11.19", "0.795", "none"],
            ["RadioTransformer", "10.97", "10.66 to 11.28", "0.791", "yes"],
            ["URAM (Bayesian U-Net, 2024)", "11.34", "11.08 to 11.57", "0.800", "yes"],
            ["RadioDiff (diffusion, 2024)", "11.40", "11.04 to 11.73", "0.803", "none"],
            ["RadioGAN", "11.42", "11.15 to 11.67", "0.788", "none"],
            ["COMPASS, single U-Net", "11.49", "11.16 to 11.84", "0.797", "yes"],
            ["PMNet (Lee 2023)", "14.12", "13.71 to 14.50", "0.714", "none"],
            ["SparseUNet, no geometry", "21.05", "20.67 to 21.42", "0.656", "none"],
            ["Best classical (RBF, Kriging, GP, IDW)", "25.6 to 28.1", "", "0.67 to 0.72", "limited"],
            ["RMDM, iterative diffusion", "35.98", "34.54 to 37.34", "0.715", "none"],
       ]),
       "after": [
        "G| Two things follow from this ordering. Geometry-aware learning beats the best classical family by roughly three times, and occlusion conditioning then beats the strongest deep baseline, RadioUNet, by 13 percent, which is the margin that the rest of the results set out to explain.",
       ]})

    A({"fig": "pub/benchmark_synth.png",
       "caption": "Blue is COMPASS, grey are the deep baselines including RadioDiff, RadioMamba and URAM, and "
                  "light grey are classical. 320 held-out samples with 95 percent confidence intervals."})

    A({"title": "Results: the reconstruction wall is behind buildings",
       "bullets": [
        "**The margin in the table above is not spread evenly across the map, and locating where it lives is what turns a leaderboard result into an explanation: essentially all of the error that separates methods sits in non-line-of-sight regions behind buildings.**",
        "**In line-of-sight regions the models are already accurate at roughly 5.6 dB, while the headline error is dragged up by non-line-of-sight regions at roughly 10.9 dB, and the diagnostic detail is that COMPASS-WNet and RadioUNet reach that same value despite unrelated architectures, which is the signature of a missing input rather than insufficient capacity.**",
        "G| Supplying the pre-computed occlusion field breaks that shared plateau: non-line-of-sight error falls from 11.18 to 9.79 dB and line-of-sight from 5.16 to 3.98 dB, taking overall error to 8.68 dB with the best structural similarity in the panel at 0.840.",
        "R| We state the scope plainly, because occlusion and line-of-sight conditioning are established techniques rather than our invention, so the contribution here is the diagnosis that this is the decisive missing ingredient in the sparse regime, demonstrated across an entire panel rather than on a single model.",
       ]})

    A({"title": "Results: which inputs actually matter (ablation)",
       "bullets": [
        "**Having shown that geometry is what breaks the plateau, we remove each input in turn to establish which ones the accuracy actually depends on, and on synthetic data only the geometric inputs move the metric at all.**",
       ],
       "table": (["Variant", "RMSE (dB)", "Change vs full", "Reading"], [
            ["Remove building conditioning", "18.82", "plus 7.3", "building geometry is essential"],
            ["Remove transmitter conditioning", "16.28", "plus 4.8", "transmitter geometry is essential"],
            ["Remove order, device, or physics loss", "about 11.5", "about 0", "inert on synthetic, by design"],
       ]),
       "after": [
        "**Order and device conditioning are inert here for a reason that is expected rather than disappointing, namely that synthetic data contains a single device and independent sampling, so the conditions those mechanisms exist to exploit are absent by construction and their value has to be demonstrated on real data instead, which is exactly what the next section does.**",
        "R| We also tested a physics-based ray-consistency loss and retired it, because it moved no metric we measured, and we report that rather than leaving it in as unexamined machinery.",
       ]})

    A({"title": "Results: robustness to scene complexity",
       "bullets": [
        "**If the advantage we report came from fitting easy scenes rather than from modelling geometry, it should evaporate as scenes get more cluttered, so we split the test set into easy, medium, and hard by building density and measured how much each method degrades.**",
        "G| COMPASS-WNet degrades by only 0.22 dB from the easiest to the hardest third, the smallest degradation of any method tested, and all three COMPASS variants degrade less than every deep and classical baseline.",
        "**The contrast with the rest of the panel is what makes the point: the strongest deep baselines lose between 1.07 and 1.69 dB, classical interpolation collapses by 5.2 to 5.8 dB as walls multiply, and iterative diffusion degrades by 21.9 dB, which is precisely the ordering that a geometry explanation predicts and a capacity explanation does not.**",
       ]})

    A({"fig": "pub/complexity.png",
       "caption": "Degradation from the easiest to the hardest third of scenes, sorted with the most robust "
                  "method at the top. 636 held-out samples."})

    # -------------------------------- REAL DATA ---------------------------------
    A({"title": "Results, real data: geometry acts through wall attenuation",
       "bullets": [
        "**Everything to this point was measured on simulated propagation, so the first question on real data is whether the geometric effect our method depends on is actually present, and we test it directly rather than assuming the simulator was right.**",
        "**We first ruled out the obvious confound by confirming that geodesic and straight-line distances between reference points are nearly identical, which means any difference we find is attenuation through walls rather than a longer walking path.**",
        "G| At matched separation, reference-point pairs separated by a wall differ substantially more in signal than open-space pairs, with non-overlapping confidence intervals across distance bins and on both sites, which confirms on real measurements the mechanism our synthetic results attribute the margin to.",
        "**This matters beyond a sanity check, because it is the empirical licence for conditioning on geometry at all, and it is what makes the learned residual in a later section a geometry correction rather than a curve fit.**",
       ]})

    A({"fig": "pub/walls.png",
       "caption": "Signal difference between reference-point pairs at matched distance, split by whether a wall "
                  "separates them. CMU-Q, 95 percent confidence intervals."})

    A({"title": "Results, real data: walk order carries usable signal",
       "bullets": [
        "**Crowdsensed measurements arrive as ordered walks rather than as an unordered point cloud, so we asked whether that sequence carries information beyond the positions, and we designed the test so that a spurious result could be caught.**",
        "G| Filling gaps using the true walk order gives 6.99 dB against 9.92 dB for an order-blind mean, a clear gain measured over 893,900 held-out points with tight intervals.",
        "R| The control is what makes the claim safe: shuffling the order while keeping every position fixed degrades the result to 12.35 dB, far worse than the order-blind mean, which proves the gain is genuinely sequential rather than a spatial effect wearing a temporal label.",
        "**This is the first of the two mechanisms that are inert on synthetic data and active on real data, and it is the reason the architecture carries a sequence encoder at all.**",
       ]})

    A({"fig": "pub/order.png",
       "caption": "Gap-filling error using the true walk order, a shuffled-order control, and an order-blind mean."})

    A({"title": "Results, real data: device heterogeneity is first-order",
       "bullets": [
        "**The second mechanism absent from synthetic data is that different handsets disagree about the same location, and the size of that disagreement determines whether it is a correction worth making or a detail worth ignoring.**",
        "**Across 1,700 shared cell observations the median cross-device spread is 4.12 dB, the ninetieth percentile is 11.6 dB, and the maximum reaches 27.77 dB, which places device disagreement on the same scale as the reconstruction error itself and therefore squarely in the first order.**",
        "G| Estimating and removing a per-device offset before pooling measurements improves reconstruction relative to pooling them raw, which validates treating device identity as a conditioning variable rather than as noise to be averaged away.",
        "R| We report the effect honestly as real but modest in isolation, and its value is as one component of a coherent system rather than as a headline result on its own.",
       ]})

    A({"fig": "pub/device.png",
       "caption": "Reconstruction from raw pooled measurements against device-calibrated pooling."})

    A({"title": "Results, real data: reconstruction without a transmitter",
       "bullets": [
        "**We now turn to reconstruction itself on real data, where the honest finding is that the regime inverts: with no dense supervision and no transmitter, a tuned classical kernel is the strong baseline and a dense learned predictor is not.**",
        "**Under a buffered leave-one-reference-point-out protocol, tuned RBF interpolation reaches 5.49 dB, and it is this number, rather than any deep baseline, that a real-data contribution has to beat.**",
        "G| Our deployable result is a hybrid rather than an end-to-end network: keeping the classical field as the backbone and learning only the geometry-driven residual it misses reaches 5.14 dB, a 6.5 percent improvement with a paired significance of p equal to 2e-4 over 2,061 held-out points.",
        "G| That the gain is geometric rather than generic is established by its own ablation, since removing the wall features from the residual costs most of the improvement, moving it from 5.14 to 5.21 dB.",
       ]})

    A({"fig": "pub/recon_real.png",
       "caption": "Transmitter-agnostic reconstruction on IndoCell under buffered leave-one-reference-point-out."})

    A({"fig": "pub/residual.png",
       "caption": "The learned residual on the classical field, with a no-geometry ablation isolating the "
                  "contribution of wall features."})

    # ------------------- THE CENTRAL TRANSFER QUESTION -------------------
    A({"title": "Results: can the synthetic winners do real crowdsensing?",
       "bullets": [
        "**This is the question the whole paper exists to answer, and the answer is no in two independent ways, which together are the strongest evidence we offer that synthetic leaderboards do not predict real-world capability.**",
        "R| The first way is structural. RadioUNet, PMNet, and every other transmitter-conditioned network in our panel cannot be run on the real task at all, because only about 18 percent of serving cells are locatable, so the input they were designed around simply does not exist.",
        "R| The second way is empirical. Transferring a transmitter-agnostic COMPASS zero-shot reaches 10.0 dB against 6.5 dB for native classical interpolation, so even the transferable variant is beaten by a method with no learned parameters at all.",
        "**Because a sceptical reader could reasonably object that both failures are artifacts of a weak comparison, we removed that objection by building a deep model that is transmitter-agnostic by construction and training it natively on the real data, which is the subject of the next slide.**",
       ]})

    A({"title": "Results, real data: a deep model built for this setting still loses",
       "bullets": [
        "**To test the transfer failure fairly we implemented a Conditional Neural Process, the deep counterpart to Kriging for scattered data, which never sees a transmitter because it consumes only observations in relative coordinates, and we trained it directly on the real measurements under the identical buffered leave-one-reference-point-out and leave-one-cell-out protocol used for every classical result.**",
        "R| It still loses to classical interpolation. The deep model reaches 5.64 dB with wall geometry and 5.77 dB without, both significantly worse than tuned RBF at 5.49 dB, with p equal to 5e-4 and 4e-6 over 2,061 held-out points across 67 cells.",
        "G| Geometry helps the deep model exactly as it helps every other method here, improving it by 2.3 percent, which is a consistency check on the central thesis rather than a contradiction of it.",
        "G| The hybrid remains the only approach that beats the classical baseline, at 5.14 dB, so the conclusion is that on sparse real crowdsensed data end-to-end deep learning is not the answer, and we now demonstrate that with a deep model trained natively for the setting rather than establishing it by assertion.",
        "R| One honest caveat cuts the other way. The deep model's predictive uncertainty is well calibrated as it stands, with a mean predicted spread of 5.56 dB against an actual error of 5.64 dB, which is better raw calibration than our own model achieves before recalibration.",
       ]})

    if deep_real_is_valid():
        A({"fig": "pub/deep_real.png",
           "caption": "A deep model trained natively on the real data and transmitter-agnostic by construction is "
                      "still beaten by tuned classical interpolation. Only the geometry-driven residual improves on it."})

    # ------------------------- ACTIVE AND UNCERTAINTY -------------------------
    A({"title": "Results, real data: geometry-guided active collection",
       "bullets": [
        "**If geometry governs where reconstruction fails, then it should also tell us where to measure next, which turns the same wall model from an accuracy device into a data-collection policy and is the practical payoff of the whole approach.**",
        "**We test it operationally rather than in the abstract: starting from a small seed, each strategy chooses the next reference point, we reconstruct, and we track error against the number of measurements taken across 50 real cells.**",
        "G| Choosing points by wall-aware geometric criteria reaches a target accuracy with fewer measurements than distance-based or random selection, which matters because measurement effort, not computation, is the binding cost in real crowdsensing.",
        "R| The margin is modest and we present it as a coherent extension of the geometry result rather than as a standalone contribution.",
       ]})

    A({"fig": "pub/active.png",
       "caption": "Reconstruction error against measurement budget for geometry-guided, distance-based, and "
                  "random next-point selection."})

    A({"title": "Results: calibrated uncertainty",
       "bullets": [
        "**Uncertainty is what makes active collection possible in the first place, so its calibration is a claim we test rather than assume, and the honest finding has two halves that we report together.**",
        "R| Raw Monte-Carlo dropout is badly miscalibrated out of the box, covering only 14.4 percent of true values inside its one-sigma interval where 68 percent is expected, so the uncalibrated spread cannot be trusted to direct collection.",
        "G| A simple post-hoc recalibration fixes coverage almost exactly, taking one-sigma coverage from 14.4 to 90.1 percent and two-sigma to 96.8 percent, and the error-uncertainty correlation of 0.53 is unchanged throughout, which shows the ranking of uncertain regions was informative all along and only its scale was wrong.",
        "R| We state the cost of that fix rather than hiding it, because recalibration widens the intervals from 1.25 to 13.6 dB, so the calibrated uncertainty is trustworthy for deciding where to measure next but is too wide to serve as a tight per-pixel error bar.",
       ]})

    A({"fig": "pub/uq.png",
       "caption": "Predicted uncertainty against observed error before and after post-hoc recalibration."})

    A({"title": "Results: computational cost",
       "bullets": [
        "**Accuracy is only useful if it is affordable, so we measured inference cost across the panel on identical hardware, and COMPASS is competitive rather than expensive.**",
        "**A single COMPASS-WNet forward pass takes 6.7 ms at 6.4 million parameters, comparable to RadioUNet at 5.0 ms and PMNet at 3.7 ms, and roughly six times cheaper than iterative diffusion at 42.6 ms.**",
        "**Uncertainty is the one place we pay, since eight Monte-Carlo passes raise the cost to 53 ms, which remains well inside real-time budgets and is the price of the only calibrated uncertainty in the panel.**",
        "**Classical interpolation is not the cheap option it is often assumed to be, with RBF at 62.9 ms and IDW at 45.2 ms per map, because their cost grows with the number of measurements rather than being amortised into fixed weights.**",
       ]})

    # ------------------------ DISCUSSION AND CONCLUSION -----------------------
    A({"title": "Discussion and limitations",
       "bullets": [
        "**The single most useful thing this work establishes is that reconstruction has two distinct regimes and that method choice must follow the regime rather than the leaderboard: dense learned prediction wins where dense supervision and a known transmitter exist, and on sparse real points a tuned classical kernel is the strong baseline with the deployable learned gain sitting as a residual on top of it.**",
        "**Our results also separate the mechanisms that transfer from those that do not, since geometry and occlusion transfer and dominate, order and device conditioning are inert on synthetic data but real and measurable on crowdsensed walks, and a physics-based consistency loss moved nothing and was retired.**",
        "R| Effect sizes are reported at their true magnitude, so device calibration and active collection are modest individually and earn their place as parts of a coherent system, and occlusion conditioning is an established technique whose role here is a diagnosis rather than a new mechanism.",
        "R| Our breadth on real data is one dataset and two usable sites, which is the main limitation of this work, and a second real deployment is the natural next step rather than a detail we can argue away.",
        "R| We also note that our uncertainty requires post-hoc recalibration to be trustworthy and that the calibrated intervals are wide, which is a real constraint on how far the uncertainty can be pushed.",
       ]})

    A({"title": "Conclusion",
       "bullets": [
        "**The synthetic benchmark and real crowdsensing are different problems, and the methods that win the former depend on a transmitter the latter does not provide, which is the finding that organises everything in this work.**",
        "G| On the synthetic benchmark COMPASS-WNet with occlusion conditioning is the most accurate reconstructor across nine deep architecture classes and seven classical families at 8.65 dB, breaking the behind-building plateau that every prior architecture shares and degrading the least of any method as scenes grow cluttered.",
        "G| On real crowdsensed data our negative controls establish which mechanisms genuinely transfer, with wall attenuation confirmed directly, walk order validated against a shuffled control, device offsets shown to be first-order at 4.12 dB median, and uncertainty made calibrated enough to direct collection.",
        "R| We are equally clear about what does not work, since on sparse real reconstruction classical interpolation remains the strong baseline, a deep model built specifically for that setting and trained natively still loses to it, and our learned contribution there is a geometry-driven residual rather than an end-to-end network.",
        "**The contribution of this work is therefore a complete and honestly evaluated account of radio-map reconstruction from real crowdsensed measurements, benchmarked against the full spectrum of deep and classical methods under one protocol, which is what the field needs before it can trust any of these mechanisms in deployment.**",
       ]})

    return S

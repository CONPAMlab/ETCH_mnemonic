# Ego-NVS-Memory: Predictive Memory in Egocentric Video

## CVPR/ICCV-Style Abstract

Human memory is shaped by the statistics of visual experience, yet current computer-vision benchmarks rarely test whether video models learn representations that predict what people remember, forget, or distort. We propose **Ego-NVS-Memory**, an object-centric benchmark for linking natural egocentric visual statistics, predictive coding models, and human mnemonic behavior. First, we characterize the temporal structure of object features in large-scale first-person video using modern detection, tracking, segmentation, motion, saliency, and embedding models. Second, we train a self-supervised predictive neural network to anticipate object trajectories, appearances, persistence, and future visual states from natural video. We test whether the model's prediction errors, uncertainty, and learned object dynamics spontaneously organize into two psychologically meaningful dimensions: **mnemonic precision**, reflecting stable and low-uncertainty feature representations, and **mnemonic strength**, reflecting distinctiveness, salience, and distance from environmental priors. Finally, we evaluate these model-derived variables against human memory behavior on matched natural video clips. This work reframes predictive video modeling as a mechanistic account of memory, introducing a new human-aligned evaluation axis for computer vision: whether models can predict not only future frames, but the structure of human mnemonic representation.

## Research Question

How do the temporal statistics of natural egocentric vision give rise to separable dimensions of human memory precision and strength, and can an object-centric predictive coding network learn this structure without direct memory supervision?

## Central Innovation

We propose **memory as a downstream test of predictive vision**. Instead of evaluating a video model only by next-frame, tracking, or recognition accuracy, Ego-NVS-Memory evaluates whether the model learns the environmental structure that determines what humans remember precisely, remember strongly, forget, or bias toward priors.

## Implementable D.4.2 Model Plan

The implemented training scaffold treats each tracked object as a temporal token sequence. For each object, D.4.1 features are converted into normalized state vectors containing location, area, color, saliency, motion, temporal drift, and prediction-error proxies. The D.4.2 model then learns to predict future object states from prior states.

Current scaffold:

- `etch train-predictive`: trains a self-supervised object-state prediction model.
- `ObjectPredictiveCodingNet`: a GRU-based uncertainty-aware predictor.
- Outputs:
  - `object_predictive_coding.pt`
  - `training_history.json`
  - `predictive_training_report.json`
  - `predictive_memory_alignment.parquet`

The alignment table exposes model-derived variables:

- `predicted_precision`: high when prediction error and uncertainty are low.
- `predicted_strength`: high when prediction surprise and normalized object-state magnitude are high.
- `prediction_surprise`: realized error scaled by predicted uncertainty.
- `mean_uncertainty`: model uncertainty over future object state.
- `latent_norm`: strength of the learned predictive representation.

## Human Benchmark Link

For CVPR/ICCV framing, D.4.2 should include a compact behavioral benchmark rather than relying only on model-internal variables. Each human trial should map onto the same object tokens used by the predictive model.

Recommended trial families:

- Object localization recall for precision.
- Old/new or forced-choice recognition for strength.
- Confidence ratings for subjective strength.
- Temporal order or feature-change judgments for predictive stability.
- Optional gaze or click attention as an attention mediator.

The decisive test is whether predictive coding variables explain human precision and strength beyond object size, saliency, motion, semantic novelty, and category frequency.

## Paper Contributions

1. **Ego-NVS**: object-centric natural vision statistics for egocentric video.
2. **Ego-NVS-Memory**: a human-aligned benchmark linking video statistics to memory precision and strength.
3. **Predictive Memory Model**: self-supervised prediction over object tokens with uncertainty-aware outputs.
4. **Model-human alignment**: trial-level predictions of what people remember precisely, remember strongly, forget, or systematically misremember.


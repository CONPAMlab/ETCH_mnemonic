# ETCH Mnemonic: Ego-NVS Natural Vision Statistics

This repository implements the computational backbone for Aim 4 of the ETCH mnemonic project: a reproducible, object-centric benchmark for natural egocentric vision statistics and a Study 4B scaffold for predictive-coding memory models.

The working target is CVPR-level: robust Ego4D-scale extraction, explicit data provenance, run-level validation, and psychology-aligned feature targets for mnemonic precision and strength.

The paper-facing research question is: **Can predictive coding over object-centric egocentric video explain not only what machines can anticipate, but what humans remember precisely, remember strongly, and systematically misremember?**

## Research Alignment

Study D.4.1 characterizes natural vision statistics in first-person video. The pipeline extracts object identity, location, persistence, feature strength, temporal drift, saliency, motion, and predictability statistics from Ego4D.

Study D.4.2 uses those statistics to prepare predictive-coding targets. Precision is operationalized as low temporal and prediction error. Strength is operationalized as distance from the environmental norm, including saturation, size, rarity, and embedding novelty. This creates a bridge between Efficient Temporal Coding and memory behavior.

The abstract-style CVPR/ICCV proposal is in [docs/aim4_cvpr_abstract.md](docs/aim4_cvpr_abstract.md).

## Data Policy

Ego4D source videos are treated as read-only. Do not write outputs under:

```text
/Volumes/HD_cv/ego4d_full_data/v2
```

All generated files go to `runs/<run_id>/` or another configured output root. Videos, weights, caches, generated tables, and local runs are ignored by git.

## Install

```bash
conda env create -f environment.yml
conda activate etch-nvs
pip install -e .
```

For CUDA/HPC extraction, install PyTorch for the target CUDA runtime if the pinned CPU wheels are not appropriate for the node image.

## Quick Start

Smoke-test one Ego4D video for five seconds:

```bash
etch extract --config configs/ego4d.yaml
etch validate --run-dir runs/ego_nvs_smoke
etch summarize --run-dir runs/ego_nvs_smoke
etch model-predictive --features runs/ego_nvs_smoke
etch train-predictive --config configs/predictive_model.yaml
```

For full CUDA/HPC extraction, start from:

```bash
etch extract --config configs/ego4d_hpc.yaml
```

`main.py` is kept only as a compatibility wrapper:

```bash
python main.py extract --config configs/ego4d.yaml
```

## Configuration

`configs/ego4d.yaml` is the local smoke-test profile. `configs/ego4d_hpc.yaml` is the production profile with full-video sampling and premium module declarations. Both configs control:

- dataset root, video folder, manifest, and read-only policy
- output root, run id, resume behavior, and optional visual artifacts
- deterministic sample size, seed, time window, and frame stride
- detector/tracker backend, weights, device, confidence, IoU, and image size
- feature families: color, saliency, flow, segmentation, embeddings, open-vocabulary audit, and predictability
- HPC controls such as workers, shard size, and flush cadence

The default detector/tracker path uses Ultralytics YOLO11 with BoT-SORT. ByteTrack can be selected by setting `model.tracker: bytetrack.yaml`. Premium modules such as SAM 2, Grounding DINO, CLIP, DINOv2, and RAFT are represented in the HPC config and output metadata so experiments can be staged cleanly; lightweight smoke tests use Farneback flow and skip heavyweight optional models. Until a dedicated RAFT worker is attached on the cluster image, RAFT requests are recorded in metadata and fall back to Farneback for extraction continuity.

## Outputs

Each run writes:

```text
runs/<run_id>/
├── metadata.json
├── video_manifest.parquet
├── features/
│   ├── object_tracks/<video_uid>.parquet
│   └── frame_stats/<video_uid>.parquet
├── qc/<video_uid>.json
├── qc/run_validation.json
├── summary.json
└── study4b_predictive_targets.parquet
```

Object tracks include geometry, track age, entry/exit placeholders, color statistics, Lab/HSV/RGB strength, saliency, optical-flow summaries, camera-relative motion, temporal drift, and predictability errors. Frame stats include object counts, track counts, mean motion, saliency, temporal autocorrelation, and feature predictability.

## D.4.2 Predictive Coding Model

`etch train-predictive` trains a self-supervised object-centric predictive coding model on D.4.1 object-track shards. Each tracked object becomes a temporal state sequence containing normalized position, size, color, saliency, motion, drift, and prediction-error features. The model predicts the next object state and per-feature uncertainty.

The training output includes:

```text
runs/<run_id>/predictive_model/
├── object_predictive_coding.pt
├── training_config.json
├── training_history.json
├── predictive_training_report.json
└── predictive_memory_alignment.parquet
```

The alignment table exposes model-derived memory variables:

- `predicted_precision`: high when prediction error and uncertainty are low.
- `predicted_strength`: high when prediction surprise and normalized object-state magnitude are high.
- `prediction_surprise`: realized next-state error scaled by model uncertainty.
- `mean_uncertainty`: uncertainty over future object state.
- `latent_norm`: strength of the learned predictive representation.

The intended behavioral benchmark is model-facing: each human trial should reference the same Ego4D clip/object tokens used by the predictive model, enabling trial-level tests of memory precision, strength, forgetting, and prior-driven bias.

## Validation

Run:

```bash
python -m unittest discover tests
```

The test suite covers box clamping, z-scoring, manifest filtering, schema checks, output-root safety, and Study 4B predictive target generation.

## Commit Hygiene

Use focused commits:

```text
chore: sync repo hygiene and ignore generated artifacts
refactor: split extraction pipeline into configurable modules
feat: add manifest-driven Ego4D extraction CLI
feat: add premium tracking flow and segmentation backends
feat: add natural vision statistics schemas and summaries
feat: add predictive-coding analysis scaffold for Study 4B
docs: polish README for D.4.1/D.4.2 workflow
test: add extraction smoke tests and schema validation
```

Do not commit Ego4D files, generated runs, model weights, annotated videos, crops, `.DS_Store`, or `__pycache__`.

## Citation Anchors

Core project anchors include Ego4D for first-person video, Ultralytics YOLO/BoT-SORT/ByteTrack for detection and tracking, SAM 2 for video segmentation, Grounding DINO for open-vocabulary detection audit, RAFT for optical flow, CLIP/DINOv2 for embedding-based typicality, and PredNet-style predictive coding for Study 4B.

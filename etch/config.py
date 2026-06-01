from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    video_dir: Path
    manifest: Path | None = None
    read_only: bool = True
    video_extension: str = ".mp4"


@dataclass(frozen=True)
class RunConfig:
    output_root: Path = Path("runs")
    run_id: str = "ego_nvs_debug"
    resume: bool = True
    write_annotated_video: bool = False
    write_crops: bool = False


@dataclass(frozen=True)
class SampleConfig:
    n_videos: int | None = 1
    seed: int = 123
    first_seconds: float | None = 5.0
    frame_stride: int = 1
    split: str | None = None
    scenario_contains: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    detector: str = "yolo"
    weights: str = "yolo11l.pt"
    device: str = "cuda"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    tracker: str = "botsort.yaml"
    imgsz: int = 1280
    half: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    color: bool = True
    saliency: bool = True
    flow: str = "raft"
    embeddings: list[str] = field(default_factory=lambda: ["clip", "dinov2"])
    segmentation: str | None = "sam2"
    open_vocab_audit: str | None = "grounding_dino"
    predictability_history: int = 5


@dataclass(frozen=True)
class HPCConfig:
    num_workers: int = 1
    shard_size: int = 1
    flush_every_n_frames: int = 200


@dataclass(frozen=True)
class EgoNVSConfig:
    dataset: DatasetConfig
    run: RunConfig = field(default_factory=RunConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)

    @property
    def run_dir(self) -> Path:
        return self.run.output_root / self.run.run_id


def _path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path) -> EgoNVSConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base = config_path.parent
    dataset_raw: dict[str, Any] = raw.get("dataset", {})
    root = _path(base, dataset_raw.get("root"))
    if root is None:
        raise ValueError("Config must define dataset.root")

    video_dir = _path(root, dataset_raw.get("video_dir", "full_scale"))
    manifest = _path(root, dataset_raw.get("manifest")) if dataset_raw.get("manifest") else None

    run_raw: dict[str, Any] = raw.get("run", {})
    sample_raw: dict[str, Any] = raw.get("sample", {})
    model_raw: dict[str, Any] = raw.get("model", {})
    features_raw: dict[str, Any] = raw.get("features", {})
    hpc_raw: dict[str, Any] = raw.get("hpc", {})

    dataset = DatasetConfig(
        root=root,
        video_dir=video_dir or root,
        manifest=manifest,
        read_only=bool(dataset_raw.get("read_only", True)),
        video_extension=str(dataset_raw.get("video_extension", ".mp4")),
    )
    run = RunConfig(
        output_root=_path(base, run_raw.get("output_root", "runs")) or Path("runs"),
        run_id=str(run_raw.get("run_id", "ego_nvs_debug")),
        resume=bool(run_raw.get("resume", True)),
        write_annotated_video=bool(run_raw.get("write_annotated_video", False)),
        write_crops=bool(run_raw.get("write_crops", False)),
    )
    sample = SampleConfig(**sample_raw)
    model = ModelConfig(**model_raw)
    features = FeatureConfig(**features_raw)
    hpc = HPCConfig(**hpc_raw)
    return EgoNVSConfig(dataset=dataset, run=run, sample=sample, model=model, features=features, hpc=hpc)


def config_to_dict(config: EgoNVSConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "__dataclass_fields__"):
            return {k: convert(getattr(value, k)) for k in value.__dataclass_fields__}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(config)

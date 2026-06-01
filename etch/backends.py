from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    cls_id: int
    cls_name: str
    track_id: int


class DetectorTracker:
    def track_frame(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class UltralyticsTracker(DetectorTracker):
    def __init__(self, config: ModelConfig):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.weights)

    def track_frame(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            imgsz=self.config.imgsz,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            if box.id is None:
                continue
            cls_id = int(box.cls[0].item()) if getattr(box.cls, "ndim", 0) > 0 else int(box.cls)
            conf = float(box.conf[0].item()) if getattr(box.conf, "ndim", 0) > 0 else float(box.conf)
            track_id = int(box.id[0].item()) if getattr(box.id, "ndim", 0) > 0 else int(box.id)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=conf,
                    cls_id=cls_id,
                    cls_name=str(names.get(cls_id, cls_id)),
                    track_id=track_id,
                )
            )
        return detections


class NullPremiumBackend:
    """Records planned premium modules without forcing heavyweight installs in smoke tests."""

    def __init__(self, name: str):
        self.name = name

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "status": "configured_optional"}


def make_tracker(config: ModelConfig) -> DetectorTracker:
    if config.detector != "yolo":
        raise ValueError(f"Unsupported detector backend: {config.detector}")
    return UltralyticsTracker(config)


def model_cache_dir(repo_root: Path) -> Path:
    path = repo_root / "models"
    path.mkdir(exist_ok=True)
    return path


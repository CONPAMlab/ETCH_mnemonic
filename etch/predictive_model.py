from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import read_tables, write_json, write_table


OBJECT_STATE_COLUMNS = [
    "cx_norm",
    "cy_norm",
    "area_rel",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_s",
    "mean_v",
    "speed_px_s",
    "saliency_score",
    "temporal_rgb_drift",
    "feature_pred_err",
]


@dataclass(frozen=True)
class PredictiveTrainingConfig:
    sequence_length: int = 8
    prediction_horizon: int = 1
    hidden_dim: int = 128
    latent_dim: int = 64
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    min_track_length: int = 12
    device: str = "cpu"
    seed: int = 123


class ObjectSequenceDataset(Dataset):
    def __init__(
        self,
        objects: pd.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        min_track_length: int,
    ):
        self.sequence_length = int(sequence_length)
        self.prediction_horizon = int(prediction_horizon)
        self.columns = OBJECT_STATE_COLUMNS
        self.windows: list[tuple[np.ndarray, np.ndarray, dict]] = []
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self._build(objects, min_track_length)

    def _build(self, objects: pd.DataFrame, min_track_length: int) -> None:
        missing = [c for c in self.columns if c not in objects.columns]
        if missing:
            raise ValueError(f"Missing object state columns for predictive training: {missing}")

        data = objects.copy()
        data[self.columns] = data[self.columns].apply(pd.to_numeric, errors="coerce")
        self.mean = data[self.columns].mean(skipna=True).fillna(0.0)
        self.std = data[self.columns].std(skipna=True).replace(0, 1.0).fillna(1.0)
        data[self.columns] = (data[self.columns].fillna(self.mean) - self.mean) / self.std

        group_cols = ["video_uid", "track_id"]
        for (video_uid, track_id), track in data.sort_values("frame").groupby(group_cols):
            if len(track) < max(min_track_length, self.sequence_length + self.prediction_horizon):
                continue
            values = track[self.columns].to_numpy(dtype=np.float32)
            frames = track["frame"].to_numpy()
            max_start = len(values) - self.sequence_length - self.prediction_horizon + 1
            for start in range(max_start):
                target_idx = start + self.sequence_length + self.prediction_horizon - 1
                x = values[start : start + self.sequence_length]
                y = values[target_idx]
                meta = {
                    "video_uid": str(video_uid),
                    "track_id": int(track_id),
                    "start_frame": int(frames[start]),
                    "target_frame": int(frames[target_idx]),
                }
                self.windows.append((x, y, meta))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x, y, _ = self.windows[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

    def metadata(self, idx: int) -> dict:
        return self.windows[idx][2]


class ObjectPredictiveCodingNet(nn.Module):
    """Uncertainty-aware object-state predictor for D.4.2.

    The model predicts the next normalized object state and a log-variance for
    each state dimension. Low predicted variance and low realized error become
    candidate precision variables; high residuals and rarity become strength or
    surprise variables.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(hidden_dim, latent_dim, batch_first=True)
        self.mean_head = nn.Linear(latent_dim, input_dim)
        self.logvar_head = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        _, z = self.recurrent(h)
        latent = z[-1]
        mean = self.mean_head(latent)
        logvar = torch.clamp(self.logvar_head(latent), min=-7.0, max=5.0)
        return mean, logvar, latent


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (target - mean).pow(2) / torch.exp(logvar)).mean()


def train_predictive_model(
    features: str | Path,
    output: str | Path,
    config: PredictiveTrainingConfig | None = None,
) -> dict:
    cfg = config or PredictiveTrainingConfig()
    _set_seed(cfg.seed)
    features_path = Path(features)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = _load_object_features(features_path)
    dataset = ObjectSequenceDataset(
        objects,
        sequence_length=cfg.sequence_length,
        prediction_horizon=cfg.prediction_horizon,
        min_track_length=cfg.min_track_length,
    )
    if len(dataset) == 0:
        raise RuntimeError("No eligible object-track windows for predictive training.")

    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = ObjectPredictiveCodingNet(
        input_dim=len(OBJECT_STATE_COLUMNS),
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred, logvar, _ = model(x)
            loss = gaussian_nll(pred, logvar, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    torch.save(
        {
            "model_state": model.state_dict(),
            "columns": OBJECT_STATE_COLUMNS,
            "normalization": {
                "mean": dataset.mean.to_dict() if dataset.mean is not None else {},
                "std": dataset.std.to_dict() if dataset.std is not None else {},
            },
            "config": asdict(cfg),
        },
        output_dir / "object_predictive_coding.pt",
    )
    write_json(output_dir / "training_config.json", asdict(cfg))
    write_json(output_dir / "training_history.json", {"history": history})
    alignment = export_memory_alignment_table(model, dataset, output_dir, device)
    report = {
        "n_windows": len(dataset),
        "n_features": len(OBJECT_STATE_COLUMNS),
        "final_loss": history[-1]["loss"],
        "device": str(device),
        "checkpoint": str(output_dir / "object_predictive_coding.pt"),
        "alignment_rows": alignment["n_rows"],
        "interpretation": (
            "Predicted precision is inverse uncertainty/error; predicted strength "
            "is prediction surprise plus feature distinctiveness."
        ),
    }
    write_json(output_dir / "predictive_training_report.json", report)
    return report


def export_memory_alignment_table(
    model: ObjectPredictiveCodingNet,
    dataset: ObjectSequenceDataset,
    output_dir: Path,
    device: torch.device,
) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            pred, logvar, latent = model(x.unsqueeze(0).to(device))
            pred = pred.squeeze(0).cpu().numpy()
            logvar = logvar.squeeze(0).cpu().numpy()
            target = y.numpy()
            abs_err = np.abs(target - pred)
            uncertainty = np.exp(logvar)
            precision = -float(np.mean(abs_err + uncertainty))
            surprise = float(np.mean(abs_err / (np.sqrt(uncertainty) + 1e-8)))
            strength = surprise + float(np.linalg.norm(target))
            row = dataset.metadata(idx)
            row.update(
                {
                    "predicted_precision": precision,
                    "predicted_strength": strength,
                    "prediction_surprise": surprise,
                    "mean_uncertainty": float(np.mean(uncertainty)),
                    "mean_abs_error": float(np.mean(abs_err)),
                    "latent_norm": float(torch.linalg.vector_norm(latent).cpu()),
                }
            )
            rows.append(row)
    write_table(output_dir / "predictive_memory_alignment.parquet", rows)
    return {"n_rows": len(rows)}


def _load_object_features(features_path: Path) -> pd.DataFrame:
    if features_path.is_dir():
        paths = sorted((features_path / "features" / "object_tracks").glob("*.parquet"))
        if not paths:
            paths = sorted(features_path.glob("*.parquet"))
        return read_tables(paths)
    if features_path.suffix == ".parquet":
        return read_tables([features_path])
    raise ValueError(f"Unsupported feature input: {features_path}")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


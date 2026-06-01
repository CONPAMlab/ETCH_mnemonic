from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .config import EgoNVSConfig


def discover_videos(config: EgoNVSConfig) -> pd.DataFrame:
    rows = _manifest_rows(config)
    rows = _filter_rows(rows, config)
    rows = rows.sort_values("video_uid").reset_index(drop=True)
    rows = _sample_rows(rows, config.sample.n_videos, config.sample.seed)
    return rows


def _manifest_rows(config: EgoNVSConfig) -> pd.DataFrame:
    ext = config.dataset.video_extension.lower()
    if config.dataset.manifest and config.dataset.manifest.exists():
        manifest = pd.read_csv(config.dataset.manifest)
        uid_col = "video_uid" if "video_uid" in manifest.columns else None
        rows = []
        for _, row in manifest.iterrows():
            video_uid = str(row[uid_col]) if uid_col else Path(str(row.get("path", ""))).stem
            candidate = config.dataset.video_dir / f"{video_uid}{ext}"
            if candidate.exists() and not candidate.name.startswith("._"):
                record = row.to_dict()
                record["video_uid"] = video_uid
                record["video_path"] = str(candidate)
                rows.append(record)
        return pd.DataFrame(rows)

    videos = [
        p
        for p in config.dataset.video_dir.rglob(f"*{ext}")
        if p.is_file() and not p.name.startswith("._")
    ]
    return pd.DataFrame(
        [{"video_uid": p.stem, "video_path": str(p)} for p in sorted(videos)]
    )


def _filter_rows(rows: pd.DataFrame, config: EgoNVSConfig) -> pd.DataFrame:
    if rows.empty:
        return rows
    filtered = rows
    split = config.sample.split
    if split:
        split_cols = [c for c in filtered.columns if c.startswith("split_")]
        if split_cols:
            mask = False
            for col in split_cols:
                mask = mask | (filtered[col].astype(str) == split)
            filtered = filtered[mask]
    scenario = config.sample.scenario_contains
    if scenario and "scenarios" in filtered.columns:
        filtered = filtered[
            filtered["scenarios"].astype(str).str.contains(scenario, case=False, na=False)
        ]
    return filtered


def _sample_rows(rows: pd.DataFrame, n_videos: int | None, seed: int) -> pd.DataFrame:
    if n_videos is None or n_videos <= 0 or len(rows) <= n_videos:
        return rows.reset_index(drop=True)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n_videos))
    return rows.iloc[indices].reset_index(drop=True)


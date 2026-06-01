from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_tables, write_json, write_table


def summarize_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    object_paths = sorted((run_dir / "features" / "object_tracks").glob("*.parquet"))
    frame_paths = sorted((run_dir / "features" / "frame_stats").glob("*.parquet"))
    objects = read_tables(object_paths)
    frames = read_tables(frame_paths)

    summary = {
        "run_dir": str(run_dir),
        "n_videos": int(frames["video_uid"].nunique()) if not frames.empty else 0,
        "n_frames": int(len(frames)),
        "n_object_rows": int(len(objects)),
        "mean_objects_per_frame": _safe_mean(frames, "n_objects"),
        "mean_area_rel": _safe_mean(objects, "area_rel"),
        "mean_speed_px_s": _safe_mean(objects, "speed_px_s"),
        "mean_saliency": _safe_mean(objects, "saliency_score"),
        "mean_feature_prediction_error": _safe_mean(objects, "feature_pred_err"),
    }
    if not objects.empty and "cls_name" in objects.columns:
        class_counts = (
            objects.groupby("cls_name")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="n_rows")
        )
        write_table(run_dir / "summary_class_counts.parquet", class_counts)
    write_json(run_dir / "summary.json", summary)
    return summary


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[col], errors="coerce")
    return float(np.nanmean(values)) if values.notna().any() else float("nan")


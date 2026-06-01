from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_tables, write_json, write_table


def build_predictive_scaffold(features: str | Path, output: str | Path | None = None) -> dict:
    """Create Study 4B proxy targets from D.4.1 feature shards.

    This is intentionally lightweight: it prepares precision/strength labels and
    baseline correlations before a heavier PredNet-style model is trained.
    """

    feature_path = Path(features)
    run_dir = feature_path if feature_path.is_dir() else feature_path.parent
    object_paths = sorted((feature_path / "features" / "object_tracks").glob("*.parquet"))
    if not object_paths and feature_path.suffix == ".parquet":
        object_paths = [feature_path]
    objects = read_tables(object_paths)
    if objects.empty:
        raise RuntimeError(f"No object feature rows found at {features}")

    proxy = objects[
        [
            "video_uid",
            "frame",
            "track_id",
            "cls_name",
            "feature_pred_err",
            "traj_pred_err_px",
            "temporal_rgb_drift",
            "area_rel",
            "mean_s",
            "novelty_score",
        ]
    ].copy()
    proxy["precision_proxy"] = -_z(
        proxy[["feature_pred_err", "traj_pred_err_px", "temporal_rgb_drift"]].mean(axis=1)
    )
    proxy["strength_proxy"] = _z(
        proxy[["area_rel", "mean_s", "novelty_score"]].mean(axis=1)
    )
    proxy["efficient_temporal_coding_target"] = proxy["precision_proxy"] + proxy["strength_proxy"]

    out_dir = Path(output) if output else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(out_dir / "study4b_predictive_targets.parquet", proxy)
    report = {
        "n_rows": int(len(proxy)),
        "precision_proxy_mean": _mean(proxy["precision_proxy"]),
        "strength_proxy_mean": _mean(proxy["strength_proxy"]),
        "precision_strength_corr": _corr(proxy["precision_proxy"], proxy["strength_proxy"]),
        "interpretation": (
            "Precision is operationalized as low temporal/prediction error; "
            "strength is operationalized as feature distinctiveness and visual saturation."
        ),
    }
    write_json(out_dir / "study4b_predictive_report.json", report)
    return report


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if not np.isfinite(std) or std < 1e-8:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values.fillna(mean) - mean) / std


def _mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean(skipna=True))


def _corr(a: pd.Series, b: pd.Series) -> float:
    corr = pd.concat([a, b], axis=1).corr().iloc[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .backends import Detection, make_tracker
from .config import EgoNVSConfig, config_to_dict
from .features import (
    clamp_box,
    compute_center_rgb,
    compute_color_stats,
    compute_contrast,
    compute_farneback_flow,
    compute_orientation_deg,
    flow_divergence,
    mean_in_box,
    safe_zscore,
    temporal_autocorr,
)
from .io import assert_output_not_inside_dataset, ensure_run_dirs, utc_now, write_json, write_table
from .manifest import discover_videos
from .schema import FRAME_STAT_COLUMNS, OBJECT_TRACK_COLUMNS


def run_extraction(config: EgoNVSConfig) -> Path:
    assert_output_not_inside_dataset(config.run.output_root, config.dataset.root)
    ensure_run_dirs(config.run_dir)

    manifest = discover_videos(config)
    write_table(config.run_dir / "video_manifest.parquet", manifest)
    write_json(
        config.run_dir / "metadata.json",
        {
            "created_at": utc_now(),
            "project": "Ego-NVS / ETCH mnemonic Aim 4",
            "study_alignment": {
                "D.4.1": "AI-powered characterization of natural vision statistics",
                "D.4.2": "Predictive-coding scaffold for memory modeling",
            },
            "config": config_to_dict(config),
            "n_videos": int(len(manifest)),
            "premium_modules": {
                "segmentation": config.features.segmentation,
                "open_vocab_audit": config.features.open_vocab_audit,
                "embedding_backends": config.features.embeddings,
                "flow": config.features.flow,
                "active_flow_backend": _active_flow_backend(config.features.flow),
            },
        },
    )

    if manifest.empty:
        raise RuntimeError("No videos matched the configured Ego4D manifest/sample policy.")

    for _, row in manifest.iterrows():
        process_video(row.to_dict(), config)

    return config.run_dir


def process_video(video_record: dict[str, Any], config: EgoNVSConfig) -> None:
    video_uid = str(video_record["video_uid"])
    video_path = Path(str(video_record["video_path"]))
    object_path = config.run_dir / "features" / "object_tracks" / f"{video_uid}.parquet"
    frame_path = config.run_dir / "features" / "frame_stats" / f"{video_uid}.parquet"
    qc_path = config.run_dir / "qc" / f"{video_uid}.json"

    if config.run.resume and object_path.exists() and frame_path.exists() and qc_path.exists():
        return

    qc = {
        "video_uid": video_uid,
        "status": "started",
        "frames_read": 0,
        "frames_written": 0,
        "objects_written": 0,
        "nan_rate_object_tracks": float("nan"),
        "track_fragmentation_proxy": float("nan"),
        "error": "",
    }

    try:
        object_rows, frame_rows, qc_updates = _extract_video_rows(video_uid, video_path, config)
        qc.update(qc_updates)
        write_table(object_path, pd.DataFrame(object_rows, columns=OBJECT_TRACK_COLUMNS))
        write_table(frame_path, pd.DataFrame(frame_rows, columns=FRAME_STAT_COLUMNS))
        qc["status"] = "complete"
        qc["objects_written"] = len(object_rows)
        qc["frames_written"] = len(frame_rows)
        qc["nan_rate_object_tracks"] = _nan_rate(pd.DataFrame(object_rows))
        qc["track_fragmentation_proxy"] = _track_fragmentation_proxy(object_rows)
    except Exception as exc:
        qc["status"] = "failed"
        qc["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(qc_path, qc)


def _extract_video_rows(
    video_uid: str,
    video_path: Path,
    config: EgoNVSConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = _max_frames(fps, total_frames, config.sample.first_seconds)
    frame_stride = max(1, int(config.sample.frame_stride))

    tracker = make_tracker(config.model)
    saliency = _make_saliency(width, height) if config.features.saliency else None
    prev_gray = None
    prev_state: dict[int, dict[str, float]] = {}
    rgb_hist = defaultdict(lambda: deque(maxlen=config.features.predictability_history))
    track_age = defaultdict(int)
    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    frames_read = 0

    pbar_total = max_frames if max_frames else total_frames if total_frames > 0 else None
    with tqdm(total=pbar_total, desc=video_uid) as pbar:
        frame_idx = 0
        while True:
            if max_frames and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = flow_mag = flow_ang = None
            if prev_gray is not None and config.features.flow not in {"none", None}:
                flow, flow_mag, flow_ang = compute_farneback_flow(prev_gray, gray)
            sal_map = _saliency_map(saliency, frame)
            detections = tracker.track_frame(frame)
            rows_this_frame = _object_rows_for_frame(
                detections=detections,
                frame=frame,
                gray=gray,
                flow=flow,
                flow_mag=flow_mag,
                flow_ang=flow_ang,
                sal_map=sal_map,
                prev_state=prev_state,
                rgb_hist=rgb_hist,
                track_age=track_age,
                video_uid=video_uid,
                video_path=video_path,
                frame_idx=frame_idx,
                fps=fps,
                width=width,
                height=height,
            )
            object_rows.extend(rows_this_frame)
            frame_rows.append(
                _frame_row(
                    rows_this_frame,
                    video_uid,
                    video_path,
                    frame_idx,
                    fps,
                    prev_gray,
                    gray,
                    flow_mag,
                )
            )
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)

    cap.release()
    return object_rows, frame_rows, {"frames_read": frames_read}


def _object_rows_for_frame(
    detections: list[Detection],
    frame: np.ndarray,
    gray: np.ndarray,
    flow: np.ndarray | None,
    flow_mag: np.ndarray | None,
    flow_ang: np.ndarray | None,
    sal_map: np.ndarray | None,
    prev_state: dict[int, dict[str, float]],
    rgb_hist: dict[int, deque],
    track_age: dict[int, int],
    video_uid: str,
    video_path: Path,
    frame_idx: int,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    rows = []
    for det in detections:
        l, t, r, b = clamp_box(*det.xyxy, width, height)
        bw = float(r - l)
        bh = float(b - t)
        area_px = bw * bh
        area_rel = area_px / max(1, width * height)
        cx = float(l + bw / 2.0)
        cy = float(t + bh / 2.0)
        cx_norm = cx / max(1, width)
        cy_norm = cy / max(1, height)
        dist_center_norm = float(
            math.hypot(cx - width / 2.0, cy - height / 2.0)
            / (math.hypot(width / 2.0, height / 2.0) + 1e-8)
        )

        color = compute_color_stats(frame, l, t, r, b)
        center_r, center_g, center_b = compute_center_rgb(frame, cx, cy)
        rgb_hist[det.track_id].append((color["mean_r"], color["mean_g"], color["mean_b"]))
        temporal_rgb_drift = _temporal_rgb_drift(rgb_hist[det.track_id])
        contrast = compute_contrast(gray, l, t, r, b)
        orientation = compute_orientation_deg(gray, l, t, r, b)
        flow_mag_mean = mean_in_box(flow_mag, l, t, r, b) if flow_mag is not None else float("nan")
        flow_ang_mean = mean_in_box(flow_ang, l, t, r, b) if flow_ang is not None else float("nan")
        flow_div = flow_divergence(flow, l, t, r, b) if flow is not None else float("nan")
        camera_motion = float(np.nanmedian(flow_mag)) if flow_mag is not None else float("nan")
        relative_motion = (
            flow_mag_mean - camera_motion
            if np.isfinite(flow_mag_mean) and np.isfinite(camera_motion)
            else float("nan")
        )
        saliency_score = mean_in_box(sal_map, l, t, r, b) if sal_map is not None else float("nan")

        vx = vy = speed = direction = accel = pred_err = float("nan")
        feature_pred_err = float("nan")
        prev = prev_state.get(det.track_id)
        if prev and fps > 0:
            dt = max((frame_idx - prev["frame"]) / fps, 1e-8)
            vx = (cx - prev["cx"]) / dt
            vy = (cy - prev["cy"]) / dt
            speed = float(math.hypot(vx, vy))
            direction = float(math.degrees(math.atan2(vy, vx)))
            if np.isfinite(prev.get("vx", float("nan"))):
                accel = float(math.hypot(vx - prev["vx"], vy - prev["vy"]) / dt)
                pred_err = float(math.hypot(cx - (prev["cx"] + prev["vx"] * dt), cy - (prev["cy"] + prev["vy"] * dt)))
            feature_pred_err = _feature_prediction_error(color, prev)

        track_age[det.track_id] += 1
        entry_event = int(track_age[det.track_id] == 1)
        row = {
            "video_uid": video_uid,
            "video_path": str(video_path),
            "frame": frame_idx,
            "time_sec": frame_idx / fps if fps > 0 else float("nan"),
            "fps": fps,
            "track_id": det.track_id,
            "cls_id": det.cls_id,
            "cls_name": det.cls_name,
            "det_confidence": det.confidence,
            "x1": l,
            "y1": t,
            "x2": r,
            "y2": b,
            "box_w": bw,
            "box_h": bh,
            "area_px": area_px,
            "area_rel": area_rel,
            "cx": cx,
            "cy": cy,
            "cx_norm": cx_norm,
            "cy_norm": cy_norm,
            "dist_center_norm": dist_center_norm,
            **color,
            "center_r": center_r,
            "center_g": center_g,
            "center_b": center_b,
            "contrast_gray_std": contrast,
            "orientation_deg": orientation,
            "vx_px_s": vx,
            "vy_px_s": vy,
            "speed_px_s": speed,
            "dir_deg": direction,
            "accel_px_s2": accel,
            "flow_mag_mean_px_per_frame": flow_mag_mean,
            "flow_mag_mean_px_per_s": flow_mag_mean * fps if np.isfinite(flow_mag_mean) else float("nan"),
            "flow_ang_mean_deg": flow_ang_mean,
            "flow_divergence": flow_div,
            "camera_motion_px_per_frame": camera_motion,
            "object_relative_motion_px_per_frame": relative_motion,
            "mask_area_px": float("nan"),
            "mask_area_rel": float("nan"),
            "mask_eccentricity": float("nan"),
            "track_age_frames": track_age[det.track_id],
            "entry_event": entry_event,
            "exit_event": 0,
            "occluded": 0,
            "saliency_score": saliency_score,
            "temporal_rgb_drift": temporal_rgb_drift,
            "traj_pred_err_px": pred_err,
            "feature_pred_err": feature_pred_err,
            "embedding_clip_norm": float("nan"),
            "embedding_dinov2_norm": float("nan"),
            "typicality_score": float("nan"),
            "novelty_score": float("nan"),
        }
        rows.append(row)
        prev_state[det.track_id] = {
            "frame": float(frame_idx),
            "cx": cx,
            "cy": cy,
            "vx": vx,
            "vy": vy,
            "mean_r": color["mean_r"],
            "mean_g": color["mean_g"],
            "mean_b": color["mean_b"],
        }

    _add_within_frame_predictability(rows)
    return rows


def _frame_row(
    object_rows: list[dict[str, Any]],
    video_uid: str,
    video_path: Path,
    frame_idx: int,
    fps: float,
    prev_gray: np.ndarray | None,
    gray: np.ndarray,
    flow_mag: np.ndarray | None,
) -> dict[str, Any]:
    speeds = [r["speed_px_s"] for r in object_rows if np.isfinite(r["speed_px_s"])]
    saliency = [r["saliency_score"] for r in object_rows if np.isfinite(r["saliency_score"])]
    predictability = [r["feature_pred_err"] for r in object_rows if np.isfinite(r["feature_pred_err"])]
    return {
        "video_uid": video_uid,
        "video_path": str(video_path),
        "frame": frame_idx,
        "time_sec": frame_idx / fps if fps > 0 else float("nan"),
        "fps": fps,
        "n_objects": len(object_rows),
        "n_tracks_confirmed": len({r["track_id"] for r in object_rows}),
        "mean_area_rel": _mean([r["area_rel"] for r in object_rows]),
        "mean_confidence": _mean([r["det_confidence"] for r in object_rows]),
        "mean_speed_px_s": _mean(speeds),
        "mean_saliency": _mean(saliency),
        "scene_motion_mean": float(np.nanmean(flow_mag)) if flow_mag is not None else float("nan"),
        "scene_motion_std": float(np.nanstd(flow_mag)) if flow_mag is not None else float("nan"),
        "temporal_autocorr_gray": temporal_autocorr(prev_gray, gray),
        "feature_predictability_mean": _mean(predictability),
    }


def _max_frames(fps: float, total_frames: int, first_seconds: float | None) -> int | None:
    if first_seconds is None:
        return total_frames if total_frames > 0 else None
    if fps > 0 and np.isfinite(fps):
        limit = int(round(first_seconds * fps))
        return min(total_frames, limit) if total_frames > 0 else limit
    return total_frames if total_frames > 0 else None


def _active_flow_backend(requested: str | None) -> str:
    if requested in {None, "none"}:
        return "none"
    if requested == "raft":
        return "farneback_fallback_until_raft_worker_enabled"
    return str(requested)


def _make_saliency(width: int, height: int):
    try:
        from pySaliencyMap import pySaliencyMap

        return pySaliencyMap(width, height)
    except Exception:
        return None


def _saliency_map(saliency: Any, frame: np.ndarray) -> np.ndarray | None:
    if saliency is None:
        return None
    try:
        sal_map = saliency.SMGetSM(frame).astype(np.float32)
        return (sal_map - np.nanmin(sal_map)) / (np.nanmax(sal_map) - np.nanmin(sal_map) + 1e-8)
    except Exception:
        return None


def _temporal_rgb_drift(values: deque) -> float:
    if len(values) < 2:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))))


def _feature_prediction_error(color: dict[str, float], prev: dict[str, float]) -> float:
    vals = [
        color["mean_r"] - prev.get("mean_r", float("nan")),
        color["mean_g"] - prev.get("mean_g", float("nan")),
        color["mean_b"] - prev.get("mean_b", float("nan")),
    ]
    if not all(np.isfinite(vals)):
        return float("nan")
    return float(math.sqrt(sum(v * v for v in vals)))


def _add_within_frame_predictability(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    novelty = safe_zscore([r["feature_pred_err"] for r in rows])
    for row, z in zip(rows, novelty):
        row["novelty_score"] = float(z)
        row["typicality_score"] = float(-z)


def _mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _nan_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0.0
    return float(numeric.isna().sum().sum() / numeric.size)


def _track_fragmentation_proxy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    by_class: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_class[str(row["cls_name"])].add(int(row["track_id"]))
    return float(np.mean([len(v) for v in by_class.values()])) if by_class else 0.0

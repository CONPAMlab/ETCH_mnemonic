from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import os

from stats_utils import (
    to_gray, spatial_autocorr, spatial_autocorr_channels,
    temporal_autocorr, compute_optical_flow, flow_magnitude,
    motion_autocorr_from_flows, frame_entropy
)

# -----------------------------
# CONFIG
# -----------------------------
video_dir = "video"  # Folder containing all your .MOV files
output_dir = "results"  # Where to save CSVs
os.makedirs(output_dir, exist_ok=True)

# Collect all video files from the folder
video_files = [
    os.path.join(video_dir, f)
    for f in os.listdir(video_dir)
    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
]

print(f"Found {len(video_files)} video(s) in '{video_dir}':")
for v in video_files:
    print(" -", v)

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight YOLOv8 model

# -----------------------------
# MAIN LOOP
# -----------------------------
for video_path in video_files:
    print(f"\n Analyzing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Cannot open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, int(fps // 5))  # sample roughly 5 fps

    prev_gray = None
    prev_flow = None
    stats = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        frame_bgr = frame
        frame_gray = to_gray(frame_bgr)

        # Spatial autocorrelation
        spatial_center = spatial_autocorr(frame_gray)
        spatial_rgb = spatial_autocorr_channels(frame_bgr, color_space="RGB")

        # Temporal autocorrelation
        t_auto = temporal_autocorr(prev_gray, frame_gray)

        motion_corr = float("nan")
        mag_mean = float("nan")
        if prev_gray is not None:
            flow = compute_optical_flow(prev_gray, frame_gray)
            mag = flow_magnitude(flow)
            mag_mean = float(np.mean(mag))
            motion_corr = (
                motion_autocorr_from_flows(prev_flow, flow)
                if prev_flow is not None
                else float("nan")
            )
            prev_flow = flow
        else:
            prev_flow = None

        # Entropy
        ent = frame_entropy(frame_gray)

        # YOLO detection
        results = model(frame_bgr, verbose=False)[0]
        objs = results.boxes
        if objs is None or len(objs) == 0:
            count, mean_area, mean_conf = 0, 0.0, 0.0
        else:
            areas = (objs.xyxy[:, 2] - objs.xyxy[:, 0]) * (objs.xyxy[:, 3] - objs.xyxy[:, 1])
            count = int(len(objs))
            mean_area = float(np.mean(areas.cpu().numpy()))
            mean_conf = float(objs.conf.mean().item())

        stats.append({
            "frame_idx": frame_idx,
            "spatial_center": spatial_center,
            "spatial_R": spatial_rgb[0],
            "spatial_G": spatial_rgb[1],
            "spatial_B": spatial_rgb[2],
            "temporal_auto": t_auto,
            "motion_corr": motion_corr,
            "motion_mag_mean": mag_mean,
            "entropy": ent,
            "obj_count": count,
            "mean_area": mean_area,
            "mean_conf": mean_conf
        })

        prev_gray = frame_gray
        frame_idx += 1

    cap.release()

    # Save per-video CSV
    df = pd.DataFrame(stats)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_name = os.path.join(output_dir, f"{base_name}_stats.csv")
    df.to_csv(csv_name, index=False)
    print(f"Saved {csv_name} ({len(df)} sampled frames)")
    print(df.describe())

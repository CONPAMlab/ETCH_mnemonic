from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

from stats_utils import (
    to_gray, spatial_autocorr, spatial_autocorr_channels,
    temporal_autocorr, compute_optical_flow, flow_magnitude,
    motion_autocorr_from_flows, frame_entropy
)

video_path = "path_to_ego4d_video.mp4"
model = YOLO("yolov8n.pt")  # or chosen model version

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

prev_gray = None
prev_flow = None
stats = []
frame_idx = 0
sample_rate = max(1, int(fps // 5))  # Keeps runtime reasonable

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % sample_rate != 0:
        frame_idx += 1
        continue

    # keep BGR for functions that expect BGR
    frame_bgr = frame
    frame_gray = to_gray(frame_bgr)

    # low-level spatial / channel autocorr
    spatial_center = spatial_autocorr(frame_gray)
    spatial_rgb = spatial_autocorr_channels(frame_bgr, color_space="RGB")

    # temporal pixel autocorr
    t_auto = temporal_autocorr(prev_gray, frame_gray)

    # optical flow & motion autocorr
    motion_corr = float("nan")
    mag_mean = float("nan")
    if prev_gray is not None:
        flow = compute_optical_flow(prev_gray, frame_gray)
        mag = flow_magnitude(flow)
        mag_mean = float(np.mean(mag))
        motion_corr = motion_autocorr_from_flows(prev_flow, flow) if prev_flow is not None else float("nan")
        prev_flow = flow
    else:
        prev_flow = None

    # entropy
    ent = frame_entropy(frame_gray)

    # YOLO detection (model(frame) accepts BGR or RGB depending on wrapper; ultralytics uses BGR)
    results = model(frame_bgr, verbose=False)[0]
    objs = results.boxes
    if objs is None or len(objs) == 0:
        count, mean_area, mean_conf = 0, 0.0, 0.0
    else:
        areas = (objs.xyxy[:, 2] - objs.xyxy[:, 0]) * (objs.xyxy[:, 3] - objs.xyxy[:, 1])
        count = int(len(objs))
        mean_area = float(np.mean(areas.cpu()))
        mean_conf = float(np.mean(objs.conf.cpu()))

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

df = pd.DataFrame(stats)
df.to_csv("scene_statistics_with_motion.csv", index=False)
print(df.describe())

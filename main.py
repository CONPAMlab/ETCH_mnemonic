import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import urllib.request
from collections import defaultdict

# -----------------------------
#  Add project root to sys.path
# -----------------------------
repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# -----------------------------
#  External libraries
# -----------------------------
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# -----------------------------
# 1. Configuration + weights helper
# -----------------------------
def ensure_weights(path, url):
    """Download weight file automatically if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Download complete: {path}")


ensure_weights(
    "weights/yolo11m.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
)

CONFIG = {
    "model": {
        "weights": "weights/yolo11m.pt",
        "device": "cpu",          # "cpu", "mps", or "cuda"
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "input": {
        "videos_dir": "videos",
        "output_dir": "outputs",
        "crop_objects": True,
        "crop_size_limit": 1024,
    },
    "tracker": {
        "max_age": 30,
        "min_hits": 3,
        "max_iou_distance": 0.7,
    },
    "viz": {
        "box_alpha": 0.25,
        "trail_len": 30,
    },
    "features": {
        "compute_orientation": True,
        "compute_contrast": True,
        "min_crop_pixels": 8,   # skip feature calc if bbox too tiny
    },
    "saliency": {
        # Heuristic weights for object-centric saliency (fast, explainable)
        # You can tune these later or replace with saliency maps.
        "w_area": 0.35,
        "w_speed": 0.35,
        "w_center": 0.20,
        "w_contrast": 0.10,
        "epsilon": 1e-8
    }
}

# -----------------------------
# 2. Prepare folders
# -----------------------------
VIDEOS_DIR = CONFIG["input"]["videos_dir"]
OUTPUTS_DIR = CONFIG["input"]["output_dir"]
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CROP_DIR = os.path.join(OUTPUTS_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

# -----------------------------
# 3. Load Ultralytics YOLO model
# -----------------------------
print("Loading YOLOv11 model...")
model = YOLO(CONFIG["model"]["weights"])
CLASS_NAMES = model.names  # dict: class_id -> name
print("Model loaded.")

# -----------------------------
# 4. Color helper – per TRACK ID
# -----------------------------
def get_color(track_id):
    track_id = int(track_id)
    np.random.seed(track_id * 12345)
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))

# -----------------------------
# Feature helpers
# -----------------------------
def clamp_box(l, t, r, b, w, h):
    l = int(max(0, min(l, w - 1)))
    t = int(max(0, min(t, h - 1)))
    r = int(max(0, min(r, w)))
    b = int(max(0, min(b, h)))
    if r <= l:
        r = min(w, l + 1)
    if b <= t:
        b = min(h, t + 1)
    return l, t, r, b

def crop_region(frame_bgr, l, t, r, b):
    return frame_bgr[t:b, l:r]

def compute_color_stats(frame_bgr, l, t, r, b):
    crop = crop_region(frame_bgr, l, t, r, b)
    if crop.size == 0:
        return dict(mean_r=np.nan, mean_g=np.nan, mean_b=np.nan,
                    mean_h=np.nan, mean_s=np.nan, mean_v=np.nan)

    mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
    return dict(mean_r=float(mean_r), mean_g=float(mean_g), mean_b=float(mean_b),
                mean_h=float(mean_h), mean_s=float(mean_s), mean_v=float(mean_v))

def compute_contrast(frame_gray, l, t, r, b):
    # simple luminance contrast proxy: std of grayscale patch
    patch = frame_gray[t:b, l:r]
    if patch.size == 0:
        return np.nan
    return float(np.std(patch))

def compute_orientation_deg(frame_gray, l, t, r, b):
    # gradient-based dominant orientation (weighted circular mean)
    patch = frame_gray[t:b, l:r]
    if patch.size == 0:
        return np.nan

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if float(np.sum(mag)) == 0.0:
        return np.nan

    angles = np.arctan2(gy, gx)  # radians
    # circular mean weighted by gradient magnitude
    sin_sum = float(np.sum(np.sin(angles) * mag))
    cos_sum = float(np.sum(np.cos(angles) * mag))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    return float(np.degrees(mean_angle))  # -180..180

def safe_zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 0.0
    if sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + eps)

def compute_object_saliency(area_rel, speed, dist_center_norm, contrast,
                            z_area, z_speed, z_center, z_contrast, weights):
    # Use z-scores by default (frame-relative). Center term should reward center proximity.
    # If z-scores are all 0 (e.g., only one object), falls back to raw terms.
    w_area = weights["w_area"]
    w_speed = weights["w_speed"]
    w_center = weights["w_center"]
    w_contrast = weights["w_contrast"]
    eps = weights["epsilon"]

    # Prefer z-scores; if everything is zero, mix in raw
    zsum = abs(float(z_area)) + abs(float(z_speed)) + abs(float(z_center)) + abs(float(z_contrast))
    if zsum < eps:
        # Raw fallback (rough normalization assumptions)
        # Center proximity: higher is better, so use (1 - dist_center_norm)
        return float(
            w_area * area_rel +
            w_speed * (speed if np.isfinite(speed) else 0.0) +
            w_center * (1.0 - dist_center_norm) +
            w_contrast * (contrast if np.isfinite(contrast) else 0.0)
        )

    return float(
        w_area * z_area +
        w_speed * z_speed +
        w_center * (-z_center) +   # because z_center is dist-to-center; smaller => more salient
        w_contrast * z_contrast
    )

# -----------------------------
# 5. Initialize DeepSORT tracker
# -----------------------------
tracker = DeepSort(
    max_age=CONFIG["tracker"]["max_age"],
    n_init=CONFIG["tracker"]["min_hits"],
    max_iou_distance=CONFIG["tracker"]["max_iou_distance"],
    embedder="mobilenet",
    half=True,
    bgr=True,
)

# -----------------------------
# 6. Process Videos
# -----------------------------
frame_summary_rows = []   # your existing summary CSV
object_rows = []          # NEW: per-object, per-frame table

for file in os.listdir(VIDEOS_DIR):
    if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    video_path = os.path.join(VIDEOS_DIR, file)
    print(f"\nProcessing video: {file}")

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(file)[0]}_annotated.mp4")
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        (width, height),
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)

    frame_idx = 0

    # store track center history per video (for trails)
    track_traces = defaultdict(list)  # track_id -> list of (cx, cy)

    # store previous kinematics per track (for speed/dir/accel)
    prev_state = {}  # track_id -> dict(frame_idx, cx, cy, vx, vy)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        #   YOLOv11 prediction
        # -----------------------------
        results = model.predict(
            frame,
            conf=CONFIG["model"]["conf_threshold"],
            iou=CONFIG["model"]["iou_threshold"],
            device=CONFIG["model"]["device"],
            verbose=False,
        )

        r = results[0]
        raw_dets = []   # list of [[x1,y1,x2,y2], conf, class_id]

        if len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item()) if box.conf.ndim > 0 else float(box.conf)
                cls = int(box.cls[0].item()) if box.cls.ndim > 0 else int(box.cls)
                raw_dets.append([[x1, y1, x2, y2], conf, cls])

        # -----------------------------
        #   Update DeepSORT tracker
        # -----------------------------
        tracks = tracker.update_tracks(raw_dets, frame=frame)

        # -----------------------------
        #   Visualization overlay
        # -----------------------------
        overlay = frame.copy()

        # Collect per-frame object feature vectors (so we can compute within-frame z-scores)
        perframe_records = []  # temp store dicts for confirmed tracks

        areas = []
        confs = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            l, t, r_, b_ = track.to_ltrb()
            l, t, r_, b_ = clamp_box(l, t, r_, b_, width, height)

            cls = track.det_class
            if cls is None:
                continue
            cls = int(cls)

            det_conf = float(track.det_conf) if track.det_conf is not None else np.nan

            # basic geometry
            bw = float(r_ - l)
            bh = float(b_ - t)
            area_px = float(bw * bh)
            area_rel = float(area_px / max(1, (width * height)))

            cx = float(l + bw / 2.0)
            cy = float(t + bh / 2.0)
            cx_norm = float(cx / max(1, width))
            cy_norm = float(cy / max(1, height))

            # distance to center (0..~0.707 when normalized by diagonal/2)
            dx_c = cx - (width / 2.0)
            dy_c = cy - (height / 2.0)
            dist_center = float(np.sqrt(dx_c**2 + dy_c**2))
            dist_center_norm = float(dist_center / (np.sqrt((width/2.0)**2 + (height/2.0)**2) + 1e-8))

            # motion from previous state
            speed_px_s = np.nan
            dir_deg = np.nan
            vx = np.nan
            vy = np.nan
            accel_px_s2 = np.nan

            if track_id in prev_state:
                prev = prev_state[track_id]
                dt_frames = frame_idx - prev["frame_idx"]
                if dt_frames > 0 and fps > 0:
                    dt = dt_frames / fps
                    dx = cx - prev["cx"]
                    dy = cy - prev["cy"]
                    vx = dx / dt
                    vy = dy / dt
                    speed_px_s = float(np.sqrt(vx**2 + vy**2))
                    dir_deg = float(np.degrees(np.arctan2(vy, vx)))

                    if np.isfinite(prev.get("vx", np.nan)) and np.isfinite(prev.get("vy", np.nan)):
                        dvx = vx - prev["vx"]
                        dvy = vy - prev["vy"]
                        accel_px_s2 = float(np.sqrt(dvx**2 + dvy**2) / max(dt, 1e-8))

            prev_state[track_id] = dict(
                frame_idx=frame_idx,
                cx=cx, cy=cy,
                vx=vx, vy=vy
            )

            # Feature extraction guard
            if bw < CONFIG["features"]["min_crop_pixels"] or bh < CONFIG["features"]["min_crop_pixels"]:
                color_stats = dict(mean_r=np.nan, mean_g=np.nan, mean_b=np.nan,
                                   mean_h=np.nan, mean_s=np.nan, mean_v=np.nan)
                contrast = np.nan
                orientation = np.nan
            else:
                color_stats = compute_color_stats(frame, l, t, r_, b_)
                contrast = compute_contrast(frame_gray, l, t, r_, b_) if CONFIG["features"]["compute_contrast"] else np.nan
                orientation = compute_orientation_deg(frame_gray, l, t, r_, b_) if CONFIG["features"]["compute_orientation"] else np.nan

            # store per-frame record (saliency will be computed after z-scoring)
            rec = {
                "video": file,
                "frame": frame_idx,
                "fps": float(fps),

                "track_id": track_id,
                "cls_id": cls,
                "cls_name": CLASS_NAMES.get(cls, str(cls)),
                "det_confidence": float(det_conf) if np.isfinite(det_conf) else np.nan,

                "x1": int(l), "y1": int(t), "x2": int(r_), "y2": int(b_),
                "box_w": bw, "box_h": bh,
                "area_px": area_px,
                "area_rel": area_rel,

                "cx": cx, "cy": cy,
                "cx_norm": cx_norm, "cy_norm": cy_norm,
                "dist_center_norm": dist_center_norm,

                "mean_r": color_stats["mean_r"],
                "mean_g": color_stats["mean_g"],
                "mean_b": color_stats["mean_b"],
                "mean_h": color_stats["mean_h"],
                "mean_s": color_stats["mean_s"],
                "mean_v": color_stats["mean_v"],

                "contrast_gray_std": float(contrast) if np.isfinite(contrast) else np.nan,
                "orientation_deg": float(orientation) if np.isfinite(orientation) else np.nan,

                "vx_px_s": float(vx) if np.isfinite(vx) else np.nan,
                "vy_px_s": float(vy) if np.isfinite(vy) else np.nan,
                "speed_px_s": float(speed_px_s) if np.isfinite(speed_px_s) else np.nan,
                "dir_deg": float(dir_deg) if np.isfinite(dir_deg) else np.nan,
                "accel_px_s2": float(accel_px_s2) if np.isfinite(accel_px_s2) else np.nan,
            }
            perframe_records.append(rec)

            # --- Visualization uses track color + label ---
            color = get_color(track_id)
            label = f"{rec['cls_name']} | ID {track_id}"

            # frame-level summary support
            areas.append(area_px)
            if np.isfinite(det_conf):
                confs.append(float(det_conf))

            # filled rectangle
            cv2.rectangle(overlay, (l, t), (r_, b_), color, thickness=-1)
            # border
            cv2.rectangle(frame, (l, t), (r_, b_), color, thickness=2)
            # label
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_bg_tl = (l, max(0, t - th - baseline - 4))
            text_bg_br = (l + tw + 4, t)
            cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
            cv2.putText(
                frame, label, (text_bg_tl[0] + 2, t - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            # save crop (use original unblended frame area)
            if CONFIG["input"]["crop_objects"]:
                crop = frame[t:b_, l:r_]
                if crop.size > 0:
                    crop_path = os.path.join(
                        CROP_DIR, f"{os.path.splitext(file)[0]}_ID{track_id}_F{frame_idx}.jpg"
                    )
                    cv2.imwrite(crop_path, crop)

            # update motion trails
            track_traces[track_id].append((int(cx), int(cy)))
            if len(track_traces[track_id]) > CONFIG["viz"]["trail_len"]:
                track_traces[track_id] = track_traces[track_id][-CONFIG["viz"]["trail_len"]:]

        # -----------------------------
        #   Compute within-frame saliency (object-centric)
        # -----------------------------
        if len(perframe_records) > 0:
            areas_rel = [rec["area_rel"] for rec in perframe_records]
            speeds = [rec["speed_px_s"] if np.isfinite(rec["speed_px_s"]) else 0.0 for rec in perframe_records]
            centers = [rec["dist_center_norm"] for rec in perframe_records]
            contrasts = [rec["contrast_gray_std"] if np.isfinite(rec["contrast_gray_std"]) else 0.0 for rec in perframe_records]

            z_area = safe_zscore(areas_rel)
            z_speed = safe_zscore(speeds)
            z_center = safe_zscore(centers)
            z_contrast = safe_zscore(contrasts)

            for i, rec in enumerate(perframe_records):
                sal = compute_object_saliency(
                    area_rel=rec["area_rel"],
                    speed=rec["speed_px_s"] if np.isfinite(rec["speed_px_s"]) else 0.0,
                    dist_center_norm=rec["dist_center_norm"],
                    contrast=rec["contrast_gray_std"],
                    z_area=z_area[i],
                    z_speed=z_speed[i],
                    z_center=z_center[i],
                    z_contrast=z_contrast[i],
                    weights=CONFIG["saliency"]
                )
                rec["z_area_rel"] = float(z_area[i])
                rec["z_speed"] = float(z_speed[i])
                rec["z_dist_center"] = float(z_center[i])
                rec["z_contrast"] = float(z_contrast[i])
                rec["saliency_score"] = float(sal)

            object_rows.extend(perframe_records)

        # apply transparency
        alpha = CONFIG["viz"]["box_alpha"]
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # draw trails
        for tid, pts in track_traces.items():
            if len(pts) < 2:
                continue
            color = get_color(tid)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, thickness=2)

        # -----------------------------
        #   Frame-level summary stats
        # -----------------------------
        n_objects = len(areas)
        mean_area = float(np.mean(areas)) if areas else 0.0
        mean_conf = float(np.mean(confs)) if confs else 0.0

        out.write(frame)

        frame_summary_rows.append(
            {
                "video": file,
                "frame": frame_idx,
                "n_objects": n_objects,
                "mean_area": mean_area,
                "mean_confidence": mean_conf,
            }
        )

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

# -----------------------------
# 7. Save Outputs
# -----------------------------
df_summary = pd.DataFrame(frame_summary_rows)
df_summary.to_csv(os.path.join(OUTPUTS_DIR, "detections_summary.csv"), index=False)

df_objects = pd.DataFrame(object_rows)

# Optional: enforce a stable column order (nice for downstream analysis)
preferred_cols = [
    "video", "frame", "fps",
    "track_id", "cls_id", "cls_name", "det_confidence",
    "x1", "y1", "x2", "y2", "box_w", "box_h",
    "area_px", "area_rel",
    "cx", "cy", "cx_norm", "cy_norm", "dist_center_norm",
    "mean_r", "mean_g", "mean_b",
    "mean_h", "mean_s", "mean_v",
    "contrast_gray_std", "orientation_deg",
    "vx_px_s", "vy_px_s", "speed_px_s", "dir_deg", "accel_px_s2",
    "z_area_rel", "z_speed", "z_dist_center", "z_contrast",
    "saliency_score",
]
cols = [c for c in preferred_cols if c in df_objects.columns] + [c for c in df_objects.columns if c not in preferred_cols]
df_objects = df_objects[cols]

df_objects.to_csv(os.path.join(OUTPUTS_DIR, "objects_tracks.csv"), index=False)

print("\nDone! All results saved.")
print(f"- Frame summary: {os.path.join(OUTPUTS_DIR, 'detections_summary.csv')}")
print(f"- Object tracks: {os.path.join(OUTPUTS_DIR, 'objects_tracks.csv')}")

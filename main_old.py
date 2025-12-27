import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import urllib.request

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


# Example: YOLO11m (more accurate than nano)
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
        "box_alpha": 0.25,        # transparency for filled boxes
        "trail_len": 30,          # how many past centers to keep per track
    },
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
    """
    Deterministic color per track ID.
    This makes each track have a persistent color over time.
    """
    track_id = int(track_id)
    np.random.seed(track_id * 12345)
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))


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
summary = []

for file in os.listdir(VIDEOS_DIR):
    if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    video_path = os.path.join(VIDEOS_DIR, file)
    print(f"\nProcessing video: {file}")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(file)[0]}_annotated.mp4")
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)

    frame_idx = 0

    # store track center history per video
    track_traces = {}  # track_id -> list of (cx, cy)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
        raw_dets = []   # list of [ [x1,y1,x2,y2], conf, class_id ]

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
        #   Visualization
        #   - Semi-transparent filled boxes
        #   - Colored by track ID
        #   - Label: class name + track ID
        #   - Motion trails
        # -----------------------------
        overlay = frame.copy()

        areas = []
        confs = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            l, t, r_, b_ = track.to_ltrb()
            l, t, r_, b_ = map(int, [l, t, r_, b_])

            cls = track.det_class
            if cls is None:
                continue
            cls = int(cls)

            color = get_color(track_id)
            label = f"{CLASS_NAMES.get(cls, str(cls))} | ID {track_id}"

            # stats
            areas.append((r_ - l) * (b_ - t))
            if track.det_conf is not None:
                confs.append(float(track.det_conf))

            # ---- filled rectangle on overlay ----
            cv2.rectangle(overlay, (l, t), (r_, b_), color, thickness=-1)

            # ---- border on main frame ----
            cv2.rectangle(frame, (l, t), (r_, b_), color, thickness=2)

            # ---- label background + text ----
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_bg_tl = (l, max(0, t - th - baseline - 4))
            text_bg_br = (l + tw + 4, t)

            cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
            cv2.putText(
                frame,
                label,
                (text_bg_tl[0] + 2, t - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # ---- save crop (from original frame) ----
            if CONFIG["input"]["crop_objects"]:
                x1c = max(0, l)
                y1c = max(0, t)
                x2c = min(width, r_)
                y2c = min(height, b_)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size > 0:
                    crop_path = os.path.join(
                        CROP_DIR, f"{os.path.splitext(file)[0]}_ID{track_id}_F{frame_idx}.jpg"
                    )
                    cv2.imwrite(crop_path, crop)

            # ---- update center traces for motion lines ----
            cx = int((l + r_) / 2)
            cy = int((t + b_) / 2)
            if track_id not in track_traces:
                track_traces[track_id] = []
            track_traces[track_id].append((cx, cy))

            # limit history length
            if len(track_traces[track_id]) > CONFIG["viz"]["trail_len"]:
                track_traces[track_id] = track_traces[track_id][-CONFIG["viz"]["trail_len"]:]

        # ---- apply transparency for filled boxes ----
        alpha = CONFIG["viz"]["box_alpha"]
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # ---- draw motion trails ----
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

        summary.append(
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
# 7. Save Summary
# -----------------------------
df = pd.DataFrame(summary)
df.to_csv(os.path.join(OUTPUTS_DIR, "detections_summary.csv"), index=False)

print("\nDone! All results saved.")

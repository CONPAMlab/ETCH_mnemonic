import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm

# -----------------------------
# 1. Load YOLOv10
# -----------------------------
from yolov10.models.common import DetectMultiBackend  # model loader
from yolov10.utils.torch_utils import select_device
from yolov10.utils.general import non_max_suppression, scale_boxes
from yolov10.utils.augmentations import letterbox

# Paths
VIDEOS_DIR = "videos"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Model setup
device = select_device('')
weights = 'yolov10n.pt'  # small model; change to yolov10s.pt or yolov10x.pt if desired
model = DetectMultiBackend(weights, device=device)
stride, names = model.stride, model.names

# -----------------------------
# 2. Process all videos
# -----------------------------
summary = []

for file in os.listdir(VIDEOS_DIR):
    if not file.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEOS_DIR, file)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(file)[0]}_annotated.mp4")

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing {file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img = letterbox(frame, 640, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, to 3xHxW
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
            n_objects = len(pred)
            mean_area = ((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])).mean().item()
            mean_conf = pred[:, 4].mean().item()

            # Draw boxes
            for *xyxy, conf, cls in pred:
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            n_objects, mean_area, mean_conf = 0, 0.0, 0.0

        # Write frame
        out.write(frame)
        summary.append({
            'video': file,
            'frame': frame_idx,
            'n_objects': n_objects,
            'mean_area': mean_area,
            'mean_confidence': mean_conf
        })

        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

# -----------------------------
# 3. Save summary stats
# -----------------------------
df = pd.DataFrame(summary)
df.to_csv(os.path.join(OUTPUTS_DIR, "detections_summary.csv"), index=False)
print(f"\n Done! Results saved to {OUTPUTS_DIR}")

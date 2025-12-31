import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
#  Add project root to sys.path
# -----------------------------
repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Itti-Koch saliency (vendor files from pySaliencyMap)
# Expected files: pySaliencyMap.py, pySaliencyMapDefs.py
from pySaliencyMap import pySaliencyMap


def ensure_weights(path, url):
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
        "device": "cpu",          # "cpu", "mps", "cuda"
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "input": {
        "videos_dir": "videos",
        "output_dir": "outputs",
        "save_annotated_video": True,
        "crop_objects": True,
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
    "scale": {
        "flush_every_n_frames": 200,     # incremental write frequency
        "resume_skip_done_videos": True, # skip videos already marked done
        "num_workers": 1,                # set >1 for multiprocess over videos
    },
    "optflow": {
        "method": "farneback",
        "fb_pyr_scale": 0.5,
        "fb_levels": 3,
        "fb_winsize": 15,
        "fb_iterations": 3,
        "fb_poly_n": 5,
        "fb_poly_sigma": 1.2,
        "fb_flags": 0,
    },
    "predictability": {
        "history_len": 12,      # frames kept for predictability metrics
    }
}


VIDEOS_DIR = CONFIG["input"]["videos_dir"]
OUTPUTS_DIR = CONFIG["input"]["output_dir"]
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CROP_DIR = os.path.join(OUTPUTS_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

DONE_DIR = os.path.join(OUTPUTS_DIR, "_done")
os.makedirs(DONE_DIR, exist_ok=True)

OBJECTS_CSV = os.path.join(OUTPUTS_DIR, "objects_tracks.csv")
EXTRA_CSV   = os.path.join(OUTPUTS_DIR, "objects_tracks_extra.csv")
SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "detections_summary.csv")


def get_color(track_id: int):
    track_id = int(track_id)
    np.random.seed(track_id * 12345)
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))


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


def compute_color_stats(frame_bgr, l, t, r, b):
    crop = frame_bgr[t:b, l:r]
    if crop.size == 0:
        return dict(mean_r=np.nan, mean_g=np.nan, mean_b=np.nan,
                    mean_h=np.nan, mean_s=np.nan, mean_v=np.nan)
    mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
    return dict(mean_r=float(mean_r), mean_g=float(mean_g), mean_b=float(mean_b),
                mean_h=float(mean_h), mean_s=float(mean_s), mean_v=float(mean_v))


def compute_contrast(frame_gray, l, t, r, b):
    patch = frame_gray[t:b, l:r]
    if patch.size == 0:
        return np.nan
    return float(np.std(patch))


def compute_orientation_deg(frame_gray, l, t, r, b):
    patch = frame_gray[t:b, l:r]
    if patch.size == 0:
        return np.nan
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if float(np.sum(mag)) == 0.0:
        return np.nan
    ang = np.arctan2(gy, gx)
    sin_sum = float(np.sum(np.sin(ang) * mag))
    cos_sum = float(np.sum(np.cos(ang) * mag))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    return float(np.degrees(mean_angle))


def compute_farneback_flow(prev_gray, gray):
    p = CONFIG["optflow"]
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        p["fb_pyr_scale"], p["fb_levels"], p["fb_winsize"], p["fb_iterations"],
        p["fb_poly_n"], p["fb_poly_sigma"], p["fb_flags"]
    )
    # flow[...,0]=vx, flow[...,1]=vy in px/frame
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return flow, mag, ang


def mean_in_box(arr2d, l, t, r, b):
    patch = arr2d[t:b, l:r]
    if patch.size == 0:
        return np.nan
    return float(np.mean(patch))


def append_rows(csv_path, rows, header_cols=None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(csv_path)
    if header_cols is not None:
        # ensure consistent ordering for the fixed-schema CSV
        df = df[header_cols]
    df.to_csv(csv_path, mode="a", header=(not file_exists), index=False)


def process_one_video(video_path: str):
    file = os.path.basename(video_path)
    done_flag = os.path.join(DONE_DIR, f"{os.path.splitext(file)[0]}.done")

    if CONFIG["scale"]["resume_skip_done_videos"] and os.path.exists(done_flag):
        print(f"Skip done video: {file}")
        return

    # Load model inside worker
    model = YOLO(CONFIG["model"]["weights"])
    CLASS_NAMES = model.names

    tracker = DeepSort(
        max_age=CONFIG["tracker"]["max_age"],
        n_init=CONFIG["tracker"]["min_hits"],
        max_iou_distance=CONFIG["tracker"]["max_iou_distance"],
        embedder="mobilenet",
        half=True,
        bgr=True,
    )

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Saliency model (Itti-Koch)
    sal = pySaliencyMap(width, height)

    # Optional annotated video
    out = None
    if CONFIG["input"]["save_annotated_video"]:
        out_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(file)[0]}_annotated.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))

    pbar = tqdm(total=total_frames, desc=file)

    # trails
    track_traces = defaultdict(list)

    # previous state per track for deltas + kinematics
    prev_state = {}  # track_id -> dict(frame_idx,cx,cy,vx,vy, area_rel, mean_r... orientation)
    # short history for predictability
    traj_hist = defaultdict(lambda: deque(maxlen=CONFIG["predictability"]["history_len"]))  # track_id -> [(frame,cx,cy)]

    prev_gray = None
    prev_flow_mag = None  # for debug

    # buffers for incremental writes
    objects_rows_buf = []
    extra_rows_buf = []
    summary_rows_buf = []

    frame_idx = 0

    # fixed schema column order for objects_tracks.csv
    objects_cols = [
        "video","frame","fps","track_id","cls_id","cls_name","det_confidence",
        "x1","y1","x2","y2","box_w","box_h","area_px","area_rel",
        "cx","cy","cx_norm","cy_norm","dist_center_norm",
        "mean_r","mean_g","mean_b","mean_h","mean_s","mean_v",
        "contrast_gray_std","orientation_deg",
        "vx_px_s","vy_px_s","speed_px_s","dir_deg","accel_px_s2",
        "z_area_rel","z_speed","z_dist_center","z_contrast",
        "saliency_score"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- saliency map (Itti-Koch) --------
        # pySaliencyMap expects BGR image (OpenCV default) and outputs saliency map
        sal_map = sal.SMGetSM(frame)  # 0..255-ish map (uint8/float depending on impl)
        sal_map = sal_map.astype(np.float32)
        sal_map_norm = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

        if not np.isfinite(sal_map_norm).all():
            # allow NaNs but make it explicit
            pass
        else:
            mn, mx = float(sal_map_norm.min()), float(sal_map_norm.max())
            if mn < -1e-6 or mx > 1.0 + 1e-6:
                raise ValueError(f"Saliency map out of [0,1]: min={mn}, max={mx}, frame={frame_idx}, video={file}")

        # -------- optical flow (between frames) --------
        flow_mag = None
        flow_ang = None
        if prev_gray is not None:
            _, flow_mag, flow_ang = compute_farneback_flow(prev_gray, gray)
        prev_gray = gray

        # -------- YOLO detections --------
        results = model.predict(
            frame,
            conf=CONFIG["model"]["conf_threshold"],
            iou=CONFIG["model"]["iou_threshold"],
            device=CONFIG["model"]["device"],
            verbose=False,
        )
        r = results[0]
        raw_dets = []
        if len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item()) if box.conf.ndim > 0 else float(box.conf)
                cls = int(box.cls[0].item()) if box.cls.ndim > 0 else int(box.cls)
                raw_dets.append([[x1, y1, x2, y2], conf, cls])

        tracks = tracker.update_tracks(raw_dets, frame=frame)

        overlay = frame.copy()
        areas = []
        confs = []

        # for frame-level z-scores: keep your columns stable, but z_* now used as within-frame normalizations
        # (we keep z_* columns; if you don’t want them, we can just set them to 0 without changing schema)
        perframe = []

        for trk in tracks:
            if not trk.is_confirmed():
                continue

            tid = int(trk.track_id)

            l, t, r_, b_ = trk.to_ltrb()
            l, t, r_, b_ = clamp_box(l, t, r_, b_, width, height)

            cls = trk.det_class
            if cls is None:
                continue
            cls = int(cls)

            det_conf = float(trk.det_conf) if trk.det_conf is not None else np.nan

            bw = float(r_ - l)
            bh = float(b_ - t)
            area_px = float(bw * bh)
            area_rel = float(area_px / max(1, width * height))

            cx = float(l + bw / 2.0)
            cy = float(t + bh / 2.0)
            cx_norm = float(cx / max(1, width))
            cy_norm = float(cy / max(1, height))

            # distance to center normalized (0..1)
            dx_c = cx - (width / 2.0)
            dy_c = cy - (height / 2.0)
            dist_center = float(np.sqrt(dx_c**2 + dy_c**2))
            dist_center_norm = float(dist_center / (np.sqrt((width/2.0)**2 + (height/2.0)**2) + 1e-8))

            # appearance
            color_stats = compute_color_stats(frame, l, t, r_, b_)
            contrast = compute_contrast(gray, l, t, r_, b_)
            orient = compute_orientation_deg(gray, l, t, r_, b_)

            # ---- saliency score from saliency map inside box ----
            sal_box_mean = float(np.nanmean(sal_map_norm[t:b_, l:r_])) if (b_ > t and r_ > l) else np.nan

            if np.isfinite(sal_box_mean) and sal_box_mean < -1e-6:
                raise ValueError(
                    f"Negative saliency_score={sal_box_mean} at frame={frame_idx}, track={tid}, video={file}")

            # ---- kinematics from center displacement ----
            vx = vy = speed = direction = accel = np.nan
            pred_err = np.nan
            occluded = False

            # occlusion flag: if DeepSORT track has time_since_update>0, it was predicted (no new det)
            tsu = getattr(trk, "time_since_update", None)
            if tsu is not None and tsu > 0:
                occluded = True
            elif trk.det_conf is None:
                occluded = True

            if tid in prev_state and fps > 0:
                prev = prev_state[tid]
                dt_frames = frame_idx - prev["frame_idx"]
                if dt_frames > 0:
                    dt = dt_frames / fps
                    dx = cx - prev["cx"]
                    dy = cy - prev["cy"]
                    vx = dx / dt
                    vy = dy / dt
                    speed = float(np.sqrt(vx**2 + vy**2))
                    direction = float(np.degrees(np.arctan2(vy, vx)))

                    if np.isfinite(prev.get("vx", np.nan)) and np.isfinite(prev.get("vy", np.nan)):
                        dvx = vx - prev["vx"]
                        dvy = vy - prev["vy"]
                        accel = float(np.sqrt(dvx**2 + dvy**2) / max(dt, 1e-8))

                    # predictability: constant-velocity prediction error (1-step)
                    # predict current from previous state
                    if np.isfinite(prev.get("vx", np.nan)) and np.isfinite(prev.get("vy", np.nan)):
                        cx_hat = prev["cx"] + prev["vx"] * dt
                        cy_hat = prev["cy"] + prev["vy"] * dt
                        pred_err = float(np.sqrt((cx - cx_hat)**2 + (cy - cy_hat)**2))

            # update predictability history
            traj_hist[tid].append((frame_idx, cx, cy))

            # prepare fixed-schema record
            rec = {
                "video": file,
                "frame": frame_idx,
                "fps": float(fps),
                "track_id": tid,
                "cls_id": cls,
                "cls_name": CLASS_NAMES.get(cls, str(cls)),
                "det_confidence": det_conf,

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

                "contrast_gray_std": contrast,
                "orientation_deg": orient,

                "vx_px_s": float(vx) if np.isfinite(vx) else np.nan,
                "vy_px_s": float(vy) if np.isfinite(vy) else np.nan,
                "speed_px_s": float(speed) if np.isfinite(speed) else np.nan,
                "dir_deg": float(direction) if np.isfinite(direction) else np.nan,
                "accel_px_s2": float(accel) if np.isfinite(accel) else np.nan,

                # keep z_* columns (computed below) without changing schema
                "z_area_rel": 0.0,
                "z_speed": 0.0,
                "z_dist_center": 0.0,
                "z_contrast": 0.0,

                # saliency from Itti-Koch map
                "saliency_score": float(sal_box_mean) if np.isfinite(sal_box_mean) else np.nan,
            }
            perframe.append(rec)

            # ----------------- EXTRA METRICS (new file, schema-free) -----------------
            # optical flow energy inside box: mean flow magnitude in px/frame, convert to px/s
            flow_mag_mean = np.nan
            flow_ang_mean = np.nan
            if flow_mag is not None:
                flow_mag_mean = mean_in_box(flow_mag, l, t, r_, b_)  # px/frame
                flow_ang_mean = mean_in_box(flow_ang, l, t, r_, b_)  # degrees

            # feature-change rates (per second)
            d_area_rel = d_orient = np.nan
            d_rgb = d_hsv = np.nan
            if tid in prev_state and fps > 0:
                prev = prev_state[tid]
                dt_frames = frame_idx - prev["frame_idx"]
                if dt_frames > 0:
                    dt = dt_frames / fps
                    d_area_rel = (area_rel - prev.get("area_rel", area_rel)) / max(dt, 1e-8)
                    # circular-ish: orientation is rough anyway; use simple diff
                    if np.isfinite(orient) and np.isfinite(prev.get("orientation_deg", np.nan)):
                        d_orient = (orient - prev["orientation_deg"]) / max(dt, 1e-8)

                    # color deltas: L2 of RGB & HSV mean diffs per second
                    pr = prev.get("mean_r", np.nan); pg = prev.get("mean_g", np.nan); pb = prev.get("mean_b", np.nan)
                    ph = prev.get("mean_h", np.nan); ps = prev.get("mean_s", np.nan); pv = prev.get("mean_v", np.nan)

                    if np.isfinite(pr) and np.isfinite(color_stats["mean_r"]):
                        d_rgb = float(np.sqrt((color_stats["mean_r"]-pr)**2 + (color_stats["mean_g"]-pg)**2 + (color_stats["mean_b"]-pb)**2) / max(dt, 1e-8))
                    if np.isfinite(ph) and np.isfinite(color_stats["mean_h"]):
                        d_hsv = float(np.sqrt((color_stats["mean_h"]-ph)**2 + (color_stats["mean_s"]-ps)**2 + (color_stats["mean_v"]-pv)**2) / max(dt, 1e-8))

            # disappear/reappear events (gap-based)
            # if we saw the track before but this frame is "occluded", note it; if occluded streak ends, note reappear
            prev_occluded = prev_state.get(tid, {}).get("occluded", False)
            event = ""
            if (not prev_occluded) and occluded:
                event = "disappear"
            elif prev_occluded and (not occluded):
                event = "reappear"

            extra_rows_buf.append({
                "video": file,
                "frame": frame_idx,
                "track_id": tid,
                "occluded": int(occluded),
                "occlusion_event": event,

                "flow_mag_mean_px_per_frame": flow_mag_mean,
                "flow_mag_mean_px_per_s": (flow_mag_mean * fps) if np.isfinite(flow_mag_mean) else np.nan,
                "flow_ang_mean_deg": flow_ang_mean,

                "d_area_rel_per_s": d_area_rel,
                "d_orientation_deg_per_s": d_orient,
                "d_rgb_L2_per_s": d_rgb,
                "d_hsv_L2_per_s": d_hsv,

                "traj_pred_err_px": pred_err,
            })

            # keep per-frame summary
            areas.append(area_px)
            if np.isfinite(det_conf):
                confs.append(det_conf)

            # viz
            color = get_color(tid)
            label = f"{rec['cls_name']} | ID {tid}"
            cv2.rectangle(overlay, (l, t), (r_, b_), color, thickness=-1)
            cv2.rectangle(frame, (l, t), (r_, b_), color, thickness=2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_bg_tl = (l, max(0, t - th - baseline - 4))
            text_bg_br = (l + tw + 4, t)
            cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
            cv2.putText(frame, label, (text_bg_tl[0] + 2, t - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if CONFIG["input"]["crop_objects"]:
                crop = frame[t:b_, l:r_]
                if crop.size > 0:
                    crop_path = os.path.join(CROP_DIR, f"{os.path.splitext(file)[0]}_ID{tid}_F{frame_idx}.jpg")
                    cv2.imwrite(crop_path, crop)

            track_traces[tid].append((int(cx), int(cy)))
            if len(track_traces[tid]) > CONFIG["viz"]["trail_len"]:
                track_traces[tid] = track_traces[tid][-CONFIG["viz"]["trail_len"]:]

            # update prev_state for next deltas
            prev_state[tid] = dict(
                frame_idx=frame_idx, cx=cx, cy=cy,
                vx=vx, vy=vy,
                area_rel=area_rel,
                mean_r=color_stats["mean_r"], mean_g=color_stats["mean_g"], mean_b=color_stats["mean_b"],
                mean_h=color_stats["mean_h"], mean_s=color_stats["mean_s"], mean_v=color_stats["mean_v"],
                orientation_deg=orient,
                occluded=occluded
            )

        # z-scores within frame (kept in existing columns)
        if perframe:
            area_rel_arr = np.array([r["area_rel"] for r in perframe], dtype=np.float32)
            speed_arr = np.array([0.0 if not np.isfinite(r["speed_px_s"]) else r["speed_px_s"] for r in perframe], dtype=np.float32)
            dist_arr = np.array([r["dist_center_norm"] for r in perframe], dtype=np.float32)
            con_arr = np.array([0.0 if not np.isfinite(r["contrast_gray_std"]) else r["contrast_gray_std"] for r in perframe], dtype=np.float32)

            def z(x):
                mu = float(np.mean(x)); sd = float(np.std(x))
                if sd < 1e-8:
                    return np.zeros_like(x)
                return (x - mu) / (sd + 1e-8)

            za = z(area_rel_arr); zs = z(speed_arr); zd = z(dist_arr); zc = z(con_arr)
            for i in range(len(perframe)):
                perframe[i]["z_area_rel"] = float(za[i])
                perframe[i]["z_speed"] = float(zs[i])
                perframe[i]["z_dist_center"] = float(zd[i])
                perframe[i]["z_contrast"] = float(zc[i])

        objects_rows_buf.extend(perframe)

        # frame-level summary
        n_objects = len(areas)
        mean_area = float(np.mean(areas)) if areas else 0.0
        mean_conf = float(np.mean(confs)) if confs else 0.0
        summary_rows_buf.append({
            "video": file,
            "frame": frame_idx,
            "n_objects": n_objects,
            "mean_area": mean_area,
            "mean_confidence": mean_conf,
        })

        # apply transparency + trails + write video
        if out is not None:
            alpha = CONFIG["viz"]["box_alpha"]
            frame_blend = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            for tid, pts in track_traces.items():
                if len(pts) < 2:
                    continue
                color = get_color(tid)
                for i in range(1, len(pts)):
                    cv2.line(frame_blend, pts[i - 1], pts[i], color, thickness=2)
            out.write(frame_blend)

        # incremental flush (Ego4D-scale)
        if (frame_idx + 1) % CONFIG["scale"]["flush_every_n_frames"] == 0:
            append_rows(OBJECTS_CSV, objects_rows_buf, header_cols=objects_cols)
            append_rows(EXTRA_CSV, extra_rows_buf, header_cols=None)
            append_rows(SUMMARY_CSV, summary_rows_buf, header_cols=None)
            objects_rows_buf.clear()
            extra_rows_buf.clear()
            summary_rows_buf.clear()

        frame_idx += 1
        pbar.update(1)

    # final flush
    append_rows(OBJECTS_CSV, objects_rows_buf, header_cols=objects_cols)
    append_rows(EXTRA_CSV, extra_rows_buf, header_cols=None)
    append_rows(SUMMARY_CSV, summary_rows_buf, header_cols=None)

    pbar.close()
    cap.release()
    if out is not None:
        out.release()

    # mark done
    with open(done_flag, "w") as f:
        f.write("done\n")

    print(f"Done: {file}")


def main():
    videos = []
    for f in os.listdir(VIDEOS_DIR):
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            videos.append(os.path.join(VIDEOS_DIR, f))
    videos.sort()

    nw = int(CONFIG["scale"]["num_workers"])
    if nw <= 1:
        for vp in videos:
            process_one_video(vp)
    else:
        # Multiprocess across videos (each process loads its own YOLO + DeepSORT + saliency)
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(process_one_video, vp) for vp in videos]
            for fut in as_completed(futs):
                # surface exceptions early
                fut.result()

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()

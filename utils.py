# utils.py
import os
import cv2
import json
import pandas as pd
from datetime import datetime
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_detections_csv(out_dir, detections_list):
    """
    detections_list: list of dicts:
      { 'frame': int, 'bbox':[x1,y1,x2,y2], 'conf':float, 'cls':int, 'label':str }
    """
    df = pd.DataFrame([{
        'frame': d['frame'],
        'x1': d['bbox'][0],
        'y1': d['bbox'][1],
        'x2': d['bbox'][2],
        'y2': d['bbox'][3],
        'conf': d['conf'],
        'cls': d.get('cls'),
        'label': d.get('label')
    } for d in detections_list])
    df.to_csv(os.path.join(out_dir, 'detections.csv'), index=False)

def save_tracks_csv(out_dir, tracks_list):
    """
    tracks_list: list of dicts:
      { 'frame':int, 'id':int, 'bbox':[x1,y1,x2,y2], 'cls':int, 'conf':float }
    """
    df = pd.DataFrame([{
        'frame': t['frame'],
        'id': t['id'],
        'x1': t['bbox'][0],
        'y1': t['bbox'][1],
        'x2': t['bbox'][2],
        'y2': t['bbox'][3],
        'conf': t.get('conf'),
        'cls': t.get('cls')
    } for t in tracks_list])
    df.to_csv(os.path.join(out_dir, 'tracks.csv'), index=False)

def save_stats(out_dir, stats):
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

def draw_boxes(frame, detections, tracks=None):
    """
    detections: list of dicts with bbox/conf/label
    tracks: list of dicts with id/bbox/cls
    """
    img = frame.copy()
    # draw detections (thin)
    for d in detections:
        x1,y1,x2,y2 = map(int, d['bbox'])
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,165,0), 1)
        text = f"{d.get('label',d.get('cls',''))} {d.get('conf',0):.2f}"
        cv2.putText(img, text, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,165,0), 1)

    # draw tracks (thicker)
    if tracks:
        for t in tracks:
            x1,y1,x2,y2 = map(int, t['bbox'])
            tid = t['id']
            color = ((tid*37) % 255, (tid*97) % 255, (tid*13) % 255)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, f"ID:{tid}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def crop_and_save(frame, bbox, out_path, max_dim=1024):
    x1,y1,x2,y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame[y1:y2, x1:x2]
    # resize if too big
    hh, ww = crop.shape[:2]
    scale = 1.0
    if max(hh, ww) > max_dim:
        scale = max_dim / max(hh, ww)
        crop = cv2.resize(crop, (int(ww*scale), int(hh*scale)))
    cv2.imwrite(out_path, crop)
    return True

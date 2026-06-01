from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def clamp_box(l: float, t: float, r: float, b: float, w: int, h: int) -> tuple[int, int, int, int]:
    l_f = float(np.clip(l, 0.0, float(max(0, w - 1))))
    t_f = float(np.clip(t, 0.0, float(max(0, h - 1))))
    r_f = float(np.clip(r, 0.0, float(max(0, w))))
    b_f = float(np.clip(b, 0.0, float(max(0, h))))
    l_i, t_i, r_i, b_i = int(l_f), int(t_f), int(r_f), int(b_f)
    if r_i <= l_i:
        r_i = min(w, l_i + 1)
    if b_i <= t_i:
        b_i = min(h, t_i + 1)
    return l_i, t_i, r_i, b_i


def safe_zscore(values: Iterable[float], eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    filled = arr.copy()
    mean = float(np.mean(filled[finite]))
    filled[~finite] = mean
    sd = float(np.std(filled))
    if sd < eps:
        return np.zeros_like(filled)
    return (filled - mean) / (sd + eps)


def mean_in_box(arr2d: np.ndarray, l: int, t: int, r: int, b: int) -> float:
    patch = arr2d[t:b, l:r]
    if patch.size == 0:
        return float("nan")
    return float(np.nanmean(patch))


def compute_color_stats(frame_bgr: np.ndarray, l: int, t: int, r: int, b: int) -> dict[str, float]:
    crop = frame_bgr[t:b, l:r]
    if crop.size == 0:
        return _nan_color_stats()
    mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    bb = lab[:, :, 2].astype(np.float32)
    spread_rms, spread_mean = compute_rgb_3d_spread(frame_bgr, l, t, r, b)
    return {
        "mean_r": float(mean_r),
        "mean_g": float(mean_g),
        "mean_b": float(mean_b),
        "mean_h": float(mean_h),
        "mean_s": float(mean_s),
        "mean_v": float(mean_v),
        "mean_L": float(np.mean(L)),
        "mean_a": float(np.mean(a)),
        "mean_b_lab": float(np.mean(bb)),
        "mean_a_c": float(np.mean(a) - 128.0),
        "mean_b_c": float(np.mean(bb) - 128.0),
        "std_L": float(np.std(L)),
        "std_a": float(np.std(a)),
        "std_b_lab": float(np.std(bb)),
        "rgb_spread_rms": spread_rms,
        "rgb_spread_mean": spread_mean,
    }


def compute_center_rgb(frame_bgr: np.ndarray, cx: float, cy: float) -> tuple[float, float, float]:
    h, w = frame_bgr.shape[:2]
    x = int(np.clip(round(cx), 0, w - 1))
    y = int(np.clip(round(cy), 0, h - 1))
    b, g, r = frame_bgr[y, x]
    return float(r), float(g), float(b)


def compute_rgb_3d_spread(frame_bgr: np.ndarray, l: int, t: int, r: int, b: int) -> tuple[float, float]:
    crop = frame_bgr[t:b, l:r]
    if crop.size == 0:
        return float("nan"), float("nan")
    rgb = crop[:, :, ::-1].astype(np.float32).reshape(-1, 3)
    mu = rgb.mean(axis=0, dtype=np.float32)
    dist = np.sqrt(np.sum((rgb - mu) ** 2, axis=1))
    return float(np.sqrt(np.mean(dist ** 2))), float(np.mean(dist))


def compute_contrast(frame_gray: np.ndarray, l: int, t: int, r: int, b: int) -> float:
    patch = frame_gray[t:b, l:r]
    if patch.size == 0:
        return float("nan")
    return float(np.std(patch))


def compute_orientation_deg(frame_gray: np.ndarray, l: int, t: int, r: int, b: int) -> float:
    patch = frame_gray[t:b, l:r]
    if patch.size == 0:
        return float("nan")
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if float(np.sum(mag)) == 0.0:
        return float("nan")
    ang = np.arctan2(gy, gx)
    return float(np.degrees(np.arctan2(np.sum(np.sin(ang) * mag), np.sum(np.cos(ang) * mag))))


def compute_farneback_flow(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_gray = np.ascontiguousarray(prev_gray)
    gray = np.ascontiguousarray(gray)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return flow, mag, ang


def flow_divergence(flow: np.ndarray, l: int, t: int, r: int, b: int) -> float:
    if flow.size == 0:
        return float("nan")
    fx = flow[..., 0]
    fy = flow[..., 1]
    dfx_dx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
    dfy_dy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
    return mean_in_box(dfx_dx + dfy_dy, l, t, r, b)


def temporal_autocorr(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    if prev_gray is None:
        return float("nan")
    a = prev_gray.astype(np.float32).reshape(-1)
    b = gray.astype(np.float32).reshape(-1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _nan_color_stats() -> dict[str, float]:
    keys = [
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_h",
        "mean_s",
        "mean_v",
        "mean_L",
        "mean_a",
        "mean_b_lab",
        "mean_a_c",
        "mean_b_c",
        "std_L",
        "std_a",
        "std_b_lab",
        "rgb_spread_rms",
        "rgb_spread_mean",
    ]
    return {k: float("nan") for k in keys}


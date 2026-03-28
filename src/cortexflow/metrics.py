from __future__ import annotations

import numpy as np


def corr_per_voxel(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y = y_true - y_true.mean(axis=0, keepdims=True)
    yh = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (y * yh).sum(axis=0)
    den = np.sqrt((y * y).sum(axis=0) * (yh * yh).sum(axis=0) + 1e-9)
    return (num / den).astype(np.float32)


def r2_per_voxel(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    resid = y_true - y_pred
    ss_res = (resid * resid).sum(axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (centered * centered).sum(axis=0) + 1e-9
    return (1.0 - ss_res / ss_tot).astype(np.float32)


def lateralization_index(scores: np.ndarray, hemi_labels: np.ndarray) -> float:
    left = scores[hemi_labels == 0]
    right = scores[hemi_labels == 1]
    l_mean = float(left.mean()) if left.size else 0.0
    r_mean = float(right.mean()) if right.size else 0.0
    denom = abs(l_mean) + abs(r_mean) + 1e-9
    return (l_mean - r_mean) / denom

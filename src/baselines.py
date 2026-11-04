"""Uniform scaling and center-crop baselines."""

from __future__ import annotations

import cv2
import numpy as np


def scale_to(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize to the exact target. Area interpolation when shrinking."""
    height, width = image.shape[:2]
    shrinking = target_h <= height and target_w <= width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def center_crop_resize(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Crop the centered window with the target aspect, then resize to it."""
    height, width = image.shape[:2]
    target_aspect = target_w / float(target_h)
    aspect = width / float(height)
    if aspect > target_aspect:
        new_w = max(1, int(round(height * target_aspect)))
        x0 = max(0, (width - new_w) // 2)
        cropped = image[:, x0:x0 + new_w]
    else:
        new_h = max(1, int(round(width / target_aspect)))
        y0 = max(0, (height - new_h) // 2)
        cropped = image[y0:y0 + new_h, :]
    if cropped.size == 0:
        cropped = image
    return scale_to(cropped, target_h, target_w)


def limit_long_side(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Shrink so the long side is at most ``max_side``. Never enlarges."""
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return np.ascontiguousarray(image)
    scale = max_side / float(long_side)
    new_w = max(2, int(round(width * scale)))
    new_h = max(2, int(round(height * scale)))
    return scale_to(image, new_h, new_w)

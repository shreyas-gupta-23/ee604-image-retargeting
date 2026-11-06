"""RGB overlays. We use these in the demo and in the experiment figures."""

from __future__ import annotations

import cv2
import numpy as np


def colorize(values: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` array to an RGB image with the viridis colormap."""
    scaled = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_vertical_seam(image: np.ndarray, seam: np.ndarray, color=(220, 40, 40)) -> np.ndarray:
    """Paint a vertical seam, two pixels wide, on a copy of ``image``."""
    painted = image.copy()
    height, width = painted.shape[:2]
    rows = np.arange(height)
    cols = np.clip(seam, 0, width - 1)
    painted[rows, cols] = color
    painted[rows, np.clip(cols - 1, 0, width - 1)] = color
    return painted

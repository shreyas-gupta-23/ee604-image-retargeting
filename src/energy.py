"""Luminance, Hou and Zhang saliency, an optional U2-Net-small map, and oriented lines.

We compute saliency and the line maps once, then crop them with each seam. We recompute forward energy from the luminance that is left. We tried a curve map and chose not to put it in the cost.
"""

from __future__ import annotations

import cv2
import numpy as np


def luminance(image: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance in ``[0, 1]``. ``image`` is RGB uint8 or float."""
    rgb = image.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    if rgb.ndim == 2:
        return rgb
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def spectral_residual_saliency(lum: np.ndarray) -> np.ndarray:
    """Spectral residual in ``[0, 1]``. We subtract a 3x3 average from the log amplitude and keep the phase. We did not use a learned model for this map."""
    gray = lum.astype(np.float32)
    spectrum = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    average = cv2.blur(log_amplitude.astype(np.float32), (3, 3))
    residual = log_amplitude - average
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed).astype(np.float32) ** 2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
    return _normalize(saliency)


_LEARNED_SESSION = None


def learned_saliency(rgb_uint8: np.ndarray) -> np.ndarray:
    """U2-Net-small through rembg ``u2netp``, in ``[0, 1]``. We load the checkpoint once and call it once per image, not on every seam."""
    global _LEARNED_SESSION
    rgb = np.ascontiguousarray(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("learned_saliency expects uint8 RGB of shape (H, W, 3)")
    if _LEARNED_SESSION is None:
        from rembg import new_session
        _LEARNED_SESSION = new_session("u2netp")
    from rembg import remove
    mask = np.asarray(remove(rgb, session=_LEARNED_SESSION, only_mask=True))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    height, width = rgb.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return _normalize(np.asarray(mask, dtype=np.float32))


def gradient_magnitude(lum: np.ndarray) -> np.ndarray:
    """Backward-energy map: sum of absolute Sobel responses."""
    gx = cv2.Sobel(lum.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def curve_protection(lum: np.ndarray) -> np.ndarray:
    """Strong edges in ``[0, 1]``, including curves. We computed this map and then chose not to use it in the cost."""
    gray = np.clip(np.rint(lum * 255.0), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.2)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    cutoff = float(np.quantile(magnitude, 0.90))
    magnitude[magnitude < cutoff] = 0.0
    kernel = np.ones((3, 3), np.uint8)
    magnitude = cv2.dilate(magnitude, kernel)
    return _normalize(magnitude)


def line_protection(lum: np.ndarray, seam_direction: str, detector: str = "hough") -> np.ndarray:
    """Line map in ``[0, 1]``. ``hough`` or ``lsd``. We keep segments of at least 30 px and 12% of the short side, and at most 40 of them.

    We chose the weight from the seam direction: ``|sin theta|`` when the seam is vertical, ``|cos theta|`` when it is horizontal.
    """
    if seam_direction not in ("vertical", "horizontal"):
        raise ValueError("seam_direction must be 'vertical' or 'horizontal'")

    gray_u8 = np.clip(np.rint(lum * 255.0), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.2)
    height, width = lum.shape
    min_length = max(30, int(round(0.12 * min(height, width))))
    segments = _line_segments(blurred, min_length, detector)
    protection = np.zeros((height, width), np.float32)
    if len(segments) == 0:
        return protection
    ranked = []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        ranked.append((length, int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    ranked.sort(reverse=True)
    layer = np.zeros((height, width), np.float32)
    for _, x1, y1, x2, y2 in ranked[:40]:
        theta = np.arctan2(float(y2 - y1), float(x2 - x1))
        if seam_direction == "vertical":
            weight = abs(np.sin(theta))
        else:
            weight = abs(np.cos(theta))
        if weight < 1e-3:
            continue
        layer.fill(0.0)
        cv2.line(layer, (x1, y1), (x2, y2), float(weight), thickness=2)
        protection = np.maximum(protection, layer)
    return protection


def extra_energy(saliency: np.ndarray, lines: np.ndarray, w_s: float, w_l: float,
                 curves: np.ndarray | None = None, w_c: float = 0.0) -> np.ndarray:
    """The extra cost we add: saliency, oriented lines, and the curve map only if we turned it on."""
    cost = w_s * saliency + w_l * lines
    if curves is not None and w_c != 0.0:
        cost = cost + w_c * curves
    return cost.astype(np.float32)


def _line_segments(blurred_u8: np.ndarray, min_length: int, detector: str) -> np.ndarray:
    if detector == "lsd":
        raw = cv2.createLineSegmentDetector().detect(blurred_u8)[0]
    elif detector == "hough":
        edges = cv2.Canny(blurred_u8, 80, 160, apertureSize=3)
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=10,
        )
    else:
        raise ValueError("detector must be 'hough' or 'lsd'")
    if raw is None or len(raw) == 0:
        return np.zeros((0, 4), np.float32)
    # We reshape both detectors to (N, 4). Hough is (N, 1, 4) on OpenCV 4 and (N, 4) on OpenCV 5. LSD is already (N, 4).
    return np.asarray(raw, dtype=np.float32).reshape(-1, 4)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.zeros_like(values)
    return (values - low) / (high - low)

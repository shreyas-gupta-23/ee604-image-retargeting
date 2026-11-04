"""Vertical-seam dynamic programs, removal, and insertion.

A horizontal seam is a vertical seam on the transposed image. The caller
owns that transpose so saliency and line maps stay aligned.

Forward energy follows Rubinstein, Shamir, and Avidan (SIGGRAPH 2008).
Backward energy follows Avidan and Shamir (SIGGRAPH 2007).
"""

from __future__ import annotations

import numpy as np

from .energy import extra_energy, gradient_magnitude, luminance


def backward_seam(energy: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum seam under the classic backward cost.

    ``M(i, j) = E(i, j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))``.
    Returns the seam as one column index per row, and the seam cost.
    """
    height, width = energy.shape
    cumulative = np.zeros((height, width), np.float64)
    step = np.zeros((height, width), np.int8)
    cumulative[0] = energy[0]
    from_left = np.empty(width, np.float64)
    from_right = np.empty(width, np.float64)

    for i in range(1, height):
        previous = cumulative[i - 1]
        from_left[0] = np.inf
        from_left[1:] = previous[:-1]
        from_right[-1] = np.inf
        from_right[:-1] = previous[1:]
        # 0 = from j-1, 1 = from j, 2 = from j+1. Up is the starting choice.
        which = np.ones(width, np.int8)
        best = previous + 0.0
        better = from_left < best
        best = np.where(better, from_left, best)
        which = np.where(better, np.int8(0), which)
        better = from_right < best
        best = np.where(better, from_right, best)
        which = np.where(better, np.int8(2), which)
        cumulative[i] = energy[i] + best
        step[i] = which - 1

    return _backtrack(step, cumulative)


def forward_seam(lum: np.ndarray, pixel_cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum seam under forward energy plus a per-pixel cost.

    Removing pixel ``(i, j)`` joins its left and right neighbors. The new
    edge, and the edge against the pixel the seam just left, is:

        C_U(i, j) = |I(i, j+1) - I(i, j-1)|
        C_L(i, j) = C_U(i, j) + |I(i-1, j) - I(i, j-1)|
        C_R(i, j) = C_U(i, j) + |I(i-1, j) - I(i, j+1)|

    ``C_L``, ``C_U``, and ``C_R`` are the costs of arriving from ``(i-1, j-1)``,
    ``(i-1, j)``, and ``(i-1, j+1)``. The stored cost adds ``pixel_cost``,
    which carries saliency and line protection. Out-of-range neighbors are
    clamped to the border pixel; a step that would leave the image is ``inf``.
    """
    height, width = lum.shape
    cumulative = np.zeros((height, width), np.float64)
    step = np.zeros((height, width), np.int8)
    cumulative[0] = pixel_cost[0]
    left = np.empty(width, np.float64)
    right = np.empty(width, np.float64)
    from_left = np.empty(width, np.float64)
    from_right = np.empty(width, np.float64)

    for i in range(1, height):
        row = lum[i]
        above = lum[i - 1]
        left[0] = row[0]
        left[1:] = row[:-1]
        right[-1] = row[-1]
        right[:-1] = row[1:]

        cost_up = np.abs(right - left)
        cost_left = cost_up + np.abs(above - left)
        cost_right = cost_up + np.abs(above - right)

        previous = cumulative[i - 1]
        from_left[0] = np.inf
        from_left[1:] = previous[:-1] + cost_left[1:]
        from_right[-1] = np.inf
        from_right[:-1] = previous[1:] + cost_right[:-1]
        best = previous + cost_up
        which = np.ones(width, np.int8)
        better = from_left < best
        best = np.where(better, from_left, best)
        which = np.where(better, np.int8(0), which)
        better = from_right < best
        best = np.where(better, from_right, best)
        which = np.where(better, np.int8(2), which)
        cumulative[i] = pixel_cost[i] + best
        step[i] = which - 1

    return _backtrack(step, cumulative)


def vertical_seam(image: np.ndarray, saliency: np.ndarray, lines: np.ndarray,
                  mode: str, w_s: float, w_l: float,
                  curves: np.ndarray | None = None, w_c: float = 0.0) -> tuple[np.ndarray, float]:
    """One vertical seam on ``image`` using the matching line map."""
    lum = luminance(image)
    if mode == "classic":
        return backward_seam(gradient_magnitude(lum))
    if mode != "forward":
        raise ValueError("mode must be 'classic' or 'forward'")
    pixel_cost = extra_energy(saliency, lines, w_s, w_l, curves, w_c)
    return forward_seam(lum, pixel_cost)


def remove_vertical_seam(array: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Drop one pixel per row. Works for ``HxW``, ``HxWxC``, and index maps."""
    height, width = array.shape[:2]
    keep = np.ones((height, width), dtype=bool)
    keep[np.arange(height), seam] = False
    if array.ndim == 2:
        return array[keep].reshape(height, width - 1)
    return array[keep].reshape(height, width - 1, array.shape[2])


def insert_vertical_seams(image: np.ndarray, seams: list[np.ndarray]) -> np.ndarray:
    """Duplicate each recorded seam pixel, averaged with its right neighbor.

    ``seams`` stores column indices into ``image`` itself (the picture before
    insertion), one array of length ``H`` per seam. Seams must not reuse a pixel.
    """
    if not seams:
        return image
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    source = image.reshape(height, width, channels).astype(np.float32)
    order = np.sort(np.stack(seams, axis=1).astype(np.int32), axis=1)
    count = order.shape[1]
    output = np.empty((height, width + count, channels), np.float32)

    for row in range(height):
        cols = order[row]
        written = 0
        cursor = 0
        src_row = source[row]
        dst_row = output[row]
        for col in cols:
            run = int(col) - cursor + 1
            dst_row[written:written + run] = src_row[cursor:int(col) + 1]
            written += run
            neighbor = src_row[int(col) + 1] if int(col) + 1 < width else src_row[int(col)]
            dst_row[written] = 0.5 * (src_row[int(col)] + neighbor)
            written += 1
            cursor = int(col) + 1
        if cursor < width:
            dst_row[written:] = src_row[cursor:]
    result = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return result[:, :, 0]
    return result


def _backtrack(step: np.ndarray, cumulative: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = step.shape
    seam = np.zeros(height, np.int32)
    seam[-1] = int(np.argmin(cumulative[-1]))
    cost = float(cumulative[-1, seam[-1]])
    for i in range(height - 1, 0, -1):
        seam[i - 1] = int(np.clip(seam[i] + int(step[i, seam[i]]), 0, width - 1))
    return seam, cost

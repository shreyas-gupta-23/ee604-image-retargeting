"""Retarget one RGB image to an exact height and width.

We resize for ``scale`` and keep the centered window for ``crop``. ``classic`` is backward energy. For ``ours`` we add saliency and Hough lines to forward energy, with ``weights=(w_s, w_l)``. For ``hybrid`` we use that same cost, stop the seams at 75% of a side, and scale the rest.
"""

from __future__ import annotations

import numpy as np

from .baselines import center_crop_resize, scale_to
from .energy import (
    curve_protection,
    learned_saliency,
    line_protection,
    luminance,
    spectral_residual_saliency,
)
from .metrics import accumulate_seam, empty_counts
from .seam import insert_vertical_seams, remove_vertical_seam, vertical_seam

DEFAULT_WEIGHTS = (0.5, 0.8)
# We chose 0.25 for the learned source only. It was the smallest of {0.25, 0.5, 1, 2, 4} that kept the learned mass under half the forward-only value on Johanneskirche and obama, without a tear. Spectral runs still use DEFAULT_WEIGHTS.
LEARNED_W_S = 0.25
LEARNED_W_L = 0.8
# We let the seams remove at most this fraction of a side, then we scale.
HYBRID_KEEP = 0.75


def retarget(image: np.ndarray, target_hw: tuple[int, int], method: str = "ours",
             weights: tuple[float, float] = DEFAULT_WEIGHTS, return_info: bool = False,
             saliency_source: str = "spectral", score_both: bool = False):
    """RGB uint8 image. With ``return_info`` we also return the seam counts. Scale and crop have no seams, and for hybrid we score only the seams we actually cut."""
    rgb = _as_uint8_rgb(image)
    target_h = max(2, int(target_hw[0]))
    target_w = max(2, int(target_hw[1]))
    w_s, w_l = float(weights[0]), float(weights[1])

    if method == "scale":
        out = scale_to(rgb, target_h, target_w)
        info = _baseline_info(out, (target_h, target_w))
        return (out, info) if return_info else out
    if method == "crop":
        out = center_crop_resize(rgb, target_h, target_w)
        info = _baseline_info(out, (target_h, target_w))
        return (out, info) if return_info else out
    if method not in ("classic", "ours", "hybrid"):
        raise ValueError("method must be 'scale', 'crop', 'classic', 'ours', or 'hybrid'")

    mode = "classic" if method == "classic" else "forward"
    height, width = rgb.shape[:2]
    if method == "hybrid":
        # We use the same cost as ours, and we stop the seams at 75% of the original side.
        shrink_h = _hybrid_stop(height, target_h)
        shrink_w = _hybrid_stop(width, target_w)
        detector, w_c = "hough", 0.0
    else:
        shrink_h, shrink_w = min(target_h, height), min(target_w, width)
        detector, w_c = "hough", 0.0
    carved, counts = _shrink(
        rgb, shrink_h, shrink_w, mode, w_s, w_l, w_c, detector, saliency_source, score_both,
    )
    counts["scaled_after_seams"] = 0
    scale_h = target_h if target_h < carved.shape[0] else carved.shape[0]
    scale_w = target_w if target_w < carved.shape[1] else carved.shape[1]
    if (scale_h, scale_w) != carved.shape[:2]:
        carved = scale_to(carved, scale_h, scale_w)
        counts["scaled_after_seams"] = 1

    if target_w > carved.shape[1]:
        carved, insert_counts = _enlarge_width(
            carved, target_w, mode, w_s, w_l, w_c, detector, saliency_source,
        )
        _merge_path_counts(counts, insert_counts)
    if target_h > carved.shape[0]:
        carved, insert_counts = _enlarge_height(
            carved, target_h, mode, w_s, w_l, w_c, detector, saliency_source,
        )
        _merge_path_counts(counts, insert_counts)

    info = dict(counts)
    info["output_hw"] = tuple(carved.shape[:2])
    info["target_hw"] = (target_h, target_w)
    return (carved, info) if return_info else carved


def next_vertical_seam(image: np.ndarray, method: str = "ours",
                       weights: tuple[float, float] = DEFAULT_WEIGHTS) -> np.ndarray:
    """Column index per row of the next vertical seam, for the demo overlay."""
    rgb = _as_uint8_rgb(image)
    maps = protection_maps(rgb)
    mode = "classic" if method == "classic" else "forward"
    w_s, w_l = (0.0, 0.0) if method == "classic" else (float(weights[0]), float(weights[1]))
    seam, _ = vertical_seam(rgb, maps["saliency"], maps["lines_vertical"], mode, w_s, w_l)
    return seam


def protection_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    """Luminance, saliency, and both line maps, for the demo overlays."""
    rgb = _as_uint8_rgb(image)
    lum = luminance(rgb)
    return {
        "luminance": lum,
        "saliency": spectral_residual_saliency(lum),
        "lines_vertical": line_protection(lum, "vertical"),
        "lines_horizontal": line_protection(lum, "horizontal"),
    }


def _hybrid_stop(original: int, target: int, keep: float = HYBRID_KEEP) -> int:
    """The size we still reach with seams. We finish any deeper cut by scaling."""
    if target >= original:
        return original
    floor = max(2, int(round(original * keep)))
    return max(target, floor)


def _drop_curve(curves, seam, transpose: bool):
    if curves is None:
        return None
    if transpose:
        return remove_vertical_seam(curves.T, seam).T
    return remove_vertical_seam(curves, seam)


def _saliency_for_cost(image, lum, source: str):
    if source == "spectral":
        return spectral_residual_saliency(lum)
    if source == "learned":
        return learned_saliency(image)
    raise ValueError("saliency_source must be 'spectral' or 'learned'")


def _count_mass(counts: dict, key: str, saliency: np.ndarray, seam: np.ndarray) -> None:
    rows = np.arange(seam.shape[0])
    counts[key] += float(saliency[rows, seam].sum())


def _count_blob(counts: dict, saliency: np.ndarray, seam: np.ndarray) -> None:
    """Seam pixels whose learned-map value is above 0.5. We skip this unless we armed that count."""
    if "learned_blob_hits" not in counts:
        return
    rows = np.arange(seam.shape[0])
    counts["learned_blob_hits"] += int(np.count_nonzero(saliency[rows, seam] > 0.5))


def _shrink(image, target_h, target_w, mode, w_s, w_l, w_c=0.0, detector="hough",
            saliency_source="spectral", score_both=False):
    lum = luminance(image)
    saliency = _saliency_for_cost(image, lum, saliency_source)
    line_v = line_protection(lum, "vertical", detector)
    line_h = line_protection(lum, "horizontal", detector)
    # We score LSD only. We chose not to pass it into the dynamic program.
    lsd_v = line_protection(lum, "vertical", "lsd")
    lsd_h = line_protection(lum, "horizontal", "lsd")
    curves = curve_protection(lum) if w_c else None
    counts = empty_counts()
    counts["saliency_total"] = float(saliency.sum())
    if saliency_source == "learned" or (score_both and saliency_source == "spectral"):
        counts["learned_blob_hits"] = 0
    # We score the other saliency map only. We chose not to pass it into the dynamic program.
    other = None
    other_key = None
    if score_both:
        if saliency_source == "spectral":
            other = learned_saliency(image)
            other_key = "learned_removed"
            counts["learned_total"] = float(other.sum())
            counts["learned_removed"] = 0.0
        else:
            other = spectral_residual_saliency(lum)
            other_key = "spectral_removed"
            counts["spectral_total"] = float(other.sum())
            counts["spectral_removed"] = 0.0

    current = image
    while current.shape[0] > target_h or current.shape[1] > target_w:
        can_v = current.shape[1] > target_w and current.shape[1] > 2
        can_h = current.shape[0] > target_h and current.shape[0] > 2
        if not can_v and not can_h:
            break
        if can_v and can_h:
            seam_v, cost_v = vertical_seam(current, saliency, line_v, mode, w_s, w_l, curves, w_c)
            seam_h, cost_h = _horizontal_seam(current, saliency, line_h, mode, w_s, w_l, curves, w_c)
            take_vertical = cost_v <= cost_h
        else:
            take_vertical = can_v
            seam_v = seam_h = None

        if take_vertical:
            if seam_v is None:
                seam_v, _ = vertical_seam(current, saliency, line_v, mode, w_s, w_l, curves, w_c)
            accumulate_seam(counts, saliency, line_v, seam_v, vertical=True)
            _count_lsd(counts, lsd_v, seam_v)
            if saliency_source == "learned":
                _count_blob(counts, saliency, seam_v)
            elif other is not None:
                _count_blob(counts, other, seam_v)
            if other is not None:
                _count_mass(counts, other_key, other, seam_v)
                other = remove_vertical_seam(other, seam_v)
            current = remove_vertical_seam(current, seam_v)
            saliency = remove_vertical_seam(saliency, seam_v)
            line_v = remove_vertical_seam(line_v, seam_v)
            line_h = remove_vertical_seam(line_h, seam_v)
            lsd_v = remove_vertical_seam(lsd_v, seam_v)
            lsd_h = remove_vertical_seam(lsd_h, seam_v)
            curves = _drop_curve(curves, seam_v, transpose=False)
        else:
            if seam_h is None:
                seam_h, _ = _horizontal_seam(current, saliency, line_h, mode, w_s, w_l, curves, w_c)
            transposed_sal = saliency.T
            transposed_line = line_h.T
            accumulate_seam(counts, transposed_sal, transposed_line, seam_h, vertical=False)
            _count_lsd(counts, lsd_h.T, seam_h)
            if saliency_source == "learned":
                _count_blob(counts, transposed_sal, seam_h)
            elif other is not None:
                _count_blob(counts, other.T, seam_h)
            if other is not None:
                _count_mass(counts, other_key, other.T, seam_h)
                other = remove_vertical_seam(other.T, seam_h).T
            current = _transpose_image(remove_vertical_seam(_transpose_image(current), seam_h))
            saliency = remove_vertical_seam(transposed_sal, seam_h).T
            line_h = remove_vertical_seam(transposed_line, seam_h).T
            line_v = remove_vertical_seam(line_v.T, seam_h).T
            lsd_h = remove_vertical_seam(lsd_h.T, seam_h).T
            lsd_v = remove_vertical_seam(lsd_v.T, seam_h).T
            curves = _drop_curve(curves, seam_h, transpose=True)
    if score_both and saliency_source == "spectral":
        counts["spectral_total"] = counts["saliency_total"]
        counts["spectral_removed"] = counts["saliency_removed"]
    elif score_both:
        counts["learned_total"] = counts["saliency_total"]
        counts["learned_removed"] = counts["saliency_removed"]
    return current, counts


def _enlarge_width(image, target_w, mode, w_s, w_l, w_c=0.0, detector="hough",
                   saliency_source="spectral"):
    """We record seams by removing them, then we insert averaged duplicates back into ``image``."""
    current = image
    total = empty_counts()
    while current.shape[1] < target_w:
        room = current.shape[1] - 2
        need = target_w - current.shape[1]
        step = min(need, room)
        if step <= 0:
            break
        seams, counts = _record_vertical_seams(
            current, step, mode, w_s, w_l, w_c, detector, saliency_source,
        )
        _merge_path_counts(total, counts)
        current = insert_vertical_seams(current, seams)
    return current, total


def _enlarge_height(image, target_h, mode, w_s, w_l, w_c=0.0, detector="hough",
                    saliency_source="spectral"):
    transposed = _transpose_image(image)
    enlarged, counts = _enlarge_width(
        transposed, target_h, mode, w_s, w_l, w_c, detector, saliency_source,
    )
    counts["horizontal_seams"] = counts["vertical_seams"]
    counts["vertical_seams"] = 0
    return _transpose_image(enlarged), counts


def _record_vertical_seams(image, count, mode, w_s, w_l, w_c=0.0, detector="hough",
                           saliency_source="spectral"):
    height, width = image.shape[:2]
    lum = luminance(image)
    saliency = _saliency_for_cost(image, lum, saliency_source)
    lines = line_protection(lum, "vertical", detector)
    curves = curve_protection(lum) if w_c else None
    columns = np.tile(np.arange(width, dtype=np.int32), (height, 1))
    work = image
    seams = []
    counts = empty_counts()
    counts["saliency_total"] = float(saliency.sum())
    for _ in range(count):
        seam, _ = vertical_seam(work, saliency, lines, mode, w_s, w_l, curves, w_c)
        rows = np.arange(work.shape[0])
        seams.append(columns[rows, seam].copy())
        # We keep the pixel when we insert a seam, so we do not count that saliency as removed.
        counts["line_hits"] += int(np.count_nonzero(lines[rows, seam] > 0))
        counts["seam_pixels"] += int(seam.shape[0])
        counts["vertical_seams"] += 1
        work = remove_vertical_seam(work, seam)
        saliency = remove_vertical_seam(saliency, seam)
        lines = remove_vertical_seam(lines, seam)
        curves = _drop_curve(curves, seam, transpose=False)
        columns = remove_vertical_seam(columns, seam)
    return seams, counts


def _horizontal_seam(image, saliency, line_h, mode, w_s, w_l, curves=None, w_c=0.0):
    curve_t = None if curves is None else curves.T
    return vertical_seam(
        _transpose_image(image), saliency.T, line_h.T, mode, w_s, w_l, curve_t, w_c,
    )


def _transpose_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.ascontiguousarray(np.transpose(image, (1, 0)))
    return np.ascontiguousarray(np.transpose(image, (1, 0, 2)))


def _count_lsd(counts: dict, lsd: np.ndarray, seam: np.ndarray) -> None:
    rows = np.arange(seam.shape[0])
    counts["lsd_hits"] += int(np.count_nonzero(lsd[rows, seam] > 0))


def _merge_path_counts(total: dict, extra: dict) -> None:
    for key in ("line_hits", "lsd_hits", "seam_pixels", "vertical_seams", "horizontal_seams"):
        total[key] += extra[key]


def _baseline_info(image, target_hw) -> dict:
    counts = empty_counts()
    counts["output_hw"] = tuple(image.shape[:2])
    counts["target_hw"] = target_hw
    return counts


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        values = array.astype(np.float32)
        if values.max() <= 1.0:
            values = values * 255.0
        array = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)

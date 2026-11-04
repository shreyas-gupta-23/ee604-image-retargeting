"""Scores collected while seams are chosen.

Saliency mass is the sum of the spectral-residual map on pixels that are
actually deleted. Line crossings count seam pixels that land on a positive
line-protection value, reported per 100 seam pixels. Scale and crop do not
choose seams, so those two scores are left unset for them.
"""

from __future__ import annotations

import numpy as np


def empty_counts() -> dict:
    return {
        "saliency_total": 0.0,
        "saliency_removed": 0.0,
        "line_hits": 0,
        "seam_pixels": 0,
        "vertical_seams": 0,
        "horizontal_seams": 0,
    }


def saliency_removed_fraction(counts: dict) -> float:
    total = float(counts["saliency_total"])
    if total <= 0.0:
        return 0.0
    return float(counts["saliency_removed"]) / total


def line_crossings_per_100(counts: dict) -> float:
    pixels = int(counts["seam_pixels"])
    if pixels <= 0:
        return 0.0
    return 100.0 * float(counts["line_hits"]) / pixels


def size_matches(image: np.ndarray, target_hw: tuple[int, int]) -> bool:
    return tuple(image.shape[:2]) == (int(target_hw[0]), int(target_hw[1]))


def accumulate_seam(counts: dict, saliency: np.ndarray, lines: np.ndarray,
                    seam: np.ndarray, vertical: bool) -> None:
    rows = np.arange(seam.shape[0])
    counts["saliency_removed"] += float(saliency[rows, seam].sum())
    counts["line_hits"] += int(np.count_nonzero(lines[rows, seam] > 0))
    counts["seam_pixels"] += int(seam.shape[0])
    if vertical:
        counts["vertical_seams"] += 1
    else:
        counts["horizontal_seams"] += 1

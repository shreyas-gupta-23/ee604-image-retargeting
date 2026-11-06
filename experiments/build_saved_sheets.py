"""Contact sheets from the PNGs we already saved. We do not carve again."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_experiments  # noqa: E402
from run_experiments import contact_sheet, load_rgb, save_rgb  # noqa: E402

RESULT = ROOT / "experiments" / "results"
LOG = RESULT / "overnight_log.txt"
ORIGINAL = (
    "brick_house",
    "butterfly",
    "child",
    "eagle",
    "Johanneskirche",
    "obama",
    "Sanfrancisco",
    "waterfall",
)
EIGHT_METHODS = (
    ("original", "Original"),
    ("scale", "Scale"),
    ("classic", "Classic"),
    ("ours_spectral", "Ours spectral"),
    ("ours_learned", "Ours learned"),
)
THIRTY_METHODS = (
    ("original", "Original"),
    ("classic", "Classic"),
    ("ours_spectral", "Ours spectral"),
    ("ours_learned", "Ours learned"),
)


def log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(message, flush=True)


def csv_targets(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    targets = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["image"], row["setting"])
            target = (int(row["target_h"]), int(row["target_w"]))
            if key in targets and targets[key] != target:
                log(f"WARNING target conflict {key} {targets[key]} vs {target}")
            targets[key] = target
    return targets


def check_png(name: str, setting: str, method: str, targets: dict) -> np.ndarray:
    path = RESULT / "outputs" / setting / name / f"{method}.png"
    image = load_rgb(path)
    height, width = image.shape[:2]
    target_h, target_w = targets[(name, setting)]
    if method == "original":
        expected_w = max(2, int(round(width * (0.75 if setting == "width_75" else 0.50))))
        if height != target_h or expected_w != target_w:
            log(
                f"WARNING size mismatch {setting}/{name}/original "
                f"disk {height}x{width} implies target width {expected_w}, csv target {target_h}x{target_w}"
            )
    elif (height, width) != (target_h, target_w):
        log(
            f"WARNING size mismatch {setting}/{name}/{method} "
            f"disk {height}x{width} requested {target_h}x{target_w}"
        )
    return image


def fit_panel(image: np.ndarray, title: str) -> np.ndarray:
    """We keep the same 168-pixel panels and 26-pixel labels as the eight-image sheets."""
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * (168 / float(height)))))
    resized = cv2.resize(image, (new_w, 168), interpolation=cv2.INTER_AREA)
    bar = np.full((26, new_w, 3), 255, np.uint8)
    scale = 0.48
    while scale > 0.28:
        text_w = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_w + 6 <= new_w:
            break
        scale -= 0.02
    cv2.putText(bar, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, resized])


def sheet(names: list[str], setting: str, methods, targets: dict) -> np.ndarray:
    run_experiments.fit_panel = fit_panel
    run_experiments.PANEL_H = 168
    records = []
    for name in names:
        panels = [(title, check_png(name, setting, method, targets)) for method, title in methods]
        records.append((name, panels))
    return contact_sheet(records)


def main() -> None:
    eight_targets = csv_targets(RESULT / "metrics_learned_8.csv")
    thirty_targets = csv_targets(RESULT / "metrics_learned_n30.csv")
    names = (RESULT / "n30_images.txt").read_text(encoding="utf-8").splitlines()[1:]
    names = [name.strip() for name in names if name.strip()]
    if names != sorted(names):
        log("WARNING n30_images.txt is not alphabetical; sheets follow the file order")
    save_rgb(RESULT / "fig_learned_all50.png", sheet(list(ORIGINAL), "width_50", EIGHT_METHODS, eight_targets))
    save_rgb(RESULT / "fig_learned_all75.png", sheet(list(ORIGINAL), "width_75", EIGHT_METHODS, eight_targets))
    mid = len(names) // 2
    save_rgb(RESULT / "fig_n30_overview_a.png", sheet(names[:mid], "width_75", THIRTY_METHODS, thirty_targets))
    save_rgb(RESULT / "fig_n30_overview_b.png", sheet(names[mid:], "width_75", THIRTY_METHODS, thirty_targets))
    log(
        "saved fig_learned_all50.png fig_learned_all75.png "
        "fig_n30_overview_a.png fig_n30_overview_b.png from existing PNGs; no carving"
    )
    log(f"n30 sheet order alphabetical file order, split {mid} and {len(names) - mid}")


if __name__ == "__main__":
    main()

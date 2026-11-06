"""We rebuild the Sanfrancisco 50% failure figure from the PNGs we already saved. We do not carve again."""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "outputs" / "width_50" / "Sanfrancisco"
DEST = ROOT / "experiments" / "results" / "fig_failure.png"
PANELS = (
    ("original", "Original"),
    ("scale", "Scale"),
    ("crop", "Crop"),
    ("classic", "Classic"),
    ("ours_spectral", "Ours (spectral)"),
    ("ours_learned", "Ours (learned)"),
)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"missing {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_panel(image: np.ndarray, title: str) -> np.ndarray:
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
    else:
        print(f"WARNING label may clip: {title} panel {new_w}px")
    cv2.putText(bar, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    print(f"{title}: disk {height}x{width} panel {new_w}x168 font {scale:.2f}")
    return np.vstack([bar, resized])


def main() -> None:
    tiles = [fit_panel(load_rgb(OUT / f"{method}.png"), title) for method, title in PANELS]
    tag_h = tiles[0].shape[0]
    tag = np.full((tag_h, 150, 3), 245, np.uint8)
    cv2.putText(tag, "Sanfrancisco", (6, tag_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.hstack([tag, *tiles])
    cv2.imwrite(str(DEST), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {DEST.name} {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()

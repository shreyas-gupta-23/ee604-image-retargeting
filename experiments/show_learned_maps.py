"""Spectral residual beside U2-Net-small on the four inspection images.

Does not carve seams. Writes experiments/results/fig_learned_maps.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_experiments import contact_sheet, load_rgb, save_rgb  # noqa: E402
import run_experiments  # noqa: E402
from src.baselines import limit_long_side  # noqa: E402
from src.display import colorize  # noqa: E402
from src.energy import learned_saliency, luminance, spectral_residual_saliency  # noqa: E402

NAMES = ("Johanneskirche", "obama", "Sanfrancisco", "child")
OUT = ROOT / "experiments" / "results" / "fig_learned_maps.png"


def main() -> None:
    run_experiments.PANEL_H = 280
    records = []
    for name in NAMES:
        image = limit_long_side(load_rgb(ROOT / "experiments" / "retargetme" / f"{name}.png"), 640)
        spectral = spectral_residual_saliency(luminance(image))
        learned = learned_saliency(image)
        if learned.shape != spectral.shape:
            raise SystemExit(f"{name} shape {learned.shape} != {spectral.shape}")
        print(
            f"{name} {image.shape[0]}x{image.shape[1]} "
            f"spectral mean {spectral.mean():.3f} >0.5 {float((spectral > 0.5).mean()):.3f} "
            f"learned mean {learned.mean():.3f} min {learned.min():.3f} max {learned.max():.3f} "
            f">0.5 {float((learned > 0.5).mean()):.3f}",
            flush=True,
        )
        records.append((name, [
            ("Input", image),
            ("Spectral", colorize(spectral)),
            ("U2-Net-p", colorize(learned)),
        ]))
    save_rgb(OUT, contact_sheet(records))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

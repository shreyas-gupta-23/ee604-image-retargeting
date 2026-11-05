"""w_s sweep for learned saliency on Johanneskirche and obama only.

Selection rule, fixed before the numbers are read:
the smallest w_s in {0.25, 0.5, 1.0, 2.0, 4.0} at which the learned-mass
fraction is below half of the forward-only learned-mass fraction on both
images, unless that carved image shows visible tearing, in which case take
the next lower value. w_l stays 0.8. This script does not apply the rule.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_experiments  # noqa: E402
from run_experiments import contact_sheet, load_rgb, save_rgb  # noqa: E402
from src.baselines import limit_long_side  # noqa: E402
from src.metrics import line_crossings_per_100, lsd_crossings_per_100, saliency_removed_fraction  # noqa: E402
from src.retarget import retarget  # noqa: E402

NAMES = ("Johanneskirche", "obama")
WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
W_L = 0.8
OUT_CSV = ROOT / "experiments" / "results" / "sweep_learned_ws.csv"
OUT_FIG = ROOT / "experiments" / "results" / "fig_sweep_learned_ws.png"


def fraction(info: dict, removed_key: str, total_key: str) -> float:
    total = float(info.get(total_key, 0.0))
    if total <= 0.0:
        return 0.0
    return float(info.get(removed_key, 0.0)) / total


def blob_fraction(info: dict) -> float:
    pixels = int(info.get("seam_pixels", 0))
    if pixels <= 0:
        return 0.0
    return float(info.get("learned_blob_hits", 0)) / pixels


def carve(image, weights, source):
    height, width = image.shape[:2]
    target = (height, max(2, int(round(width * 0.75))))
    output, info = retarget(
        image,
        target,
        method="ours",
        weights=weights,
        return_info=True,
        saliency_source=source,
        score_both=True,
    )
    learned = fraction(info, "learned_removed", "learned_total")
    spectral = fraction(info, "spectral_removed", "spectral_total")
    if source == "learned":
        learned = saliency_removed_fraction(info)
    else:
        spectral = saliency_removed_fraction(info)
    return output, {
        "learned_mass": learned,
        "spectral_mass": spectral,
        "hough": line_crossings_per_100(info),
        "lsd": lsd_crossings_per_100(info),
        "blob": blob_fraction(info),
        "seam_pixels": int(info["seam_pixels"]),
    }


def main() -> None:
    run_experiments.PANEL_H = 220
    images = {
        name: limit_long_side(load_rgb(ROOT / "experiments" / "retargetme" / f"{name}.png"), 640)
        for name in NAMES
    }
    rows = []
    panels = {name: [] for name in NAMES}
    for name, image in images.items():
        print(f"{name} classic", flush=True)
        height, width = image.shape[:2]
        target = (height, max(2, int(round(width * 0.75))))
        classic, _ = retarget(image, target, method="classic", return_info=True)
        print(f"{name} spectral full", flush=True)
        spectral_full, spectral_info = carve(image, (0.5, W_L), "spectral")
        print(f"{name} forward-only", flush=True)
        _forward, forward_info = carve(image, (0.0, 0.0), "learned")
        rows.append({"image": name, "setting": "forward_only", "w_s": 0.0, "w_l": 0.0, **forward_info})
        rows.append({
            "image": name,
            "setting": "spectral_full",
            "w_s": 0.5,
            "w_l": W_L,
            "learned_mass": spectral_info["learned_mass"],
            "spectral_mass": spectral_info["spectral_mass"],
            "hough": spectral_info["hough"],
            "lsd": spectral_info["lsd"],
            "blob": spectral_info["blob"],
            "seam_pixels": spectral_info["seam_pixels"],
        })
        print(
            f"  forward learned {forward_info['learned_mass']:.4f} "
            f"blob {forward_info['blob']:.4f} half {0.5 * forward_info['learned_mass']:.4f}",
            flush=True,
        )
        panels[name].append(("Classic", classic))
        panels[name].append(("Spectral full", spectral_full))
        for w_s in WEIGHTS:
            print(f"{name} learned w_s={w_s}", flush=True)
            output, info = carve(image, (w_s, W_L), "learned")
            rows.append({"image": name, "setting": "learned_full", "w_s": w_s, "w_l": W_L, **info})
            print(
                f"  learned {info['learned_mass']:.4f} spectral {info['spectral_mass']:.4f} "
                f"hough {info['hough']:.4f} lsd {info['lsd']:.4f} blob {info['blob']:.4f}",
                flush=True,
            )
            panels[name].append((f"Learned {w_s:g}", output))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_rgb(OUT_FIG, contact_sheet([(name, panels[name]) for name in NAMES]))
    print("wrote", OUT_CSV, flush=True)
    print("wrote", OUT_FIG, flush=True)


if __name__ == "__main__":
    main()

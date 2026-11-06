"""At 50% width we compare a full carve with seams stopped at 75% and then scaled. A 75% target has to match the full model, or we do not trust the run."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_experiments import (  # noqa: E402
    RESULT_DIR,
    contact_sheet,
    discover_images,
    load_rgb,
    save_rgb,
)
from src.baselines import limit_long_side  # noqa: E402
from src.metrics import size_matches  # noqa: E402
from src.retarget import retarget  # noqa: E402

ORDER = [
    "brick_house",
    "butterfly",
    "child",
    "eagle",
    "Johanneskirche",
    "obama",
    "Sanfrancisco",
    "waterfall",
]
PAPER = ["Johanneskirche", "obama", "Sanfrancisco", "waterfall"]


def main() -> None:
    images = {
        path.stem: limit_long_side(load_rgb(path), 640) for path in discover_images()
    }
    check = images["Sanfrancisco"]
    height, width = check.shape[:2]
    target_75 = (height, max(2, int(round(width * 0.75))))
    ours_75, info_75 = retarget(check, target_75, method="ours", return_info=True)
    hybrid_75, info_h = retarget(check, target_75, method="hybrid", return_info=True)
    diff = int(np.abs(ours_75.astype(np.int16) - hybrid_75.astype(np.int16)).max())
    print(
        f"75% Sanfrancisco max abs diff {diff} "
        f"scaled {info_h.get('scaled_after_seams')} seams {info_75.get('vertical_seams')}",
        flush=True,
    )
    if diff != 0 or info_h.get("scaled_after_seams"):
        raise SystemExit("hybrid at 75% should match ours and should not scale")

    rows = []
    outputs = {}
    for name in ORDER:
        image = images[name]
        height, width = image.shape[:2]
        target = (height, max(2, int(round(width * 0.50))))
        outputs[(name, "original")] = image
        for method in ("scale", "ours", "hybrid"):
            print(f"{name} {method}", flush=True)
            started = time.perf_counter()
            output, info = retarget(image, target, method=method, return_info=True)
            elapsed = time.perf_counter() - started
            print(
                f"  {elapsed:.1f}s {output.shape[0]}x{output.shape[1]} "
                f"seams={info.get('vertical_seams')} scaled={info.get('scaled_after_seams')}",
                flush=True,
            )
            if not size_matches(output, target):
                raise SystemExit(f"size mismatch {name} {method}")
            outputs[(name, method)] = output
            rows.append({
                "image": name,
                "method": method,
                "target_w": target[1],
                "out_w": output.shape[1],
                "vertical_seams": info.get("vertical_seams", 0),
                "scaled_after_seams": info.get("scaled_after_seams", 0),
                "runtime_s": f"{elapsed:.4f}",
            })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "hybrid_50.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def sheet(names: list[str]):
        records = []
        for name in names:
            records.append((name, [
                ("Original", outputs[(name, "original")]),
                ("Scale", outputs[(name, "scale")]),
                ("Ours", outputs[(name, "ours")]),
                ("Hybrid", outputs[(name, "hybrid")]),
            ]))
        return contact_sheet(records)

    save_rgb(RESULT_DIR / "fig_hybrid_50.png", sheet(PAPER))
    save_rgb(RESULT_DIR / "fig_hybrid_50_all.png", sheet(ORDER))
    print("wrote figures", flush=True)


if __name__ == "__main__":
    main()

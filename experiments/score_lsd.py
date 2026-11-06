"""We add an LSD score to metrics.csv. We do not change the cost. If the saved saliency or Hough scores do not match, we stop."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_experiments import (  # noqa: E402
    RESULT_DIR,
    WEIGHTS,
    discover_images,
    load_rgb,
    run_one,
)
from src.baselines import limit_long_side  # noqa: E402
from src.metrics import (  # noqa: E402
    line_crossings_per_100,
    lsd_crossings_per_100,
    saliency_removed_fraction,
)

CSV_PATH = RESULT_DIR / "metrics.csv"
METHODS = ("classic", "forward", "forward_saliency", "forward_lines", "ours")
COLUMN = "lsd_crossings_per_100"


def targets(image):
    height, width = image.shape[:2]
    return {
        "width_75": (height, max(2, int(round(width * 0.75)))),
        "width_50": (height, max(2, int(round(width * 0.50)))),
    }


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if COLUMN not in fieldnames:
        fieldnames.append(COLUMN)
    by_key = {(row["image"], row["setting"], row["method"]): row for row in rows}

    images = {
        path.stem: limit_long_side(load_rgb(path), 640) for path in discover_images()
    }
    fresh: dict[tuple[str, str, str], str] = {}
    new_rows: list[dict[str, str]] = []

    for name in list(dict.fromkeys(row["image"] for row in rows)):
        image = images[name]
        for setting, target in targets(image).items():
            saved = by_key.get((name, setting, "classic"))
            if saved is not None:
                if (int(saved["target_h"]), int(saved["target_w"])) != target:
                    raise SystemExit(f"target mismatch {name} {setting}: csv {saved['target_h']}x{saved['target_w']} vs {target}")
            for method in METHODS:
                print(f"{name} {setting} {method}", flush=True)
                output, info, elapsed = run_one(image, target, method)
                sal = f"{saliency_removed_fraction(info):.6f}"
                lines = f"{line_crossings_per_100(info):.4f}"
                lsd = f"{lsd_crossings_per_100(info):.4f}"
                print(f"  sal {sal} hough {lines} lsd {lsd} {elapsed:.1f}s", flush=True)
                key = (name, setting, method)
                existing = by_key.get(key)
                if existing is not None:
                    if existing["saliency_removed_fraction"] != sal or existing["line_crossings_per_100"] != lines:
                        raise SystemExit(
                            f"seam score changed for {name} {setting} {method}: "
                            f"csv {existing['saliency_removed_fraction']}/{existing['line_crossings_per_100']} "
                            f"rerun {sal}/{lines}"
                        )
                else:
                    new_rows.append({
                        "image": name,
                        "setting": setting,
                        "method": method,
                        "w_s": str(WEIGHTS[method][0]),
                        "w_l": str(WEIGHTS[method][1]),
                        "target_h": str(target[0]),
                        "target_w": str(target[1]),
                        "out_h": str(output.shape[0]),
                        "out_w": str(output.shape[1]),
                        "size_ok": "1" if output.shape[:2] == target else "0",
                        "saliency_removed_fraction": sal,
                        "line_crossings_per_100": lines,
                        "runtime_s": f"{elapsed:.4f}",
                        COLUMN: lsd,
                    })
                fresh[key] = lsd

    for row in rows:
        key = (row["image"], row["setting"], row["method"])
        row[COLUMN] = fresh.get(key, row.get(COLUMN, ""))
    rows.extend(new_rows)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {COLUMN} for {len(fresh)} runs, appended {len(new_rows)} rows", flush=True)


if __name__ == "__main__":
    main()

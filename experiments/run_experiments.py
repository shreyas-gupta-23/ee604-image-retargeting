"""Compare scale, crop, classic seams, and the combined energy.

Writes a CSV, a text summary, and OpenCV contact sheets. Images are the
RetargetMe subset in ``experiments/retargetme``. The long side is capped
at 640 px before any method runs.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import limit_long_side  # noqa: E402
from src.display import colorize, draw_vertical_seam  # noqa: E402
from src.energy import line_protection, luminance  # noqa: E402
from src.metrics import line_crossings_per_100, saliency_removed_fraction, size_matches  # noqa: E402
from src.retarget import next_vertical_seam, protection_maps, retarget  # noqa: E402

IMAGE_DIR = ROOT / "experiments" / "retargetme"
RESULT_DIR = ROOT / "experiments" / "results"
MAX_SIDE = 640
PANEL_H = 168
WEIGHTS = {
    "forward": (0.0, 0.0),
    "forward_saliency": (0.5, 0.0),
    "forward_lines": (0.0, 0.8),
    "ours": (0.5, 0.8),
    "classic": (0.0, 0.0),
    "scale": (0.0, 0.0),
    "crop": (0.0, 0.0),
}
SEAM_METHODS = {"classic", "ours", "forward", "forward_saliency", "forward_lines"}


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def discover_images() -> list[Path]:
    paths = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if len(paths) < 1:
        raise FileNotFoundError(
            f"No images in {IMAGE_DIR}. Run experiments/download_retargetme.py first."
        )
    return paths


def run_one(image: np.ndarray, target_hw, method: str) -> tuple[np.ndarray, dict, float]:
    weights = WEIGHTS[method]
    started = time.perf_counter()
    if method in ("forward", "forward_saliency", "forward_lines"):
        output, info = retarget(image, target_hw, method="ours", weights=weights, return_info=True)
    else:
        output, info = retarget(image, target_hw, method=method, weights=weights, return_info=True)
    elapsed = time.perf_counter() - started
    return output, info, elapsed


def row_for(name, setting, method, image, output, info, elapsed, target_hw) -> dict:
    seam = method in SEAM_METHODS or method in ("forward", "forward_saliency", "forward_lines")
    sal = ""
    lines = ""
    if seam and info.get("seam_pixels", 0) > 0:
        sal = f"{saliency_removed_fraction(info):.6f}"
        lines = f"{line_crossings_per_100(info):.4f}"
    elif seam:
        sal = "0"
        lines = "0"
    return {
        "image": name,
        "setting": setting,
        "method": method,
        "w_s": WEIGHTS[method][0],
        "w_l": WEIGHTS[method][1],
        "target_h": target_hw[0],
        "target_w": target_hw[1],
        "out_h": output.shape[0],
        "out_w": output.shape[1],
        "size_ok": int(size_matches(output, target_hw)),
        "saliency_removed_fraction": sal,
        "line_crossings_per_100": lines,
        "runtime_s": f"{elapsed:.4f}",
    }


def fit_panel(image: np.ndarray, title: str) -> np.ndarray:
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * (PANEL_H / float(height)))))
    resized = cv2.resize(image, (new_w, PANEL_H), interpolation=cv2.INTER_AREA)
    bar = np.full((26, new_w, 3), 255, np.uint8)
    cv2.putText(bar, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, resized])


def stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] == width:
            padded.append(row)
            continue
        canvas = np.full((row.shape[0], width, 3), 255, np.uint8)
        canvas[:, :row.shape[1]] = row
        padded.append(canvas)
    return np.vstack(padded)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def contact_sheet(records: list[tuple[str, list[tuple[str, np.ndarray]]]]) -> np.ndarray:
    """records: (row title, [(panel title, image), ...])."""
    rows = []
    for row_title, panels in records:
        tiles = [fit_panel(image, title) for title, image in panels]
        # A narrow name tag on the left of the row.
        tag_h = tiles[0].shape[0]
        tag = np.full((tag_h, 150, 3), 245, np.uint8)
        cv2.putText(tag, row_title[:18], (6, tag_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        body = np.hstack(tiles)
        if tag.shape[0] != body.shape[0]:
            tag = cv2.resize(tag, (tag.shape[1], body.shape[0]))
        rows.append(np.hstack([tag, body]))
    return stack_rows(rows)


def choose_roles(images: dict[str, np.ndarray]) -> dict[str, str]:
    """Line, saliency, and failure examples used in the paper and the demo.

    Johanneskirche has long vertical mullions, obama is the person, and
    Sanfrancisco is the busy scene. Other names are used when that subset
    is not the one on disk.
    """
    preferred = {
        "lines": "Johanneskirche",
        "saliency": "obama",
        "failure": "Sanfrancisco",
        "height": "waterfall",
        "enlarge": "butterfly",
    }
    if all(name in images for name in preferred.values()):
        return preferred
    line_score = {}
    saliency_peak = {}
    edge_fraction = {}
    for name, image in images.items():
        lum = luminance(image)
        lines = line_protection(lum, "vertical")
        sal = protection_maps(image)["saliency"]
        edges = cv2.Canny((lum * 255).astype(np.uint8), 80, 160)
        line_score[name] = float((lines > 0.75).mean())
        saliency_peak[name] = float(sal.max() / (sal.mean() + 1e-6))
        edge_fraction[name] = float(np.count_nonzero(edges)) / edges.size
    names = list(images)
    return {
        "lines": max(line_score, key=line_score.get),
        "saliency": max(saliency_peak, key=saliency_peak.get),
        "failure": max(edge_fraction, key=edge_fraction.get),
        "height": names[0],
        "enlarge": names[min(1, len(names) - 1)],
    }


def write_summary(rows: list[dict], path: Path) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["setting"], row["method"]), []).append(row)
    lines = ["setting,method,n,mean_saliency_removed,mean_line_crossings_per_100,mean_runtime_s,all_sizes_ok"]
    for key in sorted(groups):
        group = groups[key]
        sal_vals = [float(r["saliency_removed_fraction"]) for r in group if r["saliency_removed_fraction"] != ""]
        line_vals = [float(r["line_crossings_per_100"]) for r in group if r["line_crossings_per_100"] != ""]
        run_vals = [float(r["runtime_s"]) for r in group]
        sal = sum(sal_vals) / len(sal_vals) if sal_vals else float("nan")
        crossed = sum(line_vals) / len(line_vals) if line_vals else float("nan")
        runtime = sum(run_vals) / len(run_vals)
        sizes_ok = all(int(r["size_ok"]) == 1 for r in group)
        lines.append(
            f"{key[0]},{key[1]},{len(group)},{sal:.6f},{crossed:.4f},{runtime:.4f},{int(sizes_ok)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths = discover_images()
    images = {path.stem: limit_long_side(load_rgb(path), MAX_SIDE) for path in paths}
    roles = choose_roles(images)
    (RESULT_DIR / "roles.txt").write_text(
        "\n".join(f"{role}={name}" for role, name in roles.items()) + "\n",
        encoding="utf-8",
    )
    print("roles", roles, flush=True)

    rows: list[dict] = []
    outputs: dict[tuple[str, str, str], np.ndarray] = {}

    width_75_methods = ["scale", "crop", "classic", "forward", "forward_saliency", "forward_lines", "ours"]
    width_50_methods = ["scale", "crop", "classic", "ours"]

    for name, image in images.items():
        height, width = image.shape[:2]
        jobs = [("width_75", (height, max(2, int(round(width * 0.75)))), width_75_methods)]
        jobs.append(("width_50", (height, max(2, int(round(width * 0.50)))), width_50_methods))
        if name == roles["height"]:
            jobs.append(("height_75", (max(2, int(round(height * 0.75))), width), ["scale", "classic", "ours"]))
        if name == roles["enlarge"]:
            jobs.append(("enlarge_150", (height, max(2, int(round(width * 1.50)))), ["scale", "ours"]))
        for setting, target, methods in jobs:
            for method in methods:
                print(f"{name} {setting} {method}", flush=True)
                output, info, elapsed = run_one(image, target, method)
                rows.append(row_for(name, setting, method, image, output, info, elapsed, target))
                outputs[(name, setting, method)] = output
                print(f"  {elapsed:.1f}s -> {output.shape[0]}x{output.shape[1]}", flush=True)

    csv_path = RESULT_DIR / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, RESULT_DIR / "summary.csv")

    compare_75 = []
    compare_50 = []
    for name, image in images.items():
        compare_75.append((name, [
            ("Original", image),
            ("Scale", outputs[(name, "width_75", "scale")]),
            ("Crop", outputs[(name, "width_75", "crop")]),
            ("Classic", outputs[(name, "width_75", "classic")]),
            ("Ours", outputs[(name, "width_75", "ours")]),
        ]))
        compare_50.append((name, [
            ("Original", image),
            ("Scale", outputs[(name, "width_50", "scale")]),
            ("Crop", outputs[(name, "width_50", "crop")]),
            ("Classic", outputs[(name, "width_50", "classic")]),
            ("Ours", outputs[(name, "width_50", "ours")]),
        ]))
    save_rgb(RESULT_DIR / "fig_compare_75.png", contact_sheet(compare_75))
    save_rgb(RESULT_DIR / "fig_compare_50.png", contact_sheet(compare_50))
    paper_names = [roles["lines"], roles["saliency"], roles["failure"], roles["height"]]
    by_name_75 = {name: row for name, row in compare_75}
    by_name_50 = {name: row for name, row in compare_50}
    save_rgb(
        RESULT_DIR / "fig_compare_75_paper.png",
        contact_sheet([(name, by_name_75[name]) for name in paper_names]),
    )
    save_rgb(
        RESULT_DIR / "fig_compare_50_paper.png",
        contact_sheet([(name, by_name_50[name]) for name in paper_names]),
    )

    height_name = roles["height"]
    height_image = images[height_name]
    save_rgb(RESULT_DIR / "fig_height.png", contact_sheet([(height_name, [
        ("Original", height_image),
        ("Scale", outputs[(height_name, "height_75", "scale")]),
        ("Classic", outputs[(height_name, "height_75", "classic")]),
        ("Ours", outputs[(height_name, "height_75", "ours")]),
    ])]))

    enlarge_name = roles["enlarge"]
    enlarge_image = images[enlarge_name]
    save_rgb(RESULT_DIR / "fig_enlarge.png", contact_sheet([(enlarge_name, [
        ("Original", enlarge_image),
        ("Scale", outputs[(enlarge_name, "enlarge_150", "scale")]),
        ("Ours", outputs[(enlarge_name, "enlarge_150", "ours")]),
    ])]))

    failure_name = roles["failure"]
    save_rgb(RESULT_DIR / "fig_failure.png", contact_sheet([(failure_name + " 50%", [
        ("Original", images[failure_name]),
        ("Scale", outputs[(failure_name, "width_50", "scale")]),
        ("Crop", outputs[(failure_name, "width_50", "crop")]),
        ("Classic", outputs[(failure_name, "width_50", "classic")]),
        ("Ours", outputs[(failure_name, "width_50", "ours")]),
    ])]))

    line_name = roles["lines"]
    line_image = images[line_name]
    save_rgb(RESULT_DIR / "fig_ablation.png", contact_sheet([(line_name, [
        ("Forward", outputs[(line_name, "width_75", "forward")]),
        ("+Saliency", outputs[(line_name, "width_75", "forward_saliency")]),
        ("+Lines", outputs[(line_name, "width_75", "forward_lines")]),
        ("Full", outputs[(line_name, "width_75", "ours")]),
        ("Classic", outputs[(line_name, "width_75", "classic")]),
    ])]))

    maps = protection_maps(line_image)
    seam = next_vertical_seam(line_image, method="ours", weights=WEIGHTS["ours"])
    save_rgb(RESULT_DIR / "fig_maps.png", contact_sheet([(line_name, [
        ("Input", line_image),
        ("Saliency", colorize(maps["saliency"])),
        ("Lines", colorize(maps["lines_vertical"])),
        ("Next seam", draw_vertical_seam(line_image, seam)),
    ])]))
    print("wrote", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()

"""The unattended learned-saliency run. We chose not to overwrite metrics.csv, metrics_n30.csv, or main.tex."""

from __future__ import annotations

import csv
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_experiments  # noqa: E402
from report_numbers import analyze_learned  # noqa: E402
from run_experiments import contact_sheet, load_rgb, save_rgb  # noqa: E402
from run_n30 import (  # noqa: E402
    ORIGINAL,
    SAMPLE_DIR,
    SEED,
    TUNING,
    archive_pngs,
    carve,
    choose_sample,
    extract_sample,
    load_working,
    old_rows,
)
from src.baselines import limit_long_side  # noqa: E402
from src.energy import learned_saliency, luminance, spectral_residual_saliency  # noqa: E402
from src.metrics import line_crossings_per_100, lsd_crossings_per_100  # noqa: E402
from src.retarget import DEFAULT_WEIGHTS, LEARNED_W_L, LEARNED_W_S, retarget  # noqa: E402

RESULT = ROOT / "experiments" / "results"
LOG_PATH = RESULT / "overnight_log.txt"
STATUS_PATH = RESULT / "overnight_status.txt"
CSV8 = RESULT / "metrics_learned_8.csv"
CSV30 = RESULT / "metrics_learned_n30.csv"
PAPER = ROOT / "paper"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = ROOT / "experiments" / f"backup_{STAMP}"

SEAM_JOBS = (
    ("classic", "classic", "spectral", (0.0, 0.0)),
    ("forward", "ours", "spectral", (0.0, 0.0)),
    ("forward_lines", "ours", "spectral", (0.0, LEARNED_W_L)),
    ("ours_spectral", "ours", "spectral", DEFAULT_WEIGHTS),
    ("forward_saliency_learned", "ours", "learned", (LEARNED_W_S, 0.0)),
    ("ours_learned", "ours", "learned", (LEARNED_W_S, LEARNED_W_L)),
)
PNG_METHODS = (
    "original",
    "scale",
    "crop",
    "classic",
    "forward",
    "forward_lines",
    "ours_spectral",
    "ours_learned",
    "forward_saliency_learned",
)
CSV_FIELDS = (
    "image", "setting", "method", "w_s", "w_l", "saliency_source",
    "target_h", "target_w", "out_h", "out_w", "size_ok",
    "spectral_mass", "learned_mass", "hough", "lsd", "learned_blob",
    "seam_pixels", "runtime_s",
)

STATE = {"failed": "", "warnings": [], "created": [], "steps": {}, "printed": ""}


def log(message: str) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(message, flush=True)


def warn(message: str) -> None:
    STATE["warnings"].append(message)
    log("WARNING " + message)


def created(path: Path) -> None:
    STATE["created"].append(str(path))


def backup() -> None:
    if BACKUP.exists():
        raise SystemExit(f"backup already exists: {BACKUP}")
    paper_dest = BACKUP / "paper"
    paper_dest.mkdir(parents=True)
    for name in ("main.tex", "main.bib", "preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"):
        source = PAPER / name
        if source.exists():
            shutil.copy2(source, paper_dest / name)
    shutil.copytree(RESULT, BACKUP / "results")
    log(f"backup {BACKUP}")


def assert_frozen() -> None:
    if DEFAULT_WEIGHTS != (0.5, 0.8) or LEARNED_W_S != 0.25 or LEARNED_W_L != 0.8 or SEED != 604:
        raise SystemExit(
            f"frozen settings moved: weights {DEFAULT_WEIGHTS} learned {LEARNED_W_S} {LEARNED_W_L} seed {SEED}"
        )
    log("frozen spectral (0.5, 0.8), LEARNED_W_S 0.25, w_l 0.8, seed 604, max side 640")


def log_versions() -> None:
    import cv2
    import numpy
    import onnxruntime
    import rembg

    weight = Path.home() / ".u2net" / "u2netp.onnx"
    alt = Path.home() / ".rembg" / "models" / "u2netp" / "u2netp.onnx"
    found = weight if weight.exists() else alt
    size = found.stat().st_size if found.exists() else -1
    log(f"numpy {numpy.__version__}")
    log(f"opencv {cv2.__version__}")
    log(f"rembg {rembg.__version__}")
    log(f"onnxruntime {onnxruntime.__version__}")
    log(f"u2netp {found} bytes {size}")
    STATE["versions"] = {
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "rembg": rembg.__version__,
        "onnxruntime": onnxruntime.__version__,
        "u2netp_bytes": size,
    }


def load_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["image"], row["setting"], row["method"]) for row in csv.DictReader(handle)}


def append_row(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
            created(path)
        writer.writerow(row)


def png_path(setting: str, image: str, method: str) -> Path:
    return RESULT / "outputs" / setting / image / f"{method}.png"


def write_png(path: Path, image, expected_hw: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(path, image)
    loaded = load_rgb(path)
    if loaded.shape[:2] != expected_hw or image.shape[:2] != expected_hw:
        warn(
            f"size mismatch {path.name} expected {expected_hw[0]}x{expected_hw[1]} "
            f"memory {image.shape[0]}x{image.shape[1]} disk {loaded.shape[0]}x{loaded.shape[1]}"
        )
    else:
        log(f"saved {path.relative_to(RESULT)} {expected_hw[0]}x{expected_hw[1]}")
    created(path)


def save_maps(name: str, image) -> None:
    dest = RESULT / "maps"
    dest.mkdir(parents=True, exist_ok=True)
    spectral_path = dest / f"{name}_spectral.png"
    learned_path = dest / f"{name}_learned.png"
    if spectral_path.exists() and learned_path.exists():
        return
    import cv2
    import numpy as np

    spectral = spectral_residual_saliency(luminance(image))
    learned = learned_saliency(image)
    for path, values in ((spectral_path, spectral), (learned_path, learned)):
        gray = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(path), gray):
            raise RuntimeError(f"could not write {path}")
        created(path)
    log(f"maps {name} {image.shape[0]}x{image.shape[1]}")


def mass(info: dict, removed: str, total: str) -> float:
    denom = float(info.get(total, 0.0))
    if denom <= 0.0:
        return 0.0
    return float(info.get(removed, 0.0)) / denom


def carve_seam(image, setting: str, method: str, retarget_method: str, source: str, weights):
    import time

    height, width = image.shape[:2]
    fraction = 0.75 if setting == "width_75" else 0.50
    target = (height, max(2, int(round(width * fraction))))
    started = time.perf_counter()
    output, info = retarget(
        image,
        target,
        method=retarget_method,
        weights=weights,
        return_info=True,
        saliency_source=source,
        score_both=True,
    )
    elapsed = time.perf_counter() - started
    pixels = int(info.get("seam_pixels", 0))
    blob_hits = int(info.get("learned_blob_hits", 0))
    row = {
        "image": "",
        "setting": setting,
        "method": method,
        "w_s": weights[0],
        "w_l": weights[1],
        "saliency_source": source,
        "target_h": target[0],
        "target_w": target[1],
        "out_h": output.shape[0],
        "out_w": output.shape[1],
        "size_ok": int(output.shape[:2] == target),
        "spectral_mass": f"{mass(info, 'spectral_removed', 'spectral_total'):.6f}",
        "learned_mass": f"{mass(info, 'learned_removed', 'learned_total'):.6f}",
        "hough": f"{line_crossings_per_100(info):.4f}",
        "lsd": f"{lsd_crossings_per_100(info):.4f}",
        "learned_blob": f"{(blob_hits / pixels) if pixels else 0.0:.6f}",
        "seam_pixels": pixels,
        "runtime_s": f"{elapsed:.4f}",
    }
    return output, row, elapsed


def ensure_image(name: str, setting: str, image, keys: set, csv_path: Path) -> None:
    height, width = image.shape[:2]
    fraction = 0.75 if setting == "width_75" else 0.50
    target = (height, max(2, int(round(width * fraction))))
    originals = {
        "original": (image, image.shape[:2]),
    }
    if not png_path(setting, name, "original").exists():
        write_png(png_path(setting, name, "original"), image, image.shape[:2])
    for baseline in ("scale", "crop"):
        path = png_path(setting, name, baseline)
        if path.exists():
            continue
        output, _info = retarget(image, target, method=baseline, return_info=True)
        write_png(path, output, target)
        originals[baseline] = (output, target)
    for method, retarget_method, source, weights in SEAM_JOBS:
        path = png_path(setting, name, method)
        if path.exists() and (name, setting, method) in keys:
            continue
        log(f"{name} {setting} {method}")
        output, row, elapsed = carve_seam(image, setting, method, retarget_method, source, weights)
        row["image"] = name
        write_png(path, output, target)
        if (name, setting, method) not in keys:
            append_row(csv_path, row)
            keys.add((name, setting, method))
        log(f"  {elapsed:.1f}s spectral {row['spectral_mass']} learned {row['learned_mass']} blob {row['learned_blob']}")


def step1() -> None:
    saved = old_rows()
    anchor = saved[("brick_house", "width_75", "classic")]
    if (anchor["saliency_removed_fraction"], anchor["line_crossings_per_100"]) != ("0.153154", "3.2319"):
        raise SystemExit("metrics.csv anchor is not 0.153154 / 3.2319")
    for name in ORIGINAL:
        carve(name, saved)
    log("STEP 1 matched metrics.csv including brick_house classic 0.153154 / 3.2319")


def images_for(names: list[str]) -> dict:
    found = {}
    for name in names:
        image = load_working(name)
        if max(image.shape[:2]) > 640:
            raise SystemExit(f"{name} exceeded the 640 cap")
        save_maps(name, image)
        found[name] = image
    return found


def step2() -> None:
    keys = load_keys(CSV8)
    images = images_for(list(ORIGINAL))
    for name, image in images.items():
        for setting in ("width_75", "width_50"):
            ensure_image(name, setting, image, keys, CSV8)
    log(f"STEP 2 rows {len(load_keys(CSV8))}")


def sheet_from_pngs(names, setting: str, methods, titles) -> None:
    run_experiments.PANEL_H = 200
    records = []
    for name in names:
        panels = []
        for method, title in zip(methods, titles):
            panels.append((title, load_rgb(png_path(setting, name, method))))
        records.append((name, panels))
    return contact_sheet(records)


def step3() -> None:
    methods = ("original", "scale", "classic", "ours_spectral", "ours_learned")
    titles = ("Original", "Scale", "Classic", "Spectral", "Learned")
    first = RESULT / "fig_learned_sf50.png"
    second = RESULT / "fig_learned_50_set.png"
    save_rgb(first, sheet_from_pngs(("Sanfrancisco",), "width_50", methods, titles))
    save_rgb(second, sheet_from_pngs(("brick_house", "child", "obama"), "width_50", methods, titles))
    created(first)
    created(second)
    log(f"STEP 3 wrote {first.name} and {second.name}")


def expected_sample() -> list[str]:
    summary = (RESULT / "n30_summary.txt").read_text(encoding="utf-8")
    for line in summary.splitlines():
        if line.startswith("sample 30:"):
            return [item.strip() for item in line.split(":", 1)[1].split(",")]
    raise SystemExit("n30_summary.txt has no sample line")


def step4() -> None:
    members = archive_pngs()
    sample = choose_sample(list(members))
    previous = expected_sample()
    if sample != previous:
        raise SystemExit("seed 604 sample does not match n30_summary.txt")
    held = [name for name in TUNING if name in sample]
    if held:
        raise SystemExit(f"tuning images were drawn: {held}")
    log("Johanneskirche and obama are not in the seed-604 sample")
    list_path = RESULT / "n30_images.txt"
    list_path.write_text("seed 604\n" + "\n".join(sample) + "\n", encoding="utf-8")
    created(list_path)
    extract_sample(members, sample)
    keys = load_keys(CSV30)
    done = len(keys)
    if CSV8.exists():
        with CSV8.open(newline="", encoding="utf-8") as handle:
            times = [float(row["runtime_s"]) for row in csv.DictReader(handle) if row["setting"] == "width_75"]
        mean_s = sum(times) / len(times) if times else 15.0
    else:
        mean_s = 15.0
    remaining = 30 * len(SEAM_JOBS) - done
    log(f"STEP 4 runtime estimate {remaining * mean_s / 60.0:.1f} minutes for {remaining} remaining seam calls, from the 75% mean {mean_s:.2f}s")
    images = images_for(sample)
    for name in sample:
        ensure_image(name, "width_75", images[name], keys, CSV30)
    run_experiments.PANEL_H = 180
    methods = ("original", "classic", "ours_spectral", "ours_learned")
    titles = ("Original", "Classic", "Spectral", "Learned")
    sheet_dir = RESULT / "sheets_n30"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in sample:
        sheet = sheet_from_pngs((name,), "width_75", methods, titles)
        path = sheet_dir / f"{name}.png"
        save_rgb(path, sheet)
        created(path)
        records.append((name, [
            (title, load_rgb(png_path("width_75", name, method)))
            for method, title in zip(methods, titles)
        ]))
    run_experiments.PANEL_H = 72
    overview = RESULT / "fig_n30_overview.png"
    save_rgb(overview, contact_sheet(records))
    created(overview)
    log(f"STEP 4 rows {len(load_keys(CSV30))} overview {overview.name}")


def step5() -> str:
    blocks = []
    if not CSV8.exists() or not CSV30.exists():
        raise SystemExit("learned csv missing")
    with CSV8.open(newline="", encoding="utf-8") as handle:
        rows8 = list(csv.DictReader(handle))
    with CSV30.open(newline="", encoding="utf-8") as handle:
        rows30 = list(csv.DictReader(handle))
    payloads = {}
    for setting in ("width_75", "width_50"):
        text, payload = analyze_learned(rows8, setting)
        blocks.append(text)
        payloads[("8", setting)] = payload
        log(text)
    present = {row["image"] for row in rows30}
    if any(name in present for name in TUNING):
        raise SystemExit("tuning images are in metrics_learned_n30.csv")
    text, payload = analyze_learned(rows30, "width_75")
    blocks.append(text)
    payloads[("30", "width_75")] = payload
    log(text)
    sweep_text = sweep_lines()
    blocks.append(sweep_text)
    log(sweep_text)
    printed = "\n".join(blocks)
    STATE["printed"] = printed
    STATE["payloads"] = payloads
    return printed


def sweep_lines() -> str:
    path = RESULT / "sweep_learned_ws.csv"
    if not path.exists():
        raise SystemExit("sweep_learned_ws.csv is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = ["sweep check from sweep_learned_ws.csv"]
    blobs = []
    for image in ("Johanneskirche", "obama"):
        forward = next(row for row in rows if row["image"] == image and row["setting"] == "forward_only")
        half = float(forward["learned_mass"]) / 2.0
        lines.append(f"sweep {image} forward learned_mass {float(forward['learned_mass']):.4f} half {half:.4f}")
        for row in rows:
            if row["image"] != image or row["setting"] != "learned_full":
                continue
            mass_value = float(row["learned_mass"])
            blob = float(row["blob"])
            blobs.append(blob)
            lines.append(
                f"sweep {image} w_s {float(row['w_s']):.2f} learned_mass {mass_value:.4f} "
                f"below_half {mass_value < half} blob {blob:.4f}"
            )
    lines.append(f"sweep blob all_zero {all(value == 0.0 for value in blobs)} n {len(blobs)}")
    lines.append("sweep frozen w_s 0.25")
    return "\n".join(lines) + "\n"


def latex_table(payload: dict, caption: str, label: str) -> str:
    header = "Method & Spectral & Learned & Hough & LSD & Blob \\\\"
    body = []
    for method, title in (
        ("classic", "Classic"),
        ("forward", "Forward only"),
        ("forward_lines", "Forward + lines"),
        ("ours_spectral", "Ours spectral"),
        ("forward_saliency_learned", "Learned saliency only"),
        ("ours_learned", "Ours learned"),
    ):
        cells = payload["cells"]
        body.append(
            f"{title} & {cells[(method, 'spectral_mass')]} & {cells[(method, 'learned_mass')]} "
            f"& {cells[(method, 'hough')]} & {cells[(method, 'lsd')]} & {cells[(method, 'learned_blob')]} \\\\"
        )
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\scriptsize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def verdict(test: dict) -> str:
    if test["lower"] > test["higher"] and float(test["p"]) < 0.05:
        outcome = "The sign test favors ours learned on this map."
    else:
        outcome = "This does not show that learned-chosen seams are better on this map."
    return (
        f"Against {test['right_name']} under the {test['map_name']} map, ours learned is lower on "
        f"{test['lower']} images, higher on {test['higher']}, and tied on {test['tied']} "
        f"($p = {test['p']}$). {outcome}"
    )


def step5b() -> None:
    printed = STATE["printed"]
    versions = STATE["versions"]
    eight = STATE["payloads"][("8", "width_75")]
    eight50 = STATE["payloads"][("8", "width_50")]
    thirty = STATE["payloads"][("30", "width_75")]
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "main_learned" in source:
        raise SystemExit("refusing to read a file that already mentions main_learned as main.tex")
    text = source
    replacements = {
        "plus two maps that use no learned weights: spectral-residual saliency, and a line map whose weight depends on the direction of the seam.":
        "plus two maps: spectral-residual saliency, and a line map whose weight depends on the direction of the seam. A pretrained salient-object model is an optional comparison and is not the default cost.",
        "There is no trained retargeting network and no learned saliency model.\nThe implementation uses NumPy and OpenCV,":
        "There is no trained retargeting network. A pretrained salient-object model is an optional comparison and is not the default cost.\n"
        f"The implementation uses NumPy {versions['numpy']}, OpenCV {versions['opencv']}, and, only for that comparison, rembg {versions['rembg']} with onnxruntime {versions['onnxruntime']} and the u2netp checkpoint ({versions['u2netp_bytes']} bytes).",
        "Saliency raises the cost of pixels that the spectral residual marks. Those pixels are often edges, not a detected person.":
        "The default saliency term raises the cost of pixels that the spectral residual marks. Those pixels are often edges, not a detected person. The optional learned map is a pretrained salient-object model.",
        "No second saliency model and no summed forward-energy of the chosen seams is reported.":
        "A pretrained saliency model is reported as an optional comparison. No summed forward-energy of the chosen seams is reported.",
        "Spectral residual saliency responds strongly to edges.\nIt is not a subject detector.":
        "Spectral residual saliency responds strongly to edges and is not a subject detector. The optional learned comparison is a pretrained salient-object model \\cite{qin2020u2net}.",
        "The weights were chosen on two of the eight test images, and they were not swept.":
        "The spectral weights were chosen on two of the eight test images, and they were not swept.",
    }
    for old, new in replacements.items():
        if old not in text:
            raise SystemExit(f"paper sentence missing: {old[:80]}")
        text = text.replace(old, new, 1)
    tests = " ".join(verdict(test) for test in thirty["tests"].values())
    spec_cell = thirty["cells"][("ours_spectral", "spectral_mass")]
    spec_on_learned = thirty["cells"][("ours_spectral", "learned_mass")]
    learned_on_spec = thirty["cells"][("ours_learned", "spectral_mass")]
    learned_cell = thirty["cells"][("ours_learned", "learned_mass")]
    subsection = "\n".join([
        r"\subsection{Learned saliency comparison}",
        r"The learned map is U$^2$-Net-small, loaded through \texttt{rembg} as the \texttt{u2netp} checkpoint \cite{qin2020u2net}.",
        "It is pretrained. We did not train it.",
        r"Its output is near-binary, so in the seam cost it acts like a mask.",
        r"The spectral default remains $w_s = 0.5$ and $w_l = 0.8$.",
        r"For the learned source, $w_l = 0.8$ and $w_s = 0.25$.",
        r"That $0.25$ was the smallest of the five weights $\{0.25, 0.5, 1, 2, 4\}$ tested on Johanneskirche and \texttt{obama} only.",
        "Every tested weight put the learned-mass fraction below half the forward-only value on both images, and the fraction of seam pixels above $0.5$ on the learned map was 0 at every tested weight.",
        r"\texttt{obama} is one of the two tuning images, and its learned map marks the car door rather than the people, so this weight is not evidence that people are protected.",
        "Ours learned always includes the line weight.",
        r"It is compared with forward plus lines and with ours spectral, which also use $w_l = 0.8$.",
        "It is not compared with forward-only, because that comparison confounds the line weight.",
        r"Ours learned uses $w_s = 0.25$ and ours spectral uses $w_s = 0.5$, so the comparison is also not a pure swap of the map.",
        "Each source is expected to score better under its own map.",
        f"On the 30-image sample, ours spectral scores {spec_cell} under the spectral map and {spec_on_learned} under the learned map.",
        f"Ours learned scores {learned_on_spec} under the spectral map and {learned_cell} under the learned map.",
        tests,
        "Tables~\\ref{tab:learned8} and~\\ref{tab:learned30} are the 75\\% width cross-evaluation.",
        "Table~\\ref{tab:learned850} is the same six methods on the original eight at 50\\% width.",
        "Blob is the fraction of seam pixels whose learned-map value is above 0.5.",
        latex_table(
            eight,
            "Original eight at 75\\% width. Spectral and Learned are fractions of that map deleted. Hough and LSD are crossings per 100 seam pixels. Blob is the fraction of seam pixels with learned value above 0.5. Ours learned uses $w_s = 0.25$ and $w_l = 0.8$. Ours spectral uses $w_s = 0.5$ and $w_l = 0.8$.",
            "tab:learned8",
        ),
        latex_table(
            thirty,
            "Seeded 30-image sample at 75\\% width, seed 604. Johanneskirche and \\texttt{obama} were not drawn. Columns match Table~\\ref{tab:learned8}. Ours learned is not claimed to be better unless the sign tests say so.",
            "tab:learned30",
        ),
        latex_table(
            eight50,
            "Original eight at 50\\% width. Same columns and weights as Table~\\ref{tab:learned8}.",
            "tab:learned850",
        ),
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{fig_learned_sf50.png}",
        r"\caption{\texttt{Sanfrancisco} at 50\% width. Columns are the original, uniform scale, classic seams, ours spectral, and ours learned. Any impression that the heart is preserved is not a measurement.}",
        r"\label{fig:learnedsf}",
        r"\end{figure*}",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{fig_learned_50_set.png}",
        r"\caption{\texttt{brick\_house}, \texttt{child}, and \texttt{obama} at 50\% width, with the same columns as Figure~\ref{fig:learnedsf}. These sheets are impressions, not measurements. The \texttt{obama} learned map marks the car door rather than the people.}",
        r"\label{fig:learnedset}",
        r"\end{figure*}",
        "",
    ])
    anchor = r"\subsection{Limitations}"
    if anchor not in text:
        raise SystemExit("limitations anchor missing")
    text = text.replace(anchor, subsection + "\n" + anchor, 1)
    floated = []
    for chunk in (subsection, *replacements.values()):
        floated.append(chunk)
    blob = "\n".join(floated)
    import re
    missing = []
    for number in re.findall(r"\d+\.\d+", blob):
        if number not in printed and number not in STATE.get("header", ""):
            missing.append(number)
    if missing:
        raise SystemExit(f"draft number not in the printed output: {missing[:12]}")
    destination = PAPER / "main_learned.tex"
    destination.write_text(text, encoding="utf-8")
    created(destination)
    log(f"STEP 5b wrote {destination}")


def compile_paper() -> None:
    import io
    import urllib.request
    import zipfile

    binary = PAPER / "tectonic.exe"
    if not binary.exists():
        url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ee604"}), timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extract("tectonic.exe", PAPER)
    import subprocess
    completed = subprocess.run(
        [str(binary), "--outdir", str(PAPER), "main_learned.tex"],
        cwd=PAPER,
        text=True,
        capture_output=True,
    )
    log(completed.stdout[-2000:])
    if completed.returncode != 0:
        log(completed.stderr[-2000:])
        raise SystemExit(f"tectonic exit {completed.returncode}")
    binary.unlink(missing_ok=True)
    log("compiled main_learned.pdf")


def step6(compile_ok: bool) -> None:
    lines = [
        f"stamp {STAMP}",
        f"backup {BACKUP}",
        f"failed step: {STATE['failed'] or 'none'}",
        "steps:",
    ]
    for name, status in STATE["steps"].items():
        lines.append(f"  {name}: {status}")
    lines.append("warnings:")
    if STATE["warnings"]:
        lines.extend(f"  {item}" for item in STATE["warnings"])
    else:
        lines.append("  none")
    lines.append("printed tables:")
    lines.append(STATE.get("printed") or "(none)")
    lines.append("files created:")
    if STATE["created"]:
        lines.extend(f"  {item}" for item in STATE["created"])
    else:
        lines.append("  none")
    lines.append(f"compile {'yes' if compile_ok else 'no'}")
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"STEP 6 wrote {STATUS_PATH}")


def make_zip() -> Path:
    destination = ROOT / "experiments" / f"results_{STAMP}.zip"
    if destination.exists():
        destination.unlink()
    shutil.make_archive(destination.with_suffix(""), "zip", RESULT)
    log(f"zip {destination} bytes {destination.stat().st_size}")
    return destination


def run_called(name: str, fn, retry: bool = True) -> bool:
    attempts = 2 if retry else 1
    for attempt in range(1, attempts + 1):
        try:
            fn()
            STATE["steps"][name] = "finished"
            log(f"{name} finished")
            return True
        except SystemExit as exc:
            STATE["steps"][name] = f"failed: {exc}"
            STATE["failed"] = name
            log(f"{name} failed: {exc}")
            return False
        except Exception as exc:
            log(f"{name} attempt {attempt} error: {exc}\n{traceback.format_exc()}")
            if attempt == attempts:
                STATE["steps"][name] = f"failed: {exc}"
                STATE["failed"] = name
                return False
    return False


def main() -> None:
    backup()
    assert_frozen()
    log_versions()
    STATE["header"] = "\n".join([
        "0.5", "0.8", "0.25", "604", "640",
        STATE["versions"]["numpy"],
        STATE["versions"]["opencv"],
        STATE["versions"]["rembg"],
        STATE["versions"]["onnxruntime"],
        str(STATE["versions"]["u2netp_bytes"]),
    ])
    ok1 = run_called("STEP 1", step1, retry=False)
    ok2 = ok3 = ok4 = ok5 = ok5b = False
    if ok1:
        ok2 = run_called("STEP 2", step2)
        if ok2:
            ok3 = run_called("STEP 3", step3)
        ok4 = run_called("STEP 4", step4)
        if ok2 and ok4:
            ok5 = run_called("STEP 5", step5, retry=False)
        if ok1 and ok2 and ok3 and ok4 and ok5:
            ok5b = run_called("STEP 5b", step5b, retry=False)
    compile_ok = False
    if ok5b:
        compile_ok = run_called("COMPILE", compile_paper)
    step6(compile_ok)
    make_zip()


if __name__ == "__main__":
    main()

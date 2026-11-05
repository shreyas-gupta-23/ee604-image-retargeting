"""Print the numbers quoted in the write-up from the saved experiment CSV.

Reads ``experiments/results/metrics.csv``. Does not rerun seam carving.
The sample standard deviation is the n-1 definition. Hybrid is not scored
here: the write-up judges that schedule by eye only.
"""

from __future__ import annotations

import csv
import statistics
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "results" / "metrics.csv"
N30_PATH = ROOT / "experiments" / "results" / "metrics_n30.csv"

# The two images used to choose w_s and w_l. The check drops both.
TUNING = ("Johanneskirche", "obama")

WIDTH_75 = (
    ("classic", "Classic"),
    ("forward", "Forward only"),
    ("forward_saliency", "Forward + saliency"),
    ("forward_lines", "Forward + lines"),
    ("ours", "Full model"),
)
WIDTH_50 = (
    ("classic", "Classic"),
    ("ours", "Full model"),
)
PER_IMAGE = (
    ("classic", "Classic"),
    ("forward", "Forward"),
    ("forward_saliency", "+Saliency"),
    ("forward_lines", "+Lines"),
    ("ours", "Full"),
)


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows, setting: str, method: str, skip: tuple[str, ...] = ()) -> list[dict[str, str]]:
    chosen = []
    for row in rows:
        if row["setting"] != setting or row["method"] != method:
            continue
        if row["image"] in skip:
            continue
        chosen.append(row)
    return chosen


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def print_summary(rows, setting: str, methods, skip: tuple[str, ...] = ()) -> None:
    label = setting if not skip else f"{setting} without {', '.join(skip)}"
    print(f"\n{label}")
    print(f"{'method':<22} {'n':>3} {'saliency':>16} {'lines':>16} {'seconds':>16}")
    for method, title in methods:
        group = select(rows, setting, method, skip)
        sal = [float(row["saliency_removed_fraction"]) for row in group]
        lines = [float(row["line_crossings_per_100"]) for row in group]
        seconds = [float(row["runtime_s"]) for row in group]
        sal_mean, sal_std = mean_std(sal)
        line_mean, line_std = mean_std(lines)
        sec_mean, sec_std = mean_std(seconds)
        print(
            f"{title:<22} {len(group):>3} "
            f"{sal_mean:7.3f} ({sal_std:5.3f}) "
            f"{line_mean:7.2f} ({line_std:5.2f}) "
            f"{sec_mean:7.1f} ({sec_std:4.1f})"
        )


def print_per_image(rows) -> None:
    order = []
    for row in rows:
        if row["setting"] == "width_75" and row["image"] not in order:
            order.append(row["image"])
    print("\nper-image width_75, saliency / lines")
    header = f"{'image':<16}" + "".join(f"{title:>16}" for _, title in PER_IMAGE)
    print(header)
    for image in order:
        cells = []
        mark = "*" if image in TUNING else ""
        for method, _title in PER_IMAGE:
            match = [
                row for row in rows
                if row["image"] == image and row["setting"] == "width_75" and row["method"] == method
            ]
            if len(match) != 1:
                raise SystemExit(f"expected one row for {image} {method}, found {len(match)}")
            row = match[0]
            sal = float(row["saliency_removed_fraction"])
            lines = float(row["line_crossings_per_100"])
            cells.append(f"{sal:.3f}/{lines:.2f}")
        body = "".join(f"{cell:>16}" for cell in cells)
        print(f"{image + mark:<16}{body}")
    print("* tuning image, dropped in the six-image check")


def print_lsd(rows) -> None:
    """LSD crossings per 100 seam pixels. The map is not inside the cost."""
    if not rows or "lsd_crossings_per_100" not in rows[0]:
        print("\nlsd column is not in the csv")
        return
    print("\nlsd crossings per 100 seam pixels (not used to choose seams)")
    for setting in ("width_75", "width_50"):
        print(f"\n{setting}")
        print(f"{'method':<22} {'n':>3} {'lsd':>16}")
        for method, title in PER_IMAGE:
            group = select(rows, setting, method)
            values = [float(row["lsd_crossings_per_100"]) for row in group if row["lsd_crossings_per_100"] != ""]
            if len(values) < 2:
                print(f"{title:<22} {len(values):>3}  (not enough rows)")
                continue
            mean, std = mean_std(values)
            print(f"{title:<22} {len(values):>3} {mean:7.2f} ({std:5.2f})")
        order = []
        for row in rows:
            if row["setting"] == setting and row["image"] not in order and row["method"] == "classic":
                order.append(row["image"])
        header = f"{'image':<16}" + "".join(f"{title:>12}" for _, title in PER_IMAGE)
        print(header)
        for image in order:
            cells = []
            classic = full = None
            for method, _title in PER_IMAGE:
                match = [
                    row for row in rows
                    if row["image"] == image and row["setting"] == setting and row["method"] == method
                ]
                if len(match) != 1 or match[0]["lsd_crossings_per_100"] == "":
                    raise SystemExit(f"missing lsd score for {image} {setting} {method}")
                value = float(match[0]["lsd_crossings_per_100"])
                cells.append(f"{value:.2f}")
                if method == "classic":
                    classic = value
                if method == "ours":
                    full = value
            body = "".join(f"{cell:>12}" for cell in cells)
            relation = "full < classic" if full < classic else "full >= classic"
            print(f"{image:<16}{body}   {relation}")


def binomial_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial test at p = 0.5."""
    if trials <= 0:
        return float("nan")
    probs = [comb(trials, k) / 2**trials for k in range(trials + 1)]
    observed = probs[successes]
    return sum(prob for prob in probs if prob <= observed + 1e-15)


def print_width50_per_image(rows) -> None:
    """Classic and full at 50% width: saliency / Hough / LSD."""
    order = []
    for row in rows:
        if row["setting"] == "width_50" and row["method"] == "classic" and row["image"] not in order:
            order.append(row["image"])
    print("\nper-image width_50, classic and full, saliency/hough/lsd")
    print(f"{'image':<16}{'classic':>20}{'full':>20}   saliency   hough")
    sal_lower = hough_lower = lsd_lower = 0
    for image in order:
        cells = []
        values = {}
        for method in ("classic", "ours"):
            match = [
                row for row in rows
                if row["image"] == image and row["setting"] == "width_50" and row["method"] == method
            ]
            if len(match) != 1:
                raise SystemExit(f"expected one width_50 row for {image} {method}, found {len(match)}")
            row = match[0]
            sal = float(row["saliency_removed_fraction"])
            hough = float(row["line_crossings_per_100"])
            lsd = float(row["lsd_crossings_per_100"])
            values[method] = (sal, hough, lsd)
            cells.append(f"{sal:.3f}/{hough:.2f}/{lsd:.2f}")
        if values["ours"][0] < values["classic"][0]:
            sal_lower += 1
        if values["ours"][1] < values["classic"][1]:
            hough_lower += 1
        if values["ours"][2] < values["classic"][2]:
            lsd_lower += 1
        sal_rel = "full < classic" if values["ours"][0] < values["classic"][0] else "full >= classic"
        hough_rel = "full < classic" if values["ours"][1] < values["classic"][1] else "full >= classic"
        body = "".join(f"{cell:>20}" for cell in cells)
        print(f"{image:<16}{body}   {sal_rel:<16} {hough_rel}")
    print(
        f"full lower than classic on saliency {sal_lower}/{len(order)}, "
        f"hough {hough_lower}/{len(order)}, lsd {lsd_lower}/{len(order)}"
    )


def print_n30_saliency() -> None:
    if not N30_PATH.exists():
        print("\nmetrics_n30.csv is missing")
        return
    with N30_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    order = []
    for row in rows:
        if row["image"] not in order:
            order.append(row["image"])
    headline = [name for name in order if name not in TUNING]
    by = {(row["image"], row["method"]): row for row in rows}
    print(f"\nn30 width_75 saliency, n={len(headline)}, seed sample in metrics_n30.csv")
    print(f"{'method':<22} {'n':>3} {'saliency':>16}")
    for method, title in PER_IMAGE:
        values = [float(by[(name, method)]["saliency_removed_fraction"]) for name in headline]
        mean, std = mean_std(values)
        print(f"{title:<22} {len(values):>3} {mean:7.3f} ({std:5.3f})")
    lower = higher = tied = 0
    for name in headline:
        classic = float(by[(name, "classic")]["saliency_removed_fraction"])
        full = float(by[(name, "ours")]["saliency_removed_fraction"])
        if full < classic:
            lower += 1
        elif full > classic:
            higher += 1
        else:
            tied += 1
    trials = lower + higher
    print(
        f"saliency sign test, full versus classic, lower/higher/tied "
        f"{lower}/{higher}/{tied}, "
        f"two-sided binomial p={binomial_two_sided(lower, trials):.4e} "
        f"on {trials} non-ties"
    )


LEARNED_METHODS = (
    ("classic", "Classic"),
    ("forward", "Forward only"),
    ("forward_lines", "Forward + lines"),
    ("ours_spectral", "Ours spectral"),
    ("forward_saliency_learned", "Learned saliency only"),
    ("ours_learned", "Ours learned"),
)
LEARNED_COLUMNS = (
    ("spectral_mass", "Spectral", 3),
    ("learned_mass", "Learned", 3),
    ("hough", "Hough", 2),
    ("lsd", "LSD", 2),
    ("learned_blob", "Blob", 3),
)


def format_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.4e}"
    return f"{p:.4f}"


def _learned_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze_learned(rows: list[dict[str, str]], setting: str, skip: tuple[str, ...] = ()) -> tuple[str, dict]:
    """Mean table and sign tests. Lower is better. Text is the paper source."""
    names = []
    for row in rows:
        if row["setting"] == setting and row["image"] not in names and row["image"] not in skip:
            names.append(row["image"])
    by = {(row["image"], row["setting"], row["method"]): row for row in rows}
    lines = [f"\nlearned cross-eval {setting} n={len(names)} skip={','.join(skip) if skip else 'none'}"]
    lines.append("ours_learned is compared with ours spectral and forward + lines. Not with forward-only, because that confounds w_l.")
    header = f"{'method':<24} {'n':>3}" + "".join(f"{title:>16}" for _key, title, _d in LEARNED_COLUMNS)
    lines.append(header)
    cells = {}
    for method, title in LEARNED_METHODS:
        group = []
        for name in names:
            row = by.get((name, setting, method))
            if row is None:
                raise SystemExit(f"missing {name} {setting} {method}")
            group.append(row)
        rendered = []
        for key, _title, digits in LEARNED_COLUMNS:
            values = [float(row[key]) for row in group]
            text = f"{statistics.fmean(values):.{digits}f} ({statistics.stdev(values):.{digits}f})"
            rendered.append(text)
            cells[(method, key)] = text
        lines.append(f"{title:<24} {len(group):>3}" + "".join(f"{item:>16}" for item in rendered))
    tests = {}
    for left, right, right_name in (
        ("ours_learned", "ours_spectral", "ours spectral"),
        ("ours_learned", "forward_lines", "forward + lines"),
    ):
        for key, map_name, _digits in (("spectral_mass", "spectral", 3), ("learned_mass", "learned", 3)):
            lower = higher = tied = 0
            for name in names:
                a = float(by[(name, setting, left)][key])
                b = float(by[(name, setting, right)][key])
                if a < b:
                    lower += 1
                elif a > b:
                    higher += 1
                else:
                    tied += 1
            trials = lower + higher
            p_text = format_p(binomial_two_sided(lower, trials))
            line = (
                f"sign {left} vs {right} on {map_name}: lower/higher/tied "
                f"{lower}/{higher}/{tied} p={p_text} on {trials} non-ties"
            )
            lines.append(line)
            tests[(right, key)] = {
                "lower": lower,
                "higher": higher,
                "tied": tied,
                "p": p_text,
                "right_name": right_name,
                "map_name": map_name,
            }
    matrix = ["2x2 mean spectral_mass / learned_mass (chosen row, scored column)"]
    for method, title in (("ours_spectral", "chosen spectral"), ("ours_learned", "chosen learned")):
        matrix.append(f"{title}: {cells[(method, 'spectral_mass')]} / {cells[(method, 'learned_mass')]}")
    lines.extend(matrix)
    payload = {"n": len(names), "names": names, "cells": cells, "tests": tests, "text": "\n".join(lines) + "\n"}
    return payload["text"], payload


def format_learned_report() -> str:
    parts = []
    eight = ROOT / "experiments" / "results" / "metrics_learned_8.csv"
    thirty = ROOT / "experiments" / "results" / "metrics_learned_n30.csv"
    if eight.exists():
        rows = _learned_rows(eight)
        for setting in ("width_75", "width_50"):
            text, _payload = analyze_learned(rows, setting)
            parts.append(text)
    else:
        parts.append("\nmetrics_learned_8.csv is missing\n")
    if thirty.exists():
        rows = _learned_rows(thirty)
        present = {row["image"] for row in rows}
        skipped = tuple(name for name in TUNING if name in present)
        text, _payload = analyze_learned(rows, "width_75", skip=skipped)
        parts.append(text)
        parts.append(
            "tuning images in the 30-image file: "
            + (", ".join(skipped) if skipped else "none")
            + "\n"
        )
    else:
        parts.append("\nmetrics_learned_n30.csv is missing\n")
    return "".join(parts)


def main() -> None:
    rows = load_rows()
    print(f"source {CSV_PATH}")
    print("sample standard deviation (n - 1)")
    print_summary(rows, "width_75", WIDTH_75)
    print_summary(rows, "width_50", WIDTH_50)
    print_per_image(rows)
    print_width50_per_image(rows)
    print("\nsix-image check, tuning images dropped")
    print_summary(rows, "width_75", WIDTH_75, skip=TUNING)
    print_lsd(rows)
    print_n30_saliency()


if __name__ == "__main__":
    main()

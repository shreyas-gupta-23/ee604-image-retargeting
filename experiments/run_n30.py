"""Seeded 30-image check at 75% width, with LSD scored in the same run.

Figures are not written. The original eight stay in metrics.csv. This script
writes experiments/results/metrics_n30.csv.

Before any new image is trusted, the original eight are carved again at 75%
width. Their saliency fraction and Hough line score must match metrics.csv.
"""

from __future__ import annotations

import csv
import random
import sys
import urllib.request
import zipfile
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_experiments import RESULT_DIR, load_rgb, run_one  # noqa: E402
from src.baselines import limit_long_side  # noqa: E402

SEED = 604
SAMPLE_N = 30
ZIP_URL = "https://people.csail.mit.edu/mrub/retargetme/download/images-20100824.zip"
ZIP_PATH = ROOT / "experiments" / "retargetme_source.zip"
ORIGINAL_DIR = ROOT / "experiments" / "retargetme"
SAMPLE_DIR = ROOT / "experiments" / "retargetme_n30"
OLD_CSV = RESULT_DIR / "metrics.csv"
OUT_CSV = RESULT_DIR / "metrics_n30.csv"
SUMMARY_PATH = RESULT_DIR / "n30_summary.txt"

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
TUNING = ("Johanneskirche", "obama")
METHODS = ("classic", "forward", "forward_saliency", "forward_lines", "ours")
SETTING = "width_75"
ANCHOR = ("brick_house", "classic", "0.153154", "3.2319")


def download_zip() -> None:
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        return
    print(f"downloading {ZIP_URL}", flush=True)
    request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "ee604-retarget/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        ZIP_PATH.write_bytes(response.read())
    print(f"zip bytes {ZIP_PATH.stat().st_size}", flush=True)


def archive_pngs() -> dict[str, str]:
    download_zip()
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = {}
        for member in archive.namelist():
            path = Path(member)
            if path.suffix.lower() == ".png":
                names[path.stem] = member
    if len(names) < SAMPLE_N:
        raise SystemExit(f"archive has {len(names)} pngs, need at least {SAMPLE_N}")
    return names


def choose_sample(stems: list[str]) -> list[str]:
    rng = random.Random(SEED)
    sample = sorted(rng.sample(sorted(stems), SAMPLE_N))
    return sample


def extract_sample(members: dict[str, str], sample: list[str]) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for stem in sample:
            dest = SAMPLE_DIR / f"{stem}.png"
            if dest.exists():
                continue
            dest.write_bytes(archive.read(members[stem]))
            print("extracted", stem, flush=True)


def load_working(stem: str):
    original = ORIGINAL_DIR / f"{stem}.png"
    path = original if original.exists() else SAMPLE_DIR / f"{stem}.png"
    return limit_long_side(load_rgb(path), 640)


def old_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    with OLD_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["image"], row["setting"], row["method"]): row for row in rows}


def check_saved(saved: dict, name: str, method: str, info) -> None:
    from src.metrics import line_crossings_per_100, saliency_removed_fraction

    row = saved[(name, SETTING, method)]
    sal = f"{saliency_removed_fraction(info):.6f}"
    lines = f"{line_crossings_per_100(info):.4f}"
    if row["saliency_removed_fraction"] != sal or row["line_crossings_per_100"] != lines:
        raise SystemExit(
            f"regression failed {name} {method}: "
            f"csv {row['saliency_removed_fraction']}/{row['line_crossings_per_100']} "
            f"rerun {sal}/{lines}"
        )


def carve(name: str, saved_info: dict | None = None):
    image = load_working(name)
    height, width = image.shape[:2]
    target = (height, max(2, int(round(width * 0.75))))
    rows = []
    for method in METHODS:
        print(f"{name} {SETTING} {method}", flush=True)
        output, info, elapsed = run_one(image, target, method)
        if saved_info is not None:
            check_saved(saved_info, name, method, info)
        from run_experiments import row_for

        row = row_for(name, SETTING, method, image, output, info, elapsed, target)
        print(
            f"  sal {row['saliency_removed_fraction']} hough {row['line_crossings_per_100']} "
            f"lsd {row['lsd_crossings_per_100']} {elapsed:.1f}s",
            flush=True,
        )
        rows.append(row)
    return rows


def binomial_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial test at p = 0.5."""
    if trials <= 0:
        return float("nan")
    probs = [comb(trials, k) / 2**trials for k in range(trials + 1)]
    observed = probs[successes]
    return sum(prob for prob in probs if prob <= observed + 1e-15)


def summarize(rows: list[dict], sample: list[str]) -> str:
    import statistics

    headline = [name for name in sample if name not in TUNING]
    lines = [
        f"seed {SEED}",
        f"sample {len(sample)}: {', '.join(sample)}",
        f"headline drops {', '.join(TUNING)}; n={len(headline)}",
        "setting width_75, sample standard deviation (n - 1)",
        "LSD is not inside the cost",
        "",
    ]
    by = {(row["image"], row["method"]): row for row in rows}

    def values(method: str, names: list[str], column: str) -> list[float]:
        return [float(by[(name, method)][column]) for name in names]

    lines.append(f"{'method':<22} {'n':>3} {'saliency':>16} {'hough':>16} {'lsd':>16}")
    for method in METHODS:
        sal = values(method, headline, "saliency_removed_fraction")
        hough = values(method, headline, "line_crossings_per_100")
        lsd = values(method, headline, "lsd_crossings_per_100")
        lines.append(
            f"{method:<22} {len(headline):>3} "
            f"{statistics.fmean(sal):7.3f} ({statistics.stdev(sal):5.3f}) "
            f"{statistics.fmean(hough):7.2f} ({statistics.stdev(hough):5.2f}) "
            f"{statistics.fmean(lsd):7.2f} ({statistics.stdev(lsd):5.2f})"
        )

    def signs(left: str, right: str, column: str) -> tuple[int, int, int]:
        lower = higher = tied = 0
        for name in headline:
            a = float(by[(name, left)][column])
            b = float(by[(name, right)][column])
            if b < a:
                lower += 1
            elif b > a:
                higher += 1
            else:
                tied += 1
        return lower, higher, tied

    full_lower, full_higher, full_tied = signs("classic", "ours", "lsd_crossings_per_100")
    trials = full_lower + full_higher
    lines.append("")
    lines.append(
        f"LSD sign test, full versus classic, lower/higher/tied "
        f"{full_lower}/{full_higher}/{full_tied}, "
        f"two-sided binomial p={binomial_two_sided(full_lower, trials):.4f} "
        f"on {trials} non-ties"
    )
    line_lower, line_higher, line_tied = signs("forward", "forward_lines", "lsd_crossings_per_100")
    line_trials = line_lower + line_higher
    lines.append(
        f"LSD sign test, +lines versus forward-only, lower/higher/tied "
        f"{line_lower}/{line_higher}/{line_tied}, "
        f"two-sided binomial p={binomial_two_sided(line_lower, line_trials):.4f} "
        f"on {line_trials} non-ties"
    )
    sal_lower, sal_higher, sal_tied = signs("classic", "ours", "saliency_removed_fraction")
    sal_trials = sal_lower + sal_higher
    lines.append(
        f"saliency sign test, full versus classic, lower/higher/tied "
        f"{sal_lower}/{sal_higher}/{sal_tied}, "
        f"two-sided binomial p={binomial_two_sided(sal_lower, sal_trials):.4e} "
        f"on {sal_trials} non-ties"
    )
    text = "\n".join(lines) + "\n"
    return text


def write_csv(rows: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_learned_reports() -> None:
    """Cross-evaluation of the learned-saliency CSVs. Does not carve."""
    from report_numbers import format_learned_report

    print(format_learned_report(), flush=True)


def summary_from_csv() -> None:
    """Rewrite n30_summary.txt from metrics_n30.csv. Does not carve."""
    with OUT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample = []
    for row in rows:
        if row["image"] not in sample:
            sample.append(row["image"])
    text = summarize(rows, sample)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--summary-only":
        summary_from_csv()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--learned":
        print_learned_reports()
        return
    members = archive_pngs()
    sample = choose_sample(list(members))
    print("sample", sample, flush=True)
    extract_sample(members, sample)
    saved = old_rows()
    anchor = saved[("brick_house", SETTING, "classic")]
    if (anchor["saliency_removed_fraction"], anchor["line_crossings_per_100"]) != ANCHOR[2:]:
        raise SystemExit(f"metrics.csv anchor moved: {anchor['saliency_removed_fraction']} {anchor['line_crossings_per_100']}")

    reused: dict[str, list[dict]] = {}
    print("regression on the original eight", flush=True)
    for name in ORIGINAL:
        reused[name] = carve(name, saved)
    print("regression passed", flush=True)

    rows: list[dict] = []
    for name in sample:
        if name in reused:
            print(f"reuse regression rows for {name}", flush=True)
            rows.extend(reused[name])
        else:
            rows.extend(carve(name))
        write_csv(rows)

    text = summarize(rows, sample)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()

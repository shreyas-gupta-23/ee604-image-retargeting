"""We download the RetargetMe source-image zip and keep eight images. We do not commit the zip itself."""

from __future__ import annotations

import re
import sys
import urllib.request
import zipfile
from pathlib import Path

PAGE = "https://people.csail.mit.edu/mrub/retargetme/download.html"
_OPENER = urllib.request.build_opener()
_OPENER.addheaders = [("User-Agent", "Mozilla/5.0 (compatible; ee604-retarget/1.0)")]
urllib.request.install_opener(_OPENER)
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments" / "retargetme"
ZIP_PATH = ROOT / "experiments" / "retargetme_source.zip"

# We chose these stems from the RetargetMe archive: people, buildings, an animal, an object, a landscape, and a street.
KEEP = (
    "obama.png",
    "child.png",
    "brick_house.png",
    "Johanneskirche.png",
    "eagle.png",
    "butterfly.png",
    "waterfall.png",
    "Sanfrancisco.png",
)


def _source_zip_url() -> str:
    with urllib.request.urlopen(PAGE, timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    anchors = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S)
    candidates = []
    for href, raw_text in anchors:
        text = re.sub(r"<[^>]+>", "", raw_text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        if href.lower().endswith(".zip"):
            candidates.append((text, urllib.request.urljoin(PAGE, href)))
            print(f"zip link: {text} -> {candidates[-1][1]}")
    for text, url in candidates:
        if "source image" in text and "only" in text:
            return url
    for text, url in candidates:
        if "source" in text and "retarget" not in text:
            return url
    raise RuntimeError("Could not find the source-image zip on the RetargetMe download page")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    present = {path.name for path in DEST.iterdir() if path.is_file()}
    if set(KEEP).issubset(present):
        print(f"subset already in {DEST}")
        return
    if not ZIP_PATH.exists():
        url = _source_zip_url()
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        by_name = {Path(name).name: name for name in archive.namelist()}
        missing = [name for name in KEEP if name not in by_name]
        if missing:
            raise RuntimeError(f"missing from archive: {missing}")
        for stale in list(DEST.iterdir()):
            if stale.is_file() and stale.name not in KEEP:
                stale.unlink()
        for name in KEEP:
            (DEST / name).write_bytes(archive.read(by_name[name]))
            print("kept", name)
    print(f"wrote {len(KEEP)} images to {DEST}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)

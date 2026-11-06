"""Gradio demo. We compare scale, crop, classic seams, forward energy, and hybrid. We cap the long side at 640 px so a result can be recomputed on request. ``w_s`` and ``w_l`` are the saliency and line weights."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np

from src.baselines import limit_long_side
from src.display import colorize, draw_vertical_seam
from src.metrics import line_crossings_per_100, saliency_removed_fraction
from src.retarget import next_vertical_seam, protection_maps, retarget

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "experiments" / "retargetme"
ROLES_PATH = ROOT / "experiments" / "results" / "roles.txt"
MAX_SIDE = 640


def _find_named(stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        candidate = IMAGE_DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def example_paths() -> list[str]:
    """We prefer the line, saliency, and failure images we chose for the experiments."""
    ordered: list[Path] = []
    if ROLES_PATH.exists():
        roles = {}
        for line in ROLES_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                roles[key.strip()] = value.strip()
        for key in ("lines", "saliency", "failure"):
            path = _find_named(roles.get(key, ""))
            if path is not None:
                ordered.append(path)
    if not ordered and IMAGE_DIR.exists():
        ordered = [
            path for path in sorted(IMAGE_DIR.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ][:3]
    return [str(path) for path in ordered]


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    import cv2
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _hstack(images: list[np.ndarray]) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    fitted = [_fit_height(image, height) for image in images]
    gap = np.full((height, 8, 3), 255, np.uint8)
    pieces: list[np.ndarray] = []
    for index, image in enumerate(fitted):
        if index:
            pieces.append(gap)
        pieces.append(image)
    return np.hstack(pieces)


def retarget_image(image, width_percent, height_percent, method, w_s, w_l,
                   show_saliency, show_lines, show_seam):
    if image is None:
        return None, None, "Upload an image or pick an example."
    rgb = limit_long_side(np.asarray(image), MAX_SIDE)
    height, width = rgb.shape[:2]
    target_h = max(2, int(round(height * float(height_percent) / 100.0)))
    target_w = max(2, int(round(width * float(width_percent) / 100.0)))
    started = time.perf_counter()
    output, info = retarget(
        rgb, (target_h, target_w), method=method, weights=(float(w_s), float(w_l)), return_info=True,
    )
    elapsed = time.perf_counter() - started
    pair = _hstack([rgb, output])

    overlays = []
    if show_saliency or show_lines or show_seam:
        maps = protection_maps(rgb)
        if show_saliency:
            overlays.append(colorize(maps["saliency"]))
        if show_lines:
            overlays.append(colorize(maps["lines_vertical"]))
        if show_seam and method in ("classic", "ours", "hybrid"):
            seam = next_vertical_seam(rgb, method=method, weights=(float(w_s), float(w_l)))
            overlays.append(draw_vertical_seam(rgb, seam))
    overlay_image = _hstack(overlays) if overlays else None

    message = (
        f"Working size {height} x {width} -> {output.shape[0]} x {output.shape[1]} "
        f"in {elapsed:.1f}s. Method: {method}."
    )
    if method in ("classic", "ours", "hybrid") and info.get("seam_pixels", 0) > 0:
        message += (
            f" Saliency mass removed: {saliency_removed_fraction(info):.3f}."
            f" Line pixels per 100 seam pixels: {line_crossings_per_100(info):.1f}."
        )
    if info.get("scaled_after_seams"):
        message += " Seams stopped at 75% of the original side; scaling finished the cut."
    return pair, overlay_image, message


def build() -> gr.Blocks:
    with gr.Blocks(title="Image retargeting") as demo:
        gr.Markdown(
            """
# Image retargeting

Spatial-domain seam carving. **Ours** adds spectral-residual saliency and an
orientation-aware line map to Rubinstein forward energy. **Hybrid** uses
that same cost, carves only down to 75% of the original side, and scales the rest.

Suggested demo: a facade (turn the line weight up), a person (saliency),
then San Francisco at 50% width, comparing ours with hybrid.
            """
        )
        with gr.Row():
            source = gr.Image(type="numpy", label="Input", height=360)
            with gr.Column():
                width_percent = gr.Slider(50, 150, value=75, step=1, label="Target width (%)")
                height_percent = gr.Slider(50, 150, value=100, step=1, label="Target height (%)")
                method = gr.Radio(
                    ["hybrid", "ours", "classic", "scale", "crop"],
                    value="hybrid",
                    label="Method",
                )
                w_s = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Saliency weight w_s")
                w_l = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Line weight w_l")
                show_saliency = gr.Checkbox(value=True, label="Show saliency")
                show_lines = gr.Checkbox(value=True, label="Show vertical-seam line map")
                show_seam = gr.Checkbox(value=True, label="Show the next vertical seam")
                run = gr.Button("Retarget", variant="primary")
        result = gr.Image(type="numpy", label="Original (left) and retargeted (right)")
        maps = gr.Image(type="numpy", label="Saliency, line map, next seam")
        status = gr.Textbox(label="Notes", lines=2)
        run.click(
            retarget_image,
            inputs=[source, width_percent, height_percent, method, w_s, w_l,
                    show_saliency, show_lines, show_seam],
            outputs=[result, maps, status],
        )
        paths = example_paths()
        if paths:
            gr.Examples(examples=[[path] for path in paths], inputs=[source], label="Examples")
    return demo


if __name__ == "__main__":
    build().launch()

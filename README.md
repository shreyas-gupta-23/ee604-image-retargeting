# Image retargeting with forward-energy seam carving

EE604 course project, IIT Kanpur (Prof. Koteswar Rao Jerripothula). The code is spatial-domain only, in OpenCV and NumPy. You give it an image and a target height and width, and it returns an image of exactly that size.

**Authors:** Arsh Jaswal (230205), Harshvardhan Gour (230467), Nikhil Verma (230699), Shreyas Gupta (230991).

## What it does

The core is Rubinstein forward-energy seam carving. We add two terms to the seam cost:

- **Spectral-residual saliency** (Hou and Zhang, 2007), so seams avoid salient regions.
- **An orientation-weighted Hough line map.** The seam direction decides which lines a seam is penalized for crossing, so a vertical seam pays for crossing vertical lines. This sits inside the one forward-energy recurrence.

There are two variants of the same carver, and they differ only in the saliency map:

| Variant | Saliency source | Weights |
|---|---|---|
| Ours (spectral), the default | Spectral residual | `w_s = 0.5`, `w_l = 0.8` |
| Ours (learned) | Pretrained U²-Net-small (`u2netp`, loaded through `rembg`) | `w_s = 0.25`, `w_l = 0.8` |

No retargeting network is trained. The learned variant only swaps in a pretrained salient-object map. The spectral weights were chosen by eye on Johanneskirche and obama, and they were not swept. The learned `w_s` came from a small sweep on the same two images (`experiments/sweep_learned_ws.py`).

Baselines are uniform scaling, center-crop, classic backward-energy seam carving (Avidan and Shamir, 2007), and forward energy without the extra terms. The hybrid schedule stops seams at 75% of a side and scales the rest. Images are limited to a long side of 640 px before carving.

An independent LSD line check (`experiments/score_lsd.py`) is used only as a score. It never enters the cost.

## What we found

- On the original eight RetargetMe images at 75% width, the saliency term lowers deleted spectral mass from 0.089 to 0.068. The line term lowers Hough-map crossings from 1.14 to 0.31 (Ours spectral: 0.37). Both scores reuse maps from the cost, so they show the weights move the seams, not that the pictures look better.
- On a seeded sample of 30 images, the saliency term lowers deleted spectral mass from 0.118 to 0.093. On the independent LSD map the line term shows no detectable benefit (lower on 16 of 29 images, p = 0.71), so the in-cost line score overstates it.
- Cross-evaluation: each saliency source wins under its own map. Ours (learned) against Ours (spectral) is 0/30/0 under the spectral map and 30/0/0 under the learned map. Against forward energy with lines, under the spectral map, it is 11/19/0 (p = 0.20, not significant).
- We did not compare with published RetargetMe operators and we did not run a user study.

The full numbers, tables, and figures are in the paper.

## Run

```bash
pip install -r requirements.txt
python experiments/download_retargetme.py
python experiments/run_experiments.py
python experiments/run_hybrid.py
python experiments/run_n30.py --summary-only
python experiments/report_numbers.py
python app.py
```

`download_retargetme.py` fetches the public RetargetMe source-image archive and keeps eight images in `experiments/retargetme/`. `run_n30.py --summary-only` reprints the saved CSV. Running it without that flag re-carves the 30-image sample and takes a long time.

The large result folders (`experiments/results/outputs/`, `sheets_n30/`, `maps/`) are not in this repository. The CSVs and summary files that the paper's numbers come from are.

## Layout

- `src/energy.py`: luminance, spectral-residual saliency, learned saliency, and the line maps
- `src/seam.py`: forward and backward dynamic programs, seam removal and insertion
- `src/baselines.py`: scale, center-crop, and the long-side limit
- `src/retarget.py`: `retarget(image, target_hw, method, weights)`, with methods `scale`, `crop`, `classic`, `ours`, and `hybrid`. The saliency source is `spectral` by default, or `learned`
- `src/metrics.py`: saliency mass removed and line crossings
- `app.py`: the Gradio demo
- `experiments/run_experiments.py`: comparison figures and `metrics.csv`
- `experiments/run_hybrid.py`: 50% width sheet of scale, full seams, and the hybrid schedule
- `experiments/run_n30.py`: the seeded sample of 30 images (seed 604)
- `experiments/score_lsd.py`: the independent LSD score
- `experiments/sweep_learned_ws.py`, `run_overnight.py`, `show_learned_maps.py`, `build_saved_sheets.py`: the learned-saliency runs and sheets
- `experiments/report_numbers.py`: means and sign tests, read from the saved CSVs
- `experiments/results/*.csv`: the saved metrics behind every number in the paper
- `paper/main.tex`, `paper/main.pdf`: the CVPR-format write-up (the PDF is the built paper)
- `slides/outline.md`: the presentation outline

## Write-up

`paper/main.tex` is the CVPR template with page numbers. Figures are read from `experiments/results/`, and the source-code appendix is read from `src/`, `app.py`, and `experiments/`. Appendix A lists each member's contribution.

## Links

- RetargetMe dataset: <http://people.csail.mit.edu/mrub/retargetme/>
- RetargetMe source images: <http://people.csail.mit.edu/mrub/retargetme/download.html>
- OpenCV: <https://github.com/opencv/opencv>
- NumPy: <https://github.com/numpy/numpy>
- Gradio: <https://github.com/gradio-app/gradio>
- U²-Net: <https://github.com/xuebinqin/U-2-Net>
- rembg: <https://github.com/danielgatis/rembg>

# Image retargeting

Spatial-domain image retargeting for EE604. An input image and a target height and width produce an image of that exact size. The default method is Rubinstein forward-energy seam carving, plus a spectral-residual saliency map and an orientation-aware line map. There is no trained retargeting network. A pretrained U²-Net-small salient-object map, loaded through rembg as `u2netp`, is an optional comparison and is not the default cost. The hybrid schedule uses the spectral cost, stops seams at 75% of the original side, and scales anything beyond that.

Uniform scaling, center-crop, and classic backward-energy seam carving (Avidan and Shamir, 2007) are the baselines.

Frozen spectral weights, after a visual check: saliency `w_s = 0.5`, lines `w_l = 0.8`. The optional learned comparison uses `w_s = 0.25` and `w_l = 0.8`. The demo sliders override the spectral weights. Images are limited to a long side of 640 px before carving.

## Run

```bash
pip install -r requirements.txt
python experiments/download_retargetme.py
python experiments/run_experiments.py
python experiments/run_hybrid.py
python experiments/run_n30.py --summary-only
python app.py
```

The download step fetches the public RetargetMe source-image archive and keeps eight images in `experiments/retargetme/`.

- Dataset: <http://people.csail.mit.edu/mrub/retargetme/>
- Source images: <http://people.csail.mit.edu/mrub/retargetme/download.html>

## Layout

- `src/energy.py` — luminance, spectral residual saliency, optional learned saliency, line protection
- `src/seam.py` — forward and backward dynamic programs, removal, insertion
- `src/baselines.py` — scale, center-crop, long-side limit
- `src/retarget.py` — `retarget(image, target_hw, method, weights)`; methods include `hybrid`. The default saliency source is `spectral`
- `src/metrics.py` — saliency mass removed and line crossings
- `app.py` — Gradio demo
- `experiments/run_experiments.py` — comparison figures and `experiments/results/metrics.csv`
- `experiments/run_hybrid.py` — 50% width sheet of scale, full seams, and the hybrid schedule
- `experiments/run_n30.py` — seeded sample of 30 images. `--summary-only` reprints the saved CSV
- `experiments/report_numbers.py` — means and sign tests from the saved CSVs
- `paper/main.tex` — CVPR-format write-up
- `slides/outline.md` — presentation outline

## Links

- OpenCV: <https://github.com/opencv/opencv>
- NumPy: <https://github.com/numpy/numpy>
- Gradio: <https://github.com/gradio-app/gradio>
- RetargetMe: <http://people.csail.mit.edu/mrub/retargetme/>

## Write-up

`paper/main.tex` is the CVPR camera-ready template (the roll number is on the author line, the review ruler is off) with page numbers. The figures are read from `experiments/results/`, and the appendix listings are read from `src/`, `app.py`, and `experiments/`.

The reported numbers come from `metrics.csv`, `metrics_n30.csv`, `metrics_learned_8.csv`, and `metrics_learned_n30.csv`.

## Author line

230991. The write-up does not print a name.

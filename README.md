# Image retargeting

Spatial-domain image retargeting for EE604. An input image and a target height and width produce an image of that exact size. The method is Rubinstein forward-energy seam carving, plus a spectral-residual saliency map and an orientation-aware line map. There is no learned model. The hybrid schedule uses that same cost, stops seams at 75% of the original side, and scales anything beyond that.

Uniform scaling, center-crop, and classic backward-energy seam carving (Avidan and Shamir, 2007) are the baselines.

Frozen weights, after a visual check: saliency `w_s = 0.5`, lines `w_l = 0.8`. The demo sliders override them. Images are limited to a long side of 640 px before carving.

## Run

```bash
pip install -r requirements.txt
python experiments/download_retargetme.py
python experiments/run_experiments.py
python experiments/run_hybrid.py
python app.py
```

The download step fetches the public RetargetMe source-image archive and keeps eight images in `experiments/retargetme/`.

- Dataset: <http://people.csail.mit.edu/mrub/retargetme/>
- Source images: <http://people.csail.mit.edu/mrub/retargetme/download.html>

## Layout

- `src/energy.py` — luminance, spectral residual saliency, line protection
- `src/seam.py` — forward and backward dynamic programs, removal, insertion
- `src/baselines.py` — scale, center-crop, long-side limit
- `src/retarget.py` — `retarget(image, target_hw, method, weights)`; methods include `hybrid`
- `src/metrics.py` — saliency mass removed and line crossings
- `app.py` — Gradio demo
- `experiments/run_experiments.py` — comparison figures and `experiments/results/metrics.csv`
- `experiments/run_hybrid.py` — 50% width sheet of scale, full seams, and the hybrid schedule
- `paper/main.tex` — CVPR-format write-up
- `slides/outline.md` — presentation outline

## Links

- OpenCV: <https://github.com/opencv/opencv>
- NumPy: <https://github.com/numpy/numpy>
- Gradio: <https://github.com/gradio-app/gradio>
- RetargetMe: <http://people.csail.mit.edu/mrub/retargetme/>

## Write-up

`paper/main.tex` is the CVPR camera-ready template (authors visible, review ruler off) with page numbers. This machine does not have `pdflatex`. Upload the whole `project` directory to the [CVPR Overleaf template](https://www.overleaf.com/latex/templates/cvpr-2022-author-kit/qbmjsdxryffn) or to a new Overleaf project, and set the main file to `paper/main.tex`. The figures are read from `experiments/results/`, and the appendix listings are read from `src/`, `app.py`, and `experiments/`.

The write-up lists Shreyas Gupta, 230991. There is no public repository URL. If this is a group submission, add the other members before each person uploads.

## Authors

Shreyas Gupta, 230991. If this is a group submission, add the other members and a contribution paragraph for each before uploading.

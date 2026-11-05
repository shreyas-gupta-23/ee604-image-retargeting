# Image retargeting — slide outline

About ten slides for the 7–8 Nov demo. Put the named result image on the slide. Speaker notes are the sentences underneath each slide.

Replace the name line before presenting. The GitHub URL in the last slide is still a placeholder.

## 1. Title

**Orientation-aware seam carving for image retargeting**

EE604 course project. Names and roll numbers.

Notes: One sentence. We change an image's height and width. The default cost has no learned weights. A pretrained saliency model is only an optional comparison. The rest of the talk is the cost we put on a seam, then the pictures.

## 2. Problem statement

**Input.** One color image. A target height and width. Two weights, used only by our seam cost.

**Output.** An image of that exact size.

**Constraints.** Spatial domain only: NumPy and OpenCV. No trained retargeting network. The demo caps the long side at 640 px so the dynamic program can be recomputed live. Enlargement duplicates pixels; it does not invent texture. Past about 1.5× we say so.

Notes: Say the input and output before any algorithm name. Scaling and cropping are the constraints we are refusing: scaling squashes everything, cropping deletes the border.

## 3. Literature — what was done before

- Avidan and Shamir, SIGGRAPH 2007. Delete the cheapest 8-connected seam. Cost = gradient magnitude (backward energy).
- Rubinstein, Shamir, and Avidan, SIGGRAPH 2008. Forward energy: charge the new edges the deletion creates.
- Hou and Zhang, CVPR 2007. Spectral-residual saliency. Fourier residual, no training.
- Rubinstein et al., SIGGRAPH Asia 2010. RetargetMe, the comparison set we sample.
- Rubinstein, Shamir, and Avidan, SIGGRAPH 2009. Multi-operator search over scale, crop, and seams. We do not do this search. Scale and crop are baselines only.

Notes: Be able to write the backward recurrence and the three forward costs if someone asks. Do not claim any of these papers as ours.

## 4. Literature — how this is different

Each ingredient is published. The piece to defend is the **combination inside one recurrence**:

1. Rubinstein forward energy on the current luminance.
2. Spectral-residual saliency, computed once and carried with the pixels.
3. A Hough line map whose weight depends on the seam direction: `|sin θ|` when the seam is vertical, `|cos θ|` when it is horizontal.

A vertical seam must cross a horizon. That weight does not also punish columns and mullions for the crossing.

Notes: If asked "what is novel?", say that sentence and point at the ablation, not at a claim of being first. If asked why not a deep network, the course constraint is spatial domain, and we can explain every step.

## 5. Method — steps and why

1. **Luminance** in `[0, 1]`. The forward cost is defined on intensity.
2. **Spectral residual.** Marks high-contrast structure without a dataset. Why: a smooth subject can be cheaper than textured background under a pure gradient.
3. **Long line segments only.** Blur, Canny (80/160), Hough, keep segments of at least 30 px and 12% of the short side, at most 40. Why: brick texture would otherwise fill the mask and the seam would have nowhere to go.
4. **Orientation weight.** Why: the direction of the seam decides which lines are actually at risk.
5. **Dynamic program** (next slide). Why forward energy: backward energy ignores the tear.
6. **Repeat.** If both sides still need to shrink, delete the cheaper of the two directions.
7. **Enlarge** by recording seams on a copy and inserting averaged duplicates. Why average: a raw copy leaves a hard doubled edge.

Frozen weights after a visual check on the church and the portrait: `w_s = 0.5`, `w_l = 0.8`. The demo sliders move them.

Image: `experiments/results/fig_maps.png` (Johanneskirche: input, saliency, line map, next seam).

## 6. The recurrence

Vertical seam, luminance `I`, extra cost `E = w_s S + w_l L`:

- `C_U = |I(i, j+1) - I(i, j-1)|`  (the new horizontal edge)
- `C_L = C_U + |I(i-1, j) - I(i, j-1)|`  (arrived from the upper left)
- `C_R = C_U + |I(i-1, j) - I(i, j+1)|`  (arrived from the upper right)
- `M(i,j) = E(i,j) + min` of the three arrivals

Classic baseline: `M(i,j) = |Sobel| + min` of the three neighbors. No forward term, no `E`.

Notes: Walk a 3-column toy if asked. Border neighbors are clamped. A step off the image is infinite. Horizontal seams are this recurrence on the transpose.

## 7. Experiments — data and implementation

**Dataset.** Eight RetargetMe source images: obama, child, brick_house, Johanneskirche, eagle, butterfly, waterfall, Sanfrancisco. Link: http://people.csail.mit.edu/mrub/retargetme/

**Implementation.** Python 3, NumPy, OpenCV, Gradio. Long side 640. No GPU. Vectorized row update.

**Protocol.** Width to 75% and to 50%. One height cut to 75% (waterfall). One enlargement to 150% width (butterfly). Ablation at 75% width: forward only, forward+saliency, forward+lines, full model.

**Scores.** Fraction of saliency mass actually deleted. Protected-line pixels per 100 seam pixels. Wall time. Every output size was checked.

Images: the four-row sheets `fig_compare_75_paper.png` and `fig_compare_50_paper.png`. The eight-image sheets are `fig_compare_75.png` and `fig_compare_50.png` if you need a backup.

## 8. Results

Means over all eight images. Saliency is the fraction of spectral-residual mass deleted. Lines are protected-line pixels per 100 seam pixels.

**75% width (25% reduction)**

| Method | Saliency | Lines | Seconds |
| --- | --- | --- | --- |
| Classic | 0.089 | 1.30 | 11.0 |
| Forward only | 0.089 | 1.14 | 13.7 |
| Forward + saliency | 0.068 | 0.99 | 12.9 |
| Forward + lines | 0.089 | 0.31 | 13.4 |
| Full model | 0.068 | 0.37 | 14.4 |

**50% width:** classic 0.222 saliency and 1.61 lines, in 21.5 s. Full model 0.186 and 0.69, in 26.0 s. Lower on both scores for every image.

What to say:

- The saliency weight is what moves the saliency column (0.089 to 0.068). The line weight does not undo it.
- The line weight is what moves the line column (1.14 to 0.31). The full model keeps that drop (0.37).
- At 75% width, classic and ours look similar, and both beat scaling. The crop already slices the heart and clips the child.
- At 50% width, point at the church window and the child's outline. Then show Sanfrancisco: the heart is curved, the Hough map does not guard it, and both seam methods pinch it.

Images:

- `fig_ablation.png` — church, five costs
- `fig_compare_50_paper.png` — where the methods separate
- `fig_failure.png` — Sanfrancisco at 50%
- `fig_height.png` and `fig_enlarge.png` if there is time. Height cut: saliency 0.016 classic vs 0.009 ours, line score 0 for both. Enlarge meets 150% width in about 28 s by duplicating seams.

Notes: The scores measure our own maps. Say that. They check that the maps affect the seams. They are not a user study. On `child` at 75% width both line scores are under 0.01.

## 9. Demo script

Open `python app.py`.

1. **Johanneskirche, width 75%, method classic, then ours.** Show the line map. Raise `w_l`. The next seam should avoid the bright vertical mullions.
2. **obama, width 75%.** Show saliency. Compare classic and ours. A center crop is the third picture: it keeps the middle and can cut whoever stands to the side.
3. **Sanfrancisco, width 50%.** Leave the weights at 0.5 and 0.8. This is the honest slide: every operator gives something up.
4. If asked about enlargement: butterfly, width 150%, ours versus scale.

Be ready to open `src/seam.py` at `forward_seam` and `src/energy.py` at `line_protection`.

## 10. Limitations, contributions, links

**Limitations.** Spectral residual follows edges, not faces. Hough misses short and curved structures. A 50% cut of a busy image must delete something protected. Enlargement only duplicates pixels. No user study.

**Contributions.** Each person states their own block from the appendix of the report: maps, dynamic program, experiments, demo, or writing.

**Links.** RetargetMe. OpenCV, NumPy, Gradio. Repository: `https://github.com/shreyas-gupta-23/ee604-image-retargeting`

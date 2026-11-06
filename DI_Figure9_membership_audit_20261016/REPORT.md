# Figure 9: finite-population membership audit

16 October 2026. Read-only investigation of the existing manuscript figure; no DI calculation, clustering, OT or TSP optimization was run. Only this new audit directory was created. Proposed wording follows the supplied manuscript's mathematical notation and style.

**Finding: all 20 displayed sample points in every panel are genuine population units. However, the generation code projects rotated targets onto distinct grid units. Panel (d) is therefore a grid approximation to a 45-degree rotation, not an exact rigid rotation. Panel (c) is valid and happens also to be an exact 90-degree rotation about a different center from the one stated in its panel title.**

## Source and figure identity

The exact figure is `simulations/results/temporary_2D_relative_rotation/grid_20x20_N400_DI_original_shift_rotation90_rotation45_large_2x2_n20.pdf`. It is byte-identical to `figures/DI_rotation_4x5.pdf` in six manuscript archives, including `Spread_Measure_Submission_FINAL.zip` and `JSSAM_Measure_Final_Ready.zip`. The October 12 manuscript snapshot references this filename at line 1174. This report concerns that version; hashes and archive names are in `verification.json`.

The source is [temporary_random2D_DI_two_six_panel_figures.py](../simulations/temporary_random2D_DI_two_six_panel_figures.py); despite its old filename/docstring it constructs a square grid. It is identical to code cell 1 in [13_2D_grid_DI_translation_rotation.ipynb](../simulations/jupyters/13_2D_grid_DI_translation_rotation.ipynb). Both source files match Git commit `046dbca43` (4 September 2026) byte-for-byte.

Exact source locations:

- Lines 19–20: population coordinates; lines 36–40: equal inclusion probabilities.
- Lines 48–72: reference fitting and selection of population medoids as the starting sample.
- Lines 75–81: `nearest_distinct_population_units`, minimizing total **Euclidean** target-to-population distance with `linear_sum_assignment` and returning distinct population rows.
- Lines 84–89: `transformed_sample`, rotating about `refs.mean(axis=0)` and then calling that assignment routine.
- Lines 106–107 and 121–127: exact shift and rotation specifications.
- Lines 131–134 and 92–103: independent evaluation of each assigned sample, using sample-initialized clustering, medoids, **Euclidean** medoid/sample matching and package KDE. The original scoring path must not be described as verified squared-distance matching.
- Lines 171–175: DI is displayed rounded to two decimals, with magnitudes below 0.0005 displayed as zero. Displayed zero is not evidence of exact numerical zero.

Actual units were recovered from the original vector PDF's red-marker transforms, calibrated against its axes. Their ordered coordinates reproduce the source's minimum-cost projection, including the 90-degree tie outcome. No reference optimization was rerun.

## Population, sample and exact transformations

Use integer grid coordinates `(i,j)` for clarity; divide both coordinates by 19 to obtain the source coordinates. The finite population is

\[
 U=\{(i/19,j/19):i,j=0,\ldots,19\},\quad N=400,\quad n=20,\quad \pi_\ell=1/20.
\]

The original sample is exactly the Cartesian product

\[
 s=\{1,5,9,13,17\}\times\{2,7,12,17\}/19.
\]

The source's rotation center is the sample mean `c_s=(9,9.5)/19=(9/19,1/2)`, which differs from the population center `c_U=(9.5,9.5)/19=(1/2,1/2)`. All angles below are counterclockwise. The target transformation is `t=c_s+R_theta(x-c_s)+delta`, followed by distinct-unit assignment for every panel.

| Original panel | Target angle | Explicit translation delta | Target units / 20 | Inside but off grid | Outside square | Displayed units / 20 | Displayed DI |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| (a) Original | 0° | (0,0) | 20 | 0 | 0 | 20 | 0.00 |
| (b) Shifted | 0° | (1,-1)/19 | 20 | 0 | 0 | 20 | 0.00 |
| (c) Rotated targets | 90° about c_s | (0,0) | 0 | 20 | 0 | 20 | 0.00 |
| (d) Rotated targets | 45° about c_s | (0,0) | 0 | 16 | 4 | 20 | -0.14 |

**Targets and displayed units differ.** Panel (d) is a valid finite-population sample approximating a geometric rotation; its DI does not establish what happens under an exact 45-degree rotation.

Panel (b) is an exact one-step-right, one-step-down translation. Panel (c)'s ideal targets, in grid coordinates, are `(18.5-j,i+0.5)`, all half-grid locations. Its actual plotted points are `(19-j,i+1)`: snapping adds `(0.5,0.5)/19` to every target. Their maximum target displacement is `sqrt(0.5)/19 = 0.0372161464`, far above roundoff. This particular tie resolution retains a rigid 90-degree pattern, but its effective rotation center is `(9,10)/19`, not the original sample mean. The alternative description is a population-centered 90-degree rotation followed by translation `(0,1)/19`.

Panel (d) has nonuniform projection displacements, up to `0.1034535861` (about 1.97 grid spacings). Four targets lie outside the population's convex hull `[0,1]^2`; assignment returns boundary grid units. Pairwise distances change, confirming that this is not a rigid rotation.

## Additional exact rotations about the population center

These checks use the same original sample and population and **do not project or snap the rotated points**.

| Angle about (1/2,1/2) | Population units / 20 | Inside but not units | Outside boundary | All inside? | Valid sample subset? |
| --- | ---: | ---: | ---: | --- | --- |
| 90° | 20 | 0 | 0 | Yes | Yes |
| 45° | 0 | 16 | 4 | No | No |

The square-grid symmetry gives the exact 90-degree mapping `(i,j) -> (19-j,i)`. Hence the valid rotated sample is

\[
 s_{90}=\{2,7,12,17\}\times\{1,5,9,13,17\}/19.
\]

Coordinates and population indices are in `valid_centered_90_sample.csv`. **Its DI has not been computed or assumed to equal zero.** Current panel (c) need not be replaced for feasibility; its actual construction can be described accurately instead.

The central 45-degree rotation maps `(i,j)` to `(9.5+(i-j)/sqrt(2),9.5+(i+j-19)/sqrt(2))` in grid coordinates. Here `i-j` is an integer. If nonzero, its irrational contribution prevents the first coordinate being an integer; if zero, that coordinate is 9.5, also not a grid coordinate. Thus none of these 20 points is a population unit. Boundary membership is a separate test and fails for four points.

## Interpretation and minimum text correction

**Every displayed sample is feasible**, but the wording omits projection, overstates panel (d)'s meaning and names the wrong center for panel (c)'s effective exact rotation. Reported DI values belong to the projected samples. Figure 9 alone proves neither a general translation-invariance theorem nor failure of DI invariance under an exact 45-degree rotation of a feasible sample.

Proposed caption and Appendix replacements are in `proposed_corrections.tex`; nothing was inserted into the manuscript or responses. They retain reported, rounded DI values and disclose projection. The original matching-cost discrepancy remains a separate issue.

## Verification and reproducibility

`verification.json` records hashes, six archive matches, projection optimality and preservation checks. `point_checks.csv` contains 120 target checks; `panel_and_rotation_checks.csv` contains six summaries. Scientific membership tolerance is `1e-12`. A separate `2e-6` bound only decodes PDF marker coordinates, never ideal rotated-point membership. Exact rational arithmetic checks both 90-degree results. `independent_verification.json` records agreement on all 80 plotted points between Cairo SVG and MuPDF trace extraction.

Reproduce without any DI or optimization simulation:

```bash
cd /home/bardia/projects/graphical-sampling
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python DI_Figure9_membership_audit_20261016/audit.py
```

This command writes only audit outputs in this new directory. It imports neither the GFS package nor the experimental runner and uses one numerical thread. All original manuscripts, reviewer responses, package files, figures, datasets and saved experimental results remain unchanged.

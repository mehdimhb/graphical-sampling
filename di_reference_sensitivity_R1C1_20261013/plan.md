# Reviewer 1 Comment 1 reference sensitivity implementation plan

> Execute this authorized research task inline, with numerical verification before reporting. The user's new-directory safeguard takes precedence over skill defaults about worktrees, commits, or plan approval. No package, manuscript, dataset, or previous result is modified.

**Goal:** Determine whether alternative references with equal/near manuscript clustering loss can materially change original DI for a fixed sample, restricted to Reviewer 1 Comment 1.

**Architecture:** An isolated adapter calls the actual package OT updates, ordered refinement, hardening, and original DI score routine. Separate analysis and verification read the saved state. The public DI pipeline discrepancies are documented, not silently used as defaults.

**Tech stack:** Existing Python virtualenv, graphical_sampling, NumPy/SciPy/pandas, joblib, Matplotlib. No external data or simulation study.

**Authority:** /home/bardia/Downloads/Measuring_Spatial_Spread_current_2026-10-12.tex (latest available local Measure manuscript, confirmed by modification dates); the user request in this conversation.

## Fixed design, recorded before any new sensitivity score

- Population: existing simulations/populations/simulated/clust_N144_perm.csv, N=144.
- Sample size n=8. UP: Population normalizes stored prob weights to sum 8. No reweighting across references.
- Observed samples: first two sorted size-8 WOR subsets from default_rng(20261013). Chosen without inspecting DI. This deterministic selection fixes examples, not a sampling-design estimate.
- For each sample, 10 official runs with sample coordinates as OT initialization and NumPy/Numba TSP seeds 0,...,9.
- For each sample, 10 stress runs with the first ten sorted size-8 WOR population-center subsets from default_rng(918273), fixed TSP seed0. These initializations are deliberately outside the manuscript official procedure.
- Full OT/reference pipeline recomputed every run. Same stress center list for both observed samples. Label permutations deduplicated.
- OT eps=.01; squared costs divided by current median+1e-12. Explicit row/column marginal tolerance1e-10; outer centroid tolerance1e-7, max200 iterations; numerical caps recorded. Final coupling and label stability checked.
- Existing package TSP15 restarts, local order, border splitting, argmax hardening. Record soft and hard IP totals; do not assume exact hard balance.
- Medoids minimize IP-weighted squared distance; ties largest IP then smallest actual ID. Squared Hungarian matching. Original angular calibration pi/8 and sign formula unchanged.
- Uniform box KDE, h=N^(-1/(d+4)), H=h²cov(original normalized coords,ddof=1), symmetric SPD inverse root; fixed for all original/translated clouds and references. Unweighted KDE as specified by manuscript. Scott matrix ambiguity explicitly acknowledged, no kernel/bandwidth variation within this task.
- Primary suitability diagnostic: stated normalized entropic OT loss J at the converged preliminary centers and exact transport constraints. Median normalization is reference-dependent in the manuscript; also record raw transport cost and a common-scale version using the official seed0 median, without changing solver updates.
- The manuscript specifies no minimization objective after border splitting/hardening. Record final fractional compactness Qsoft (IP-weighted squared distances to final soft weighted centroids), hard compactness Qhard, medoid objective, and hard IP mass deviations. Do not describe them as a missing manuscript global optimizer.
- Compare pairs using relative gap |J_a-J_b|/max(|J_a|,|J_b|,1e-15), and separately |Qsoft_a-Qsoft_b|/min(Qsoft_a,Qsoft_b). Show tolerances numerical equality (1e-10 relative),0.1%,1%,5%. Report OT-only and conservative joint OT+Qsoft comparisons, and both individual/combined eligibility. Lower loss is preferable, but no run is claimed globally optimal.
- Label reference differences by optimal RMS medoid-set matching, maximum matched distance, and IP-weighted partition disagreement after optimal label alignment. Same canonical hard partition+medoids is not a distinct reference.
- Predeclared descriptive material DI difference:0.05; also report exact ranges and 0.1 threshold so conclusions do not depend on that label. Flag opposite-sign pairs and decompose their DI differences, avoiding confusion with aggregation discontinuity.

## Tasks and completion checks

1. [x] Freeze settings, observed samples, stress center IDs, and source hashes in preselection.json; save raw and normalized population tables and authority snapshot. Audit manuscript/package differences in discrepancies.md.
2. [x] Implement adapter.py/run_experiment.py; execute40 runs; save every result/failure, full reference state, matching, densities/contributions. Validate against actual package original scoring algebra and constraints.
3. [x] Analyze distinct references, objective tolerance comparisons, pair geometry and score differences. Produce one compact comparison table and at most one useful multi-panel figure, PDF+PNG.
4. [x] Write short report, minimal proposed manuscript clarification and separate reviewer response in LaTeX; new manuscript text wrapped in textcolor{green}{...}. Never edit originals or Sampling paper.
5. [x] Independently verify from serialized values: weighted squared medoid minima, optimal assignment, box KDE counts, DI identity, objectives, distinctness, fixed settings/source hashes. Record limitations, failed convergence, and self-review.
6. [x] Package new results and deliver a new Desktop copy in line with the user's standing location preference, preserving previous items.

## Review focus

- Equal preliminary OT losses do not imply equally compact final hard partitions: show both stages explicitly.
- Median-normalized losses at different references are not a single fixed-scale global k-means objective: report common-scale/raw diagnostics.
- UP hardening can produce unequal hard IP totals: retain all totals and verify the border-rounding bound, rather than enforcing equality.
- Tied assignment or relabeling can mimic reference differences: canonicalize partitions and audit assignment uniqueness/relabel invariance.
- Box kernels, matching, hardening, and DI sign can introduce discrete changes: no unproved continuity/robustness theorem.

## Execution ledger

- Audit completed before scoring: latest authority identified; package public DI differs in OT initialization, distance powers, and KDE. Package OT branch itself ignores supplied sample centers; separate override required.
- No new sensitivity calculation has been run before this plan and preselection.

- Tasks1–5 complete: all40 runs converged; six distinct reference configurations per fixed sample,30 within-sample reference pairs; source hashes unchanged. Independent numerical audit passed, actual TSP/refinement replayed, all40320 squared assignments checked per run, and no assignment/hardening ties occurred.
- Exact source definitions were copied into local helper modules after the run to remove dependence on old experiment folders; two fresh full-pipeline replications reproduce the saved scores, objectives and canonical references bit-exactly.
- Scientific self-review: distinguish equal preliminary loss from final compactness; report no eligible distinct pairs rather than claiming robustness at strict tolerances; common-scale loss improves20.86% and must not be called nearly equal. The material contrast persists between two optimal TSP paths and does not reverse Dnet sign.
- Both uninserted LaTeX proposals and the green compact table compiled successfully using pdflatex in a temporary document. Original manuscript, package and data remain unchanged.

- Task6 complete: new Desktop directory copied and byte-verified; ZIP and artifact manifest generated in the delivery action. Existing destinations were rejected rather than overwritten.

# Historical-KDE rescoring of the reference-sensitivity study

**All 360 requested evaluations were rescored from verified saved reference configurations. No OT, clustering, medoid/matching optimization or TSP search was repeated.** C32 was excluded. Three saved baselines were rescored separately. These results use the historical KDE on the existing manuscript-oriented references; they do not reproduce the entire historical weighted-clustering/Euclidean-medoid pipeline or its Monte Carlo averages.

## Method and reuse

C16 (N=144,n=16), Meuse (N=155,n=10) and Swiss (N=959,n=100) retain their original fixed UP samples, seeds, 40 A/B/C outcomes and numerical hardening. A varies TSP search conditional on a fixed OT coupling and preliminary hard labels; B varies saved OT numerical routes under a deterministic TSP rule; C uses the corresponding B coupling with independent TSP seeds. All translated arrays were already saved and reconstruct bitwise as `coords + (coords[matched]-coords[medoids])[labels]`. State/summary fingerprints and reference hashes are recorded in preflight.json and results.csv; B/C coupling reuse and A's fixed coupling/preliminary labels were verified.

Normalization exactly matches the historical Population function: subtract the original per-coordinate minimum and divide all coordinates by the largest original coordinate span. Saved translated coordinates are **not normalized again**. Unweighted sklearn `KernelDensity(kernel="tophat", bandwidth="scott", rtol=1e-4)` uses Euclidean balls with scalar h=N^(-1/(d+4)); h is numerically identical for original and translated fits. The rtol setting matches the historical source. Default rtol=0 produced bitwise identical density arrays and scores for all 363 states.

Dnet and eta use the existing per-unit clipping, theta0=pi/8 and native-position density evaluations; DI=(1-eta)Dnet+eta sign(Dnet), with no added zero threshold. Neither DI definition nor reference construction was changed.

## Baselines

| Population (N,n) | Previous box DI | Historical-KDE DI | Scalar h |
| --- | ---: | ---: | ---: |
| Clustered (144, 16) | 0.365446604 | 0.276619180 | 0.436790232 |
| Meuse (155, 10) | -0.581863932 | -0.275822328 | 0.431464151 |
| Swiss (959, 100) | -0.422272375 | -0.085988697 | 0.318441919 |

## Procedure comparison

SD is the sample standard deviation of the 40 DI values (denominator 39); it also equals the SD of differences from a fixed baseline. Max is the maximum absolute deviation from the historical-KDE baseline. Flips count np.sign differences from that baseline, including any zero transitions.

| Population | Procedure | Previous box SD | Historical-KDE SD | Historical-KDE DI range | Max | Flips |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Clustered | A | 0.008878736 | 0.003794362 | [0.276619, 0.293681] | 0.017062 | 0 |
| Clustered | B | 0.001324134 | 0.001226940 | [0.274079, 0.276619] | 0.002540 | 0 |
| Clustered | C | 0.004220199 | 0.002879318 | [0.274026, 0.283761] | 0.007141 | 0 |
| Meuse | A | 0.004420932 | 0.001029656 | [-0.275822, -0.269310] | 0.006512 | 0 |
| Meuse | B | 0.000000000 | 0.000000000 | [-0.275822, -0.275822] | 0.000000 | 0 |
| Meuse | C | 0.004420932 | 0.001029656 | [-0.275822, -0.269310] | 0.006512 | 0 |
| Swiss | A | 0.022235038 | 0.008028841 | [-0.106772, -0.070290] | 0.020783 | 0 |
| Swiss | B | 0.011025456 | 0.002308552 | [-0.087118, -0.075647] | 0.010342 | 0 |
| Swiss | C | 0.019282352 | 0.006475150 | [-0.094809, -0.064337] | 0.021651 | 0 |

All 360 signs agree with their own recalculated baseline **and** their corresponding previous box-KDE evaluation: zero strict reversals, zero sign mismatches and zero zero-transitions. The retained 360 box-KDE evaluations already had zero baseline flips; absence of the old C32 reversal example results from excluding that scenario, not from an evaluated historical-KDE correction to C32. C32 was not rescored.

## Interpretation and numerical behavior

Observed spread decreases in every group with nonzero previous SD; Meuse B remains invariant over its saved outcomes. Baseline magnitudes nevertheless change substantially, especially Swiss; the largest paired change across the retained evaluations is 0.375517061. This is a change of the **full KDE specification** (kernel geometry and bandwidth scaling), not an isolated kernel-shape experiment. It is not justified to transfer old effect sizes or describe the discrepancy as numerically immaterial.

Reference invariance is still contradicted conditionally. In C16, the baseline and A02 share exactly the saved OT coupling and preliminary labels, but differ in two aligned final hard assignments and one medoid. Their historical-KDE DIs are 0.276619180 and 0.283760614 (absolute difference 0.007141434). Their open-TSP gaps are 0% and 1.891180% relative to the saved exact benchmark for their common centroid instance. Both have native/common OT gaps of 1.041835%/0.617120% relative to best-observed saved objectives. They therefore qualify at the **2% screen, not 1%**; OT optimality is not certified. This is a modest quantified dependence, not a demonstrated sign reversal or a universal stability bound. The unchanged quality screens and all 7,380 comparisons are supplied separately.

Swiss variation retains the saved numerical hardening ambiguity around coincident/near-coincident centroids. KDE rescoring does not resolve it or establish that all saved numerical outcomes represent distinct mathematical OT optima. A comparisons remain conditional on the saved preliminary hardening. The new values do not inherit the numerical sizes of the earlier box-KDE tight-control effects.

No original or translated density is zero, and no nonfinite score occurs. Every density includes its own unit. The minimum observed absolute Dnet is 0.010170379. No evaluated Dnet is zero, and no within-scenario sign change is observed. Radial top-hat counts remain discontinuous at the radius boundary; no pair lies exactly on it (minimum distance-to-radius margin 2.77e-08 in normalized units). Direct counts confirm all density arrays. These checks do not establish continuity under arbitrary perturbations.

These are descriptive conditional search summaries, not sampling standard errors or confidence intervals. Repeated references and dependent B/C trajectories are retained. Three fixed samples cannot establish population frequency, worst-case robustness, statistical consistency or stability of the manuscript's Monte Carlo averages.

## Verification and author action

All 360 primary evaluations and 3 baselines passed direct radial-count and original box-count checks, extended-precision DI checks, and independent individual-row summary verification. The largest score discrepancy was 1.18e-15; no signs differed. Original source/study/package/manuscript hashes were unchanged for all 5,387 protected files. Independent review additionally recalculated optimization gaps, eligible-set counts and all pairwise flags from source summaries.

**Revise the Comment 1 response and Appendix.** Replace the 480/four-setting design, excluded C32 examples and box-specific effects with the 360/three-setting design and the more limited conclusions above. The qualitative need to acknowledge reference dependence remains; the old stronger empirical example cannot be carried over. Proposed replacements are separate files awaiting author approval. No original manuscript, response, package, figure, dataset or saved study file was edited.

Files: results.csv (360 individual paired evaluations), baselines.csv (three controls), summary.csv, appendix_table.tex, appendix_discussion_proposed.tex, reviewer_comment1_recommendation.md, near_best_summary.csv, pairwise.csv, config.json, preflight.json, verification.json/csv and per-evaluation densities/summary records. HANDOFF.md contains inspection/resume commands.

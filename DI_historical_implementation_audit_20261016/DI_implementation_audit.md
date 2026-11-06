# Historical DI implementation audit

16 October 2026. Scope: the reported Measure simulation figures and their available computational provenance. “Reported results” below means manuscript/submission artifacts, not a claim that the paper has been accepted or published. Originals and the completed 480-run study were read only. No samples, clustering solutions, TSP searches or DI evaluations were recomputed.

**Recommendation:** retain the existing results as preserved research evidence, but do not certify them as results of the currently described OT implementation. The historical evidence supports weighted k-means as the preliminary stage for a substantial, identifiable portion of the legacy medoid results. That was consistent with an older Measure draft; the mismatch concerns the later OT description. Resolve the implementation description, replication counts and n-inference uncertainty before claiming full methodological correspondence. This audit does not justify replacing all figures or rerunning the full study.

## A. Is a discrepancy confirmed?

**Yes, for the traced legacy medoid results versus the current OT description. Not established as a blanket attribution of every reported cell or every submission version.** This conclusion now rests on historical execution outputs, saved tables and figure content, not simply on a package version containing weighted k-means.

| Evidence | Direct observation | What it establishes |
| --- | --- | --- |
| Current Measure snapshot, `DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:319–363, 481–515`; Algorithm A1 at line 1380 | Preliminary clustering solves entropic OT with row masses pi and column masses one, alternating Sinkhorn and centroid updates. The DI section initializes that procedure at the observed sample. | OT is explicitly required by this description; it is not just a possible implementation. |
| Submitted `JSSAM-2025-203.R1_Proof_hi_0.pdf`, PDF page 15 | “IPnM follows the structure of k-means with k=n, but replaces exclusive assignments by an optimal-transport (OT) formulation…” | The OT wording also occurs in the supplied publisher-generated R1 proof, not only in a local October draft. |
| Older `/home/bardia/latex-projects/GFS-Spread-Sample-JSSAM/1- Measure.tex:206–228` | “The algorithm begins … using a weighted k-means approach,” followed by TSP ordering and probability-balanced refinement. | Weighted preliminary clustering was an explicitly described earlier method. The evidence does not establish that the historical computation contradicted that earlier draft. |
| Commit `e71d6935bfe824e72a7b722067a24775ed160610`, 1 June 2026, `simulations/jupyters/measure_calculation_nmedoids_clean.ipynb`, cell 7 | `DensityDisparityFlexible._process_one_sample` selects `init_clust_method="weighted"`, passes observed sample coordinates, then constructs medoids and DI. `build_scorers` actually selects this subclass. | A historical simulation execution path independent of the package default. |
| Same notebook, retained outputs in cells 19 and 20 | 106 distinct Monte Carlo cells retain DI means/SDs and `Iterations=2000`. Of these, 104 match the legacy summary exactly at its stored six-decimal precision; two means differ as documented below. | Strong, specific execution-record support for weighted provenance of those matching results. Notebook source/output association is evidence, not an immutable runtime trace. |
| Same commit, `simulations/jupyters/results/DI_nmedoids_selected_populations.csv` and `simulations/results/master_summary_DI_nmedoids_updated.csv` | All 162 DI mean/SD pairs agree exactly between plotting input, updated master and current `simulations/results/All_DI_nmedoids_all_desings.csv`. The separate 138-row Monte Carlo summary also agrees exactly. | Direct numerical continuity between historical summaries and the currently available legacy result table. This does not independently identify the algorithm for every row. |
| Same commit, three `simulations/plots/*_measure_medoid.pdf` figures | They are byte-identical to today's corresponding local plots, and to copies in several supplied manuscript archives. | Historical figure continuity. Later FINAL archives have different PDF bytes, so hash equality alone cannot link them. |
| Synthetic figure: June-1 plot, later `Spread_Measure_Submission_FINAL.zip` figure, and R1 proof PDF page 31 | Independent extraction of all 96 DI bar heights agrees with the June-1 medoid master values within 0.0001 DI units, allowing PDF coordinate quantization. | Content-level support connecting these reported synthetic figures to the legacy medoid values, despite differing PDF bytes. This is a rendering check, not six-decimal numerical recovery or a provenance check for every other figure. |

Reproducible calculations and exact commit hashes are in [checks.json](checks.json), [figure_checks.json](figure_checks.json), [matched saved outputs](evidence/matched_saved_notebook_outputs.csv) and [archive figure matches](evidence/archive_figure_matches.json). Historical notebooks are extracted under `evidence/` with zero-based cell numbers and saved execution counts.

Two qualifications matter. First, 32 Monte Carlo cells lack comparable retained execution outputs in this notebook; the other 24 rows in the 162-row table are NMS/GMS design summaries with a separate scoring/import path. Their algorithm provenance cannot be certified merely by matching a table. Second, notebook execution order is nonsequential: the medoid replacement cell has execution count 40 while a synthetic plotting cell has count 27. Source order alone therefore cannot establish what was plotted. The independent bar-content check supplies a stronger link for the synthetic figure.

### What weighted k-means actually did

The traced historical chain is:

`Monte Carlo samples -> build_scorers -> DensityDisparityFlexible.score -> subclass _process_one_sample -> FIPBalancedNMeans.fit -> weighted preliminary clustering -> open-TSP ordering -> cumulative-IP refinement/hardening -> medoids -> matching -> translations -> KDE -> DI`.

Weighted k-means is the **preliminary clustering algorithm**, fitted to population coordinates with inclusion probabilities as weights. Normally the observed sample supplies its starting centers. It is neither the entire IP-balanced construction nor merely an initializer for a subsequent OT solve: this selected branch returns its labels/centers directly, and the downstream stages do not call OT. The final balanced refinement/hardening is a separate stage. If inferred K differs from the number of supplied centers, the weighted branch instead ignores those centers and selects among random k-means restarts; sample adaptation then fails in the intended sense.

The name `init_clust_method` is therefore potentially misleading: it selects a preliminary clustering method, not simply how the same OT solver is initialized. Likewise, the saved `Dm_kmeans`/`Ds_kmeans` backup columns distinguish earlier centroid-representative DI values from medoid-representative replacements; their names are not evidence of an OT-versus-k-means comparison.

### Where OT and other algorithms occur

Git history shows genuine method evolution: the package DI uses `expanded` at `d047278fc` (6 April), changes to `weighted` at `a6102e6cc` (12 May), and retains weighted at the June-1 medoid revision, release `v1.0.0` and current HEAD. FullPaperGFSS's older Section 2 instead describes probability expansion and `KMeansConstrained`; it is a distinct historical formulation and does not require entropic OT.

The cleaned notebook `01_measure_calculation_spreadness_indices.ipynb`, introduced at `82a7eca7d` (5 June), **does override the scorer to use `"ot"`**. Thus the earlier argument “the package says weighted, therefore all simulations used weighted” was insufficient. However, this later notebook does not displace the June-1 retained outputs and committed figures. It also has an implementation limitation: `FIPBalancedNMeans._get_labels_centroids` constructs grid centers and passes them into its OT branch, ignoring the supplied sample. Its `_get_labels_centroids_ot_kmeans` name denotes Sinkhorn/centroid iteration; it does not prove a weighted-k-means-then-OT hybrid.

Historical NMS/design-construction helpers and illustrative plotting cells explicitly select the `expanded` branch. These are separate from scoring DI for the competing R-generated samples. OT availability elsewhere in the class does not show its use on the traced weighted path. “OTSP” in the old text means **open travelling salesman problem**, not optimal transport. The separate 24 NMS/GMS summary rows first try importing `package_sampling.index.DensityDisparity`; the historical imported module is not pinned, so their precise runtime scorer remains less certain. The present environment's fallback behavior does not prove the historical environment's behavior.

## B. Is the discrepancy scientifically material?

**It is a substantive method difference with potentially material numerical effects; its effect on the manuscript's main design comparisons is not established.**

The conceptual DI aggregation is shared: compare original and cluster-translated densities and combine Dnet with eta through the same signed formula. Both pipelines aim to construct probability-balanced spatial references. They are not, however, numerically interchangeable definitions once the reference algorithm and density estimator are specified. Weighted preliminary clusters need not have OT's prescribed preliminary column masses. Changing labels, ordering, medoids or translations can change DI; matching high-level intent does not prove identical references.

Other confirmed implementation conventions also need to be accounted for. The historical medoid scorer minimizes **Euclidean** weighted medoid distance and uses **Euclidean** Hungarian matching; the current manuscript writes squared objectives. The inherited historical KDE is sklearn's **radial tophat with scalar Scott bandwidth**, whereas the current manuscript specifies a multivariate box kernel and bandwidth matrix. The scored reference is sample-adaptive only when the initial-center count is accepted. Changing only the word “weighted” to “OT” would not reconcile the numerical methods.

Existing `simulations/results/temporary_measure_DI_OT_sample_initialized_M500/` comparisons already show that substituting an OT variant can move reported means: for example, Gridded EP MAX n=4 changes from -0.129560 to -0.388020, and Meuse UP MAX n=10 from -0.014869 to -0.038180. **These are warning evidence, not an isolated causal estimate of the clustering effect.** The 500 new samples are not the original paired samples; target-n handling and numerical routes differ; historical public OT stopping did not certify the stricter feasibility used in the completed 480-run study. Other legacy conventions remain. The OT-versus-expanded comparison is a different comparison again. None proves that the published design rankings reverse, or that they are preserved.

The completed 480 evaluations concern four fixed samples under the separate manuscript-oriented implementation. They establish reference sensitivity within that implementation, not historical provenance, weighted–OT equivalence, or stability of manuscript Monte Carlo averages. Consequently there is no justified conclusion here that all main findings are invalid, and equally no justified guarantee that all findings are unaffected. No new numerical clustering comparison was needed to establish this distinction.

### Two saved-data and replication discrepancies

| Setting | Retained June-1 execution mean | Legacy plotting-summary mean | Difference |
| --- | ---: | ---: | ---: |
| Random EP, MAX, n=4 | -0.089949 | -0.089940 | +0.000009 |
| Meuse EP, LPV, n=20 | 0.034492 | 0.034992 | +0.000500 |

Their SDs agree. These are observed differences, not established causes: no claim is made about whether they arose through transcription, editing or another run. Preserve both records and resolve them against the authoritative original output rather than silently changing values.

For the 106 retained Monte Carlo cells, the notebook says **2,000 DI replications**, while the current manuscript says 5,000 per setting/design. Other indices were inherited from older 5,000-run summaries during the medoid update. A uniform 5,000-replication claim for all these DI values is therefore unsupported. The remaining 32 Monte Carlo cells lack a replication-count column in the clean summary and should not be assigned 2,000 or 5,000 solely by analogy. The 24 design-expectation rows are a different calculation altogether.

## The reported n−1 issue

The defect is present in the **historical implementation relevant to the legacy results**, not merely current source. At `e71d6935b`, `Population.__init__` sets `_n=int(np.sum(inclusions))` when n is omitted. The medoid simulation notebook omits n when constructing `Population`, despite calculating a rounded n separately for sample generation. Its `force_sum_to_n` multiplies by `n/pi.sum()` but does not guarantee an exactly integer floating sum.

An arithmetic-only reconstruction using those historical class/function definitions, the stored population vectors and today's installed probability helper/NumPy 2.4.4 gives:

| Example | Target n | Sum after historical normalization | Inferred n |
| --- | ---: | ---: | ---: |
| Clustered EP | 16 | 15.999999999999993 | 15 |
| Clustered EP | 32 | 31.999999999999986 | 31 |
| Clustered UP | 16 | 15.999999999999998 | 15 |

It triggers a shortfall in 18 of 36 population/probability/sample-size settings. Applying the listed competing-design counts would flag **70 candidate cells**, matching the earlier HTML's structural count. This is a reconstruction, **not an independently verified count of affected original simulations**. See [the complete reconstruction](evidence/historical_n_inference_reconstruction.csv).

If this occurred during a run, the historical weighted initializer's center-count check would reject the n sample centers, use K=n−1 random-start preliminary clusters, and the rectangular reference–sample matching would leave a sample unit unused. Thus it can affect reference cardinality, sample adaptation and DI, not just a displayed cluster-count label. The saved summaries retain the requested n, not the realized K, so their n column cannot rule out the defect.

No per-run K, accepted-initial-center flag, original probability-vector snapshot or complete original scoring trace was found for these Monte Carlo cells. The exact historical helper/library environment is also not frozen per execution. Therefore original incidence, affected DI differences and consequences for design rankings remain **not established**. The earlier HTML claims such incidence, but its cited `simulations/legacy_DI_variability/` scripts/raw results are absent here; reproducing the mechanism and the number 70 does not independently validate that report's run-level claims.

## C. Can the existing manuscript results be retained?

**Potentially yes, with conditions; unconditional certification is not justified yet.**

Retaining the legacy numerical results is scientifically defensible if their evaluated implementation is accurately identified, their replication counts/data discrepancies are resolved, and claims are restricted to that implementation. Weighted preliminary clustering followed by balancing is a coherent historical construction, explicitly described in the older Measure draft. It need not be discarded merely because OT was introduced in a later description.

However, results from that variant must not be presented as numerical validation of sample-initialized entropic OT, squared medoids/matching and box KDE without supporting evidence. Also, if original runs did infer K=n−1 and lose sample initialization, merely relabelling their preliminary method would not justify treating them as intended n-reference results. Recover original diagnostics or otherwise resolve that specific issue before relying on affected quantitative claims. Existing reviewer-driven sample-adaptive/medoid revisions should be preserved; this audit does not authorize reverting them or silently redefining DI to fit an old figure.

## D. Minimum necessary correction

1. **Correct the provenance account first.** Identify the legacy variant and the later OT variant separately, with version/source and precise conventions. The current AUDIT.md/REPORT.md accurately describe package code, but their package observations alone should not be treated as proof about every published calculation. This new audit supplies stronger, scoped evidence.
2. **Align methods and results without automatically replacing simulations.** If legacy results are retained, disclose the actual weighted preliminary stage, Euclidean representative/matching criteria and radial KDE where applicable, rather than describing those values as current manuscript OT results. Keep the OT construction and the completed sensitivity study explicitly separate unless validated equivalence is available. Any author/editor decision about the final presentation must preserve prior reviewer requirements.
3. **Resolve the two saved means and the replication-count statement** against original output; do not overwrite historical records or claim a uniform 5,000 DI runs without evidence.
4. **Resolve target-n handling for the historical runs.** For future reproducibility, pass the known target n consistently and verify final reference cardinality and accepted sample initialization. Such a correction would change the numerical pipeline and must be documented; it was not applied to the original package or existing results here.

This is a scoped reporting/provenance correction, not a recommendation to redesign DI. It does not assert that changing one word repairs every difference, nor that every legacy result must be replaced.

## E. Is additional simulation genuinely necessary?

**No additional Monte Carlo is needed to establish the provenance findings above, and a full rerun is not justified by this audit alone.** Existing artifacts already establish the traced method difference and the uncertainties. Repeating simulations cannot reconstruct missing historical execution metadata.

The next useful evidence is the original raw outputs, environment/import identity, probability vectors and reference-count logs, especially for the candidate n−1 settings and the two means. If the authors want to assert that the existing conclusions hold specifically for the current OT method, a small paired check on identical saved samples from a clustered case and Meuse would then be warranted, varying only the preliminary algorithm and separating target-n and KDE/medoid changes. Its purpose would be to decide whether further validation is necessary—not to assume another full study. Current evidence does not establish that equivalence, and this audit has not manufactured a new comparison to fill that gap.

**Action now:** review this implementation/provenance clarification and recover the specific missing records. Keep all existing results preserved. Do not reopen the full simulation study or modify the manuscripts automatically.

## Reproducibility and preservation

Run the checks from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python DI_historical_implementation_audit_20261016/audit_checks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python DI_historical_implementation_audit_20261016/figure_checks.py
.venv/bin/python DI_historical_implementation_audit_20261016/verify_preservation.py
```

`audit_checks.py` reads Git blobs, summaries and notebook outputs and reconstructs only population-size arithmetic; it never executes simulation notebook cells. `figure_checks.py` reads saved SVG geometry. Existing PDF/SVG extractions and exact source snapshots are under `evidence/`; their preparation and input hashes are recorded separately. All computation uses one thread, well below the approximately 55% CPU restriction. No network data, new sampling, clustering, TSP or DI runs are needed. Preservation evidence and reviewer findings are saved in this audit directory; the prior HANDOFF.md and final 480-run artifacts remain unchanged.

# Final historical implementation audit

17 October 2026. **The blanket 5,000-replication statement is unsupported for the updated DI results. The strongest source/output link instead identifies 2,000-requested computations with ordinary Euclidean matching and spherical top-hat KDE. The manuscript's squared-matching translation argument cannot be transferred to that historical matching objective.** Attribution remains incomplete for some rows; the distinctions below matter.

No Monte Carlo experiment, clustering, KDE evaluation or DI rescoring was performed. The completed paired kernel experiment was read only. Proposed corrections are separate, approval-only files; no manuscript or original result was edited.

## A. Replication count

The audit follows the **162 settings** in historical `simulations/jupyters/results/DI_nmedoids_selected_populations.csv`, through the June-1 commit `e71d6935bfe824e72a7b722067a24775ed160610`, the updated master, and today's `All_DI_nmedoids_all_desings.csv`. All 162 final DI mean/SD pairs agree across those tables. The master has four additional rows outside this scope. This establishes table continuity, not execution provenance for every row.

| Updated DI settings | Number | Requested / recorded | Completed and used in final statistics |
|---|---:|---|---|
| MC outputs exactly matching final mean/SD | 104 | 2,000 requested and printed | 2,000 scores strongly supported by successful-output dataflow; no independent raw completion log |
| MC outputs with a discrepant final mean | 2 | Associated output prints 2,000 | Final mean's origin/count unresolved |
| MC settings without retained setting output | 32 | Relevant source requests 2,000 | Actual final count unresolved |
| NMS/GMS finite designs | 24 | Weighted enumeration | 21,368 total valid enumeration entries; design-specific counts, not MC repetitions |

For the 138 MC settings, each listed design applies at each listed sample size:

| Population / probabilities | n | Designs | Exact 2,000-output links | Discrepant | Unlogged |
|---|---|---|---:|---:|---:|
| Clustered / EP | 4,8,16,32 | MAX,SCP,LPV,WAV | 16 | 0 | 0 |
| Clustered / UP | 4,8,16,32 | same | 0 | 0 | 16 |
| Gridded / EP and UP | 4,8,16,32 | same | 32 | 0 | 0 |
| Random / EP | 4,8,16,32 | same | 15 | 1 | 0 |
| Random / UP | 4,8,16,32 | same | 0 | 0 | 16 |
| Meuse / EP and UP | 5,10,20 | same | 23 | 1 | 0 |
| Swiss / EP and UP | 50,100,150 | MAX,SCP,LPV | 18 | 0 | 0 |

The two differences must not be silently corrected:

| Setting | Retained output mean | Final table mean | SD in both |
|---|---:|---:|---:|
| Random / EP / MAX / n=4 | −0.089949 | −0.089940 | 0.347306 |
| Meuse / EP / LPV / n=20 | 0.034492 | 0.034992 | 0.053606 |

**Requested is not independently observed completed.** In historical `measure_calculation_nmedoids_clean.ipynb`, cells 19/20 set 2,000, generate a full `(num_samples,n)` buffer, validate it without dropping rows, score the entire buffer, and calculate mean/sample SD (`ddof=1`). A task error aborts rather than returning fewer scores. `Iterations` is assigned the requested constant. This supports the 104 linked counts, but retained initial `0/2000` progress displays are not completion records. Original MC buffers and per-replication scores were not recovered. See the [numbered notebook](evidence/replications/historical_medoid_notebook.txt), lines 954, 1031, 1049–1051 and 1493, 1575, 1593–1595; validation is at 605–635, and inherited scoring at [source lines 112–128](evidence/replications/historical_density_disparity.py).

**5,000 belongs to the older experiment.** The March-15 ancestor `1f376e9fc1394feac1d02888b0ba0053fb48bceb` already contains the `equal_M=5000` / `unequal_M=5000` summary files. All 138 selected MC rows record `ite=5000`. The June-1 plotting notebook loads these older values, then replaces **only DI mean/SD** with the medoid update. All 138 final MC auxiliary-index pairs match those older summaries after four documented LB SD clippings to 1.5. Thus the comparison mixes stages. Old count metadata and generator source support 5,000 nominally; missing raw counts and possible pandas missing-value exclusion prevent independent certification of 5,000 valid observations per index. A later retained 5,000-requested exploratory run was interrupted and is not the source of these final rows. See [plot source](evidence/replications/historical_plot_notebook.txt):2400, 2467–2490, 2869, 3007–3008.

The 24 NMS/GMS DI summaries use normalized probability-weighted means and **population SDs**, not MC sample SDs. Static numeric-cache inspection verifies all 21,368 entries contain exactly n distinct valid units and positive finite probabilities. Enumeration counts range from 31 to 3,688 and can include repeated unordered sample sets. Their [per-design counts](evidence/replications/finite_design_support_counts.csv) and [setting ledger](evidence/replications/per_setting_replication_ledger.csv) preserve this distinction. Their actual scorer import identity remains unresolved (Section B). Two Swiss-UP-n50 auxiliary summaries are missing because older rows have the wrong N; details are in the [replication findings](evidence/replications/replication_findings.md).

## B. Matching objective and translation

**Recovered source uses ordinary Euclidean assignment:** `cost = cdist(reference_points, raw_sample)` is passed directly to `linear_sum_assignment`. There is no squaring or assignment weight. The weighted medoid objective also uses ordinary Euclidean distances. This is independent of the squared loss used upstream in clustering.

| Source | What is established |
|---|---|
| April 6 `d047278fc`, package lines 78–84 | Unsquared centroid matching; expanded preliminary method |
| May 12 `a6102e6cc` | Unsquared matching; weighted preliminary method |
| June 1 `e71d6935b`, package lines 94–102, 158–164 | Unsquared reference matching and weighted Euclidean medoids |
| June-1 medoid notebook, cell 7 lines 77–80, 97, 113–121; cell 12 lines 75–79 | Executed scorer override selects weighted preliminary clustering, ordinary medoids and ordinary matching; associated with 104 exact-linked MC pairs |
| June 5 `82a7eca7d`, notebook01 cell 8 | OT override exists, but matching and medoids remain ordinary; its retained test uses 10 repetitions, with zero exact legacy mean/SD matches; 2,000-run cell has no retained execution count or outputs |
| Release `7c61dbeaf` and audited HEAD `af36c9e3f` | Ordinary matching persists |

Full commits, paths and line anchors are in [matching_history.csv](evidence/matching/matching_history.csv) and [findings](evidence/matching/findings.md). A separate nonancestor September OT/centroid branch also uses ordinary matching; its upstream `sqeuclidean` cost is not the Hungarian cost. No squared-matching override was found in the inspected historical notebooks; an LFS-pointer source and any unsaved/external overrides remain outside that conclusion.

The best numerical/source association is the **104 linked MC rows**. The 32 unlogged rows cannot be certified by analogy. The 24 numerically linked NMS/GMS rows first import `package_sampling.index.DensityDisparity`, with graphical-sampling only as fallback; historical import identity is not pinned. Three committed medoid figure PDFs remain byte-identical to June 1. The prior audit also linked synthetic DI bars by content. Nonsequential notebook execution counts prevent treating source order alone as figure-generation proof. These limitations are recorded in the [162-row provenance table](evidence/matching/row_provenance.csv).

**Minimal deterministic counterexample.** Fix medoids r₁=(0,0), r₂=(2,0), sample s₁=(0,0), s₂=(1,1), and translate only the sample by δ=(3,0). Identity pairs like-numbered points; swap crosses them.

| Objective | Original identity | Original swap | Shifted identity | Shifted swap |
|---|---:|---:|---:|---:|
| Sum of distances | √2 = 1.414214 | 2+√2 = 3.414214 | 3+√5 = 5.236068 | 1+√17 = 5.123106 |
| Sum of squared distances | 2 | 6 | 14 | 18 |

The **unique Euclidean optimum switches identity→swap**. The strict shifted gap is 0.112962351882129…, verified with 70-digit arithmetic and SciPy. Dividing all coordinates by five gives the same reversal inside the unit square. Cluster shifts consequently do not all change by δ. This disproves general assignment invariance; it does **not** prove every such rematching changes DI.

For a fixed reference and a full unweighted bijection,

\[
Q_\delta(\tau)=Q_0(\tau)+2\delta^\top\left(\sum_j s_j-\sum_i r_i\right)+n\|\delta\|^2.
\]

The squared-cost increment is permutation independent. Its minimizing set is unchanged; consistent tie handling is needed for the selected permutation. This does not automatically cover **rectangular matching**: historical `int(sum(pi))` can infer K=n−1 in the [previous reconstruction](../DI_kernel_comparison_JSSAM_20261017/REPORT.md), allowing different sample subsets to be matched. This identifies a source-level cardinality risk, not proof of an actual n−1 reference in each old run. No historical medoid counts were reconstructed here from missing MC buffers.

With partition, medoids **and pairing held fixed**, common sample translation shifts the whole transformed population equally; either fixed-bandwidth kernel preserves its values at corresponding transformed units, so the actual DI formula is unchanged. Recomputing the partition/reference can break this condition under either assignment metric. Jointly moving the reference and sample is another, simpler invariant operation. A sample that is itself a translated copy of the medoid pattern also has a Euclidean-optimal natural pairing by the triangle inequality, explaining some regular examples without establishing a general theorem.

Therefore Section 3 and Appendix Algorithm 2 currently describe a different assignment/medoid objective from the traced historical implementation. Comment 5 must state the fixed-reference/full-bijection/tie conditions for its squared-assignment step, and cannot use that step to justify general Euclidean rematching. The recent EP/UP examples used explicit squared matching and saved unchanged reference memberships; they remain valid observations for their saved configurations, not historical replays. Older Figure 9's rounded DI=0.00 likewise supplies empirical, not universal, evidence. Grid-projected rotated samples are not exact rotations. See [proof and qualifications](assignment_mathematics.md), [reproducible diagnostic](assignment_counterexample.py), and [costs](assignment_costs.csv).

## C. Kernel

**Spherical top-hat remains the best-supported historical kernel, from code and execution-path evidence.** All six inspected package snapshots specify `KernelDensity(kernel="tophat", bandwidth="scott", rtol=self.kde_rtol)` with default tolerance 10⁻⁴. The June-1 selected notebook scorer inherits both density evaluation and the DI aggregation; it overrides neither. Its imported base and `build_scorers` path are explicit. See [kernel history](evidence/kernel/kernel_history.csv) and [execution-path record](evidence/kernel/historical_execution_path.json).

This KDE uses scalar h=N⁻¹⁄⁽ᵈ⁺⁴⁾ and H=h²I on the coordinates supplied to it, with no covariance scaling or translated-population renormalization. The recovered backend's ball boundary convention is strict `<h`; exact boundary cases can matter for grids. It is not the infinity-norm box kernel written in the available manuscript.

The existing paired experiment's box means are closer for all four n, while SD agreement is mixed; its approximate aggregate comparison interval includes zero. Such closeness neither identifies the historical runtime kernel nor warrants replacing original values. Source selection is confirmed; runtime use for the traced MC outputs is strongly supported, **not immutable environment provenance for every bar**. Missing environment bindings, the 32 unlogged MC settings, the two altered means and the NMS/GMS import ambiguity remain limitations.

## D. Proposed manuscript corrections — approval only

[PROPOSED_CORRECTIONS.md](PROPOSED_CORRECTIONS.md) quotes **15 exact existing passages** with source lines, minimal replacement bodies and applicability. [proposed_corrections.tex](proposed_corrections.tex) supplies review fragments; it is not a combined patch or a file to input wholesale. It covers Section 3 medoids/matching/KDE and ties, Section 4 counts, Appendix Algorithm 2, the translation discussion, abstract, introduction, conclusion and numerical-example wording. An optional Comment 5 proof clarification is clearly identified as new text, because the exact current proof was unavailable.

Two particularly consequential existing passages are:

```latex
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm r_i\|^2,
```

and:

```latex
\citep{sampling_pkg}. For each reported population, IP setting, sample size,
and competing design, \(5{,}000\) Monte Carlo samples were generated.
```

For historical-method alignment, the first loses its exponent and the second needs mixed-stage count disclosure, rather than substitution of a universal 2,000. However, deleting exponents alone is insufficient: weighted preliminary clustering, historical medoid ties and spherical KDE also differ from the formal modern OT/squared/box specification.

The fragments distinguish **A: describe the traced historical variant**, from **B: retain the modern formal specification and explicitly label the old results as a different implementation**. Neither authorial choice is applied automatically. The squared clustering objective in Section 2 must not be changed merely because downstream distances differ.

**Quotation limitation:** the latest located designated source is the 12-October snapshot (`DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex`, SHA256 `d91765e63c588fa646be268ca62d8fed8b69fd5bec99995f8bfb6a1c2c8a5baa`). No authoritative current source containing `RevThree`/`RevFive` or connected document session was found. Thus exact quotations are verified against that snapshot, not represented as the latest colored revision. Proposed bodies are unwrapped so the real revision-color groups can be retained at integration. No colors, braces or original files were changed.

## E. Overall conclusion and remaining evidence

| Issue | Classification | What would settle the remaining uncertainty |
|---|---|---|
| One common 5,000 count for updated DI | **Confirmed unsupported** | Original per-replication records could establish exact effective counts; cannot erase the mixed-stage source trail |
| 2,000 scores for 104 linked MC settings | **Strongly supported / inferred** | Original raw buffers, score vectors or completed-count logs |
| Two discrepant means and 32 unlogged MC settings | **Unresolved** | Original generating executions, unrounded results and edit history |
| NMS/GMS enumeration counts | **Confirmed** | Counts/statistics source and numeric caches agree; actual scorer environment still needed |
| Ordinary historical matching and medoids | **Confirmed source; strongly supported runtime for traced MC** | Frozen environment/runtime records for a blanket claim across all rows |
| Euclidean assignment translation invariance in general | **Confirmed false** | Strict two-pair counterexample; no further experiment needed |
| Conditional DI translation with fixed partition/reference/pairing | **Confirmed mathematically** | For an application, verify those conditions and admissible sample locations |
| Spherical historical KDE | **Confirmed selected source; best-supported runtime** | Original environment/module bindings, particularly `package_sampling` |
| Numerical effect of replacing matching/KDE on every old result | **Unresolved; not measured here** | A separately authorized, faithfully specified reanalysis |
| Latest revision-colored quotations / exact Comment 5 | **Unresolved** | Authoritative latest colored TeX and response source |

The historical evidence does not support claiming that the existing formal manuscript describes all retained simulations. It supports a scoped historical variant and conditional invariance, with explicit unresolved provenance. Review the separate corrections before any integration.

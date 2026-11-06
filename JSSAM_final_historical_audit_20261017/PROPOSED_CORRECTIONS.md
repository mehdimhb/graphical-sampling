# Proposed LaTeX corrections — approval only

No manuscript or response has been changed. Exact quotations below are checked against the last designated 12 October source snapshot, not a verified latest colored revision. The actual current `RevThree`/`RevFive` source and exact current Comment 5 proof were not recovered. These bodies are deliberately unwrapped: preserve the real enclosing revision groups, definitions and closing braces when integrating. Do not invent or replace color definitions.

**Decision required:** A describes the traced historical weighted/Euclidean/spherical DI variant; B retains the modern OT/squared/box definition and explicitly separates the old numerical results. They are alternatives, not one combined patch. A does not repair historical reference-count defects or certify the untraced rows. Conditional translation wording and replication disclosure are needed under either choice. The squared clustering objective in Section 2 is separate from the medoid and matching objectives and must not be changed merely by deleting their exponents.

The supplied `.tex` is a collection of numbered review fragments, not an executable patch or a file to `\input` wholesale. Complete provenance is in REPORT.md and the row-level ledger.

## section3_weighted_medoid

**A: describe traced historical implementation**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:501–509`.

Exact existing LaTeX:

```latex
\[
m_i
\in
\arg\min_{u\in U_i}
\sum_{\ell\in U_i}\pi_\ell
\|\bm c_u-\bm c_\ell\|^2,
\qquad
\bm r_i=\bm c_{m_i}.
\]
```

Proposed replacement:

```latex
\[
m_i
\in
\arg\min_{u\in U_i}
\sum_{\ell\in U_i}\pi_\ell
\|\bm c_u-\bm c_\ell\|,
\qquad
\bm r_i=\bm c_{m_i}.
\]
```

Only remove exponent 2. The clustering objective in Section 2 remains squared Euclidean; do not change it. Preserve the enclosing Comment 3 / RevThree group if a current colored source is recovered.

## section3_assignment

**A: describe traced historical implementation**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:545–556`.

Exact existing LaTeX:

```latex
 The sample units are then matched to the medoid reference points by solving the
assignment problem
\[
\tau
=
\arg\min_{\tau}
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm r_i\|^2,
\qquad
\tau(i)\in s,
\]
where \(\tau\) is a one-to-one mapping from clusters to sampled units. Each
```

Proposed replacement:

```latex
 The sample units are then matched to the medoid reference points by solving the
assignment problem
\[
\tau
=
\arg\min_{\tau}
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm r_i\|,
\qquad
\tau(i)\in s,
\]
where \(\tau\) is a one-to-one mapping from clusters to sampled units. Each
```

Only exponent removal is substantive; retain any actual current revision-color wrapper.

## section3_historical_kernel

**A: describe traced historical implementation**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:586–597`.

Exact existing LaTeX:

```latex
where \(\mathbf H\) is a \(d\times d\) symmetric positive-definite bandwidth
matrix, selected here using Scott's rule. We use the multivariate uniform
kernel
\[
K(\bm u)
=
\frac{1}{2^d}
\mathbf 1\{\|\bm u\|_\infty\leq1\},
\qquad \bm u\in\mathbb R^d,
\]
where \(\|\cdot\|_\infty\) denotes the infinity norm. Applying the same
estimator to the translated coordinates
```

Proposed replacement:

```latex
where, for the traced historical computations,
\(\mathbf H=h^2\mathbf I_d\) with \(h=N^{-1/(d+4)}\), the isotropic
Scott bandwidth. The spherical uniform (\texttt{tophat}) kernel is
\[
K(\bm u)=\frac{1}{v_d}\mathbf 1\{\|\bm u\|_2<1\},
\qquad v_d=\frac{\pi^{d/2}}{\Gamma(1+d/2)}.
\]
The same numerical bandwidth is used for the original and translated
coordinates, without covariance scaling or renormalizing the translated
population. The historical KDE evaluation used relative tolerance
\(10^{-4}\). Applying the same estimator to the translated coordinates
```

The strict ball boundary follows the recovered backend; exact boundary distances can matter on a grid. The existing matrix KDE definition can remain with H=h^2I specified. Upstream Population normalization is separate from this scalar-bandwidth KDE.

## section4_replications

**Both A and B**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:977–978`.

Exact existing LaTeX:

```latex
\citep{sampling_pkg}. For each reported population, IP setting, sample size,
and competing design, \(5{,}000\) Monte Carlo samples were generated.
```

Proposed replacement:

```latex
\citep{sampling_pkg}. The reported summaries combine simulation stages.
The auxiliary-index Monte Carlo summaries were inherited from earlier
output files recording \(5{,}000\) replications per setting. The subsequent
medoid-DI update requested \(2{,}000\) samples per setting. Retained outputs
label \(106\) setting summaries with this count; \(104\) reproduce the
reported DI mean and standard deviation, whereas two reported means differ.
For \(32\) other Monte Carlo settings, the count underlying the final DI
summary cannot be established from retained execution records. The nominal
counts are supported by source and summary metadata rather than independently
retained per-replication completion logs. The NMS/GMS DI comparisons use
probability-weighted finite-design enumeration, with the number of enumerated
entries reported separately for each design; they are not Monte Carlo
replications. Accordingly, a common \(5{,}000\)-replication count for every
reported statistic is not supported by the retained records.
```

The counts concern the 162-setting DI plotting input: 138 Monte Carlo settings and 24 finite-design settings. The ledger and final report give the population/design coverage; enumeration entries are not Monte Carlo replications.

## algorithm2_medoid

**A: describe traced historical implementation**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1490–1505`.

Exact existing LaTeX:

```latex
\State  For each cluster \(U_i\), define the reference point \(\bm q_i\) by
\[
\bm q_i
=
\begin{cases}
\bm c_{m_i}, &
\rho=\textsc{medoid},\quad
m_i\in\displaystyle\arg\min_{u\in U_i}
\sum_{\ell\in U_i}\pi_\ell\|\bm c_u-\bm c_\ell\|^2, \\[2ex]
\displaystyle
\frac{\sum_{\ell\in U_i}\pi_\ell\bm c_\ell}
{\sum_{\ell\in U_i}\pi_\ell}, &
\rho=\textsc{centroid}.
\end{cases}
\]
For the medoid option, ties are broken by largest IP and then smallest label.
```

Proposed replacement:

```latex
\State  For each cluster \(U_i\), define the reference point \(\bm q_i\) by
\[
\bm q_i
=
\begin{cases}
\bm c_{m_i}, &
\rho=\textsc{medoid},\quad
m_i\in\displaystyle\arg\min_{u\in U_i}
\sum_{\ell\in U_i}\pi_\ell\|\bm c_u-\bm c_\ell\|, \\[2ex]
\displaystyle
\frac{\sum_{\ell\in U_i}\pi_\ell\bm c_\ell}
{\sum_{\ell\in U_i}\pi_\ell}, &
\rho=\textsc{centroid}.
\end{cases}
\]
In the traced historical medoid-update implementation, the first minimizing unit in cluster order was selected.
```

Changes the medoid exponent and historical tie convention; the centroid formula remains the weighted mean.

## algorithm2_assignment

**A: describe traced historical implementation**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1508–1518`.

Exact existing LaTeX:

```latex
\State Match the reference points to the sampled units by solving
\[
\tau
=
\arg\min_{\tau}
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm q_i\|^2,
\qquad
\tau(i)\in s,
\]
where \(\tau\) is one-to-one.
```

Proposed replacement:

```latex
\State Match the reference points to the sampled units by solving
\[
\tau
=
\arg\min_{\tau}
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm q_i\|,
\qquad
\tau(i)\in s,
\]
where \(\tau\) is one-to-one.
```



## comment5_translation_qualification

**Both A and B**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:901–917`.

Exact existing LaTeX:

```latex
    \item[(iii)]  The index is translation invariant. Thus, DI evaluates
dispersion through the relative configuration of the sampled units, rather
than their absolute geographic location. Different spatial spread indices
formalize spatial balance from different viewpoints: Moran-based indices focus
on local spatial autocorrelation, whereas Voronoi-based indices focus on
spatial coverage and local representation. DI adds a complementary
configurational interpretation. Because DI uses a sample-adaptive medoid reference configuration, a
well-spread sample is not penalized merely because its spatial arrangement
differs from a single fixed global \(n\)-means solution; any departure
measured by DI is evaluated relative to the sample-induced reference
configuration illustrated in
Figure~\ref{fig:sample_adaptive_reference}. The IP\(n\)M medoids define a reference configuration for an ideally well-spread sample; hence, any matched sample configuration that preserves this medoid reference pattern, up to a uniform
translation, is regarded as equally well spread. In this sense, a sample is
perfectly balanced when it represents the spatial structure of the population
by preserving the relative arrangement of the IP\(n\)M reference medoids. DI is not generally rotation invariant; this limitation and
its interaction with sample-adaptive reference configurations are illustrated
in Appendix~\ref{Exm:rotation_4x5}.
```

Proposed replacement:

```latex
    \item[(iii)] Conditional on a fixed hard partition, fixed reference
points, and a fixed medoid-to-sample pairing, DI is invariant to a common
translation of the matched sample coordinates. Indeed, replacing every
matched sample point by \(\bm c_{\tau(i)}+\bm a\) replaces every transformed
population point by \(\widetilde{\bm c}_\ell+\bm a\). A fixed-bandwidth,
translation-invariant KDE has the same value at each corresponding
transformed unit, leaving the DI formula unchanged. This result does not
guarantee invariance after sample-adaptive reclustering or rematching.
Ordinary Euclidean optimal matching can change when only the sample is
translated. Thus, for that historical objective, unchanged matching is an
additional condition, not a general consequence of unchanged reference points.
The numerical translation examples establish the behavior of the saved
configurations only. DI is not generally rotation invariant; the illustrated
rotated targets are projected to distinct population-grid units.
```

Place inside existing RevFive braces when an authoritative colored source is recovered; do not replace or nest a macro speculatively. Finite-population admissibility of shifted sample coordinates is a separate requirement. Squared matching with a fixed reference has stronger translation behavior of its optimal permutation set, but sample-adaptive reclustering remains outside this theorem.

## section3_keep_squared_alternative

**B: retain squared definition; explicitly separate historical pipeline**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:545–556`.

Exact existing LaTeX:

```latex
 The sample units are then matched to the medoid reference points by solving the
assignment problem
\[
\tau
=
\arg\min_{\tau}
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm r_i\|^2,
\qquad
\tau(i)\in s,
\]
where \(\tau\) is a one-to-one mapping from clusters to sampled units. Each
```

Proposed replacement:

```latex
The formal definition below retains squared-distance representatives,
squared Euclidean matching, and the box kernel specified here. The traced
historical Monte Carlo computations instead used weighted preliminary
\(k\)-means followed by probability refinement, ordinary-distance medoids,
ordinary Euclidean matching, and a spherical \texttt{tophat} KDE.
The retained results must therefore be identified as results of that
historical variant, rather than numerical validation of the present
OT/squared-distance/box-kernel specification. Runtime provenance remains
unresolved for the other blocks identified in the reproducibility audit.

 The sample units are then matched to the medoid reference points by solving the
assignment problem
\[
\tau
=
\arg\min_{\tau}
\sum_{i=1}^n
\|\bm c_{\tau(i)}-\bm r_i\|^2,
\qquad
\tau(i)\in s,
\]
where \(\tau\) is a one-to-one mapping from clusters to sampled units. Each
```

Mutually exclusive with A. Retain squared equations in Section3 and Algorithm2. If retaining cube kernel too, label it as formal definition and retain explicit historical spherical-kernel distinction. This is not a claim to have rerun or validated a future squared pipeline.

## historical_clustering_disclosure

**Both A and B; essential historical-method disclosure**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:488–489`.

Exact existing LaTeX:

```latex
\(\{\bm c_\ell:\ell\in s\}\) as initial centers, we apply the hard IP\(n\)M
procedure to obtain \(n\) spatially compact clusters
```

Proposed replacement:

```latex
\(\{\bm c_\ell:\ell\in s\}\) as initial centers, the formal definition applies
the hard IP\(n\)M procedure to obtain \(n\) spatially compact clusters.
The traced historical Monte Carlo implementation used weighted preliminary
\(k\)-means followed by open-TSP ordering and inclusion-probability refinement,
rather than the entropic OT preliminary stage described in Section~2.
Its retained results are distinguished from computations of the formal OT
variant; exact runtime attribution is unresolved for the other audited blocks.
```

Do not silently replace Section2/Algorithm1 with historical weighted k-means. Historical representative/matching/kernel corrections alone do not align the historical pipeline with the formal modern OT definition. Original e71d6935b DensityDisparity passes init_clust_method="weighted" at lines72-78; older d047278fc class used expanded and centroids. Scope specific figure/code path before universal attribution.

## abstract_translation

**Both A and B**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:143–143`.

Exact existing LaTeX:

```latex
bounded and translation-invariant measure that distinguishes under-dispersion,
```

Proposed replacement:

```latex
bounded measure with a conditional translation property that distinguishes under-dispersion,
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## introduction_translation

**Both A and B**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:233–235`.

Exact existing LaTeX:

```latex
interpreted as approximately well spread. The index is translation invariant, so shifting a sample without changing its internal geometry does not change its value. Thus, DI focuses on the relative spatial arrangement of the sample and
separates under-dispersion, optimal spread, and over-dispersion on a
common interpretable scale.
```

Proposed replacement:

```latex
interpreted as approximately well spread. With the reference partition and matching held fixed, a common translation of the matched sample coordinates leaves DI unchanged. This conditional property concerns the relative arrangement of the matched sample and does not guarantee invariance after sample-adaptive reclustering or rematching.
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## conclusion_translation

**Both A and B**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1120–1121`.

Exact existing LaTeX:

```latex
The proposed index is bounded, translation invariant, and directly
interpretable. Negative values indicate under-dispersion or local clustering,
```

Proposed replacement:

```latex
The proposed index is bounded and directly interpretable, and has the conditional translation property described in Section~\ref{Sec:Measure}. Negative values indicate under-dispersion or local clustering,
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## appendix_translation_example

**Both A and B**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1150–1151`.

Exact existing LaTeX:

```latex
In panel~(b), a uniform translation is applied to all sample units and DI
remains unchanged, confirming translation invariance.
```

Proposed replacement:

```latex
In panel~(b), a uniform translation is applied to all sample units and DI
remains unchanged for this example. This observation alone does not establish invariance after general sample-adaptive reclustering or rematching.
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## section3_historical_medoid_ties

**A: describe the historical notebook tie convention**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:510–512`.

Exact existing LaTeX:

```latex
If the medoid is not unique, ties are broken by first selecting the unit with
the largest first-order inclusion probability; if a tie still remains, the unit
with the smallest population label is selected.
```

Proposed replacement:

```latex
In the traced historical medoid-update implementation, the first minimizing
unit in cluster order was selected.
```

Do not claim the historical notebook used today's np.isclose/largest-IP tie rule. If preserving the modern definition, retain its formal rule and disclose this historical difference instead. Keep any actual RevThree wrapper.

## algorithm2_historical_preliminary_stage

**A only: describe the historical pipeline; do not mix with the formal OT version**. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1485–1487`.

Exact existing LaTeX:

```latex
\State Apply Algorithm~\ref{alg:IP_balanced_nmeans} using its hard-clustering
variant on \(U\), initialized by the sample units \(s\).
\label{line:upbkm}
```

Proposed replacement:

```latex
\State For the historical DI variant, initialize weighted \(k\)-means
at the observed sample units and apply the historical open-TSP ordering and
inclusion-probability refinement to construct the reference partition.
\label{line:upbkm}
```

Do not change the squared clustering loss in Section2. This fragment distinguishes algorithms, does not repair the historical n-inference defect, and does not certify original reference cardinalities. Keep Comment3/RevThree marking where applicable.

## Optional Comment 5 proof clarification

The exact current conditional-proof passage is unavailable, so the following is proposed new text, not a claimed quotation of it. Retain its actual RevFive wrapper.

```latex
% Optional Comment5 explanatory paragraph. Preserve the actual RevFive wrapper.
For a fixed reference and a full, unweighted one-to-one assignment,
let \(Q_{\bm a}(\tau)=\sum_{i=1}^n
\|\bm c_{\tau(i)}+\bm a-\bm r_i\|^2\). Then
\[
Q_{\bm a}(\tau)=Q_{\bm0}(\tau)
+2\bm a^\top\left(\sum_{\ell\in s}\bm c_\ell-
\sum_{i=1}^n\bm r_i\right)+n\|\bm a\|^2.
\]
The added term is independent of \(\tau\), so the minimizing permutation
set is unchanged. A consistent tie rule is needed to preserve the selected
permutation. This expansion applies to squared distances and a full
bijection; it does not apply in general to ordinary Euclidean matching or
to rectangular matching that may select a different sample subset.
With ordinary distances, fixed reference points alone do not guarantee
unchanged optimal matching. In either case, the conditional DI result also
requires an unchanged partition and a fixed bandwidth evaluated at the
corresponding transformed units.
```

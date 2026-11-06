# Manuscript evidence and approval-only replacements

The last designated current snapshot is `DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex`, byte-identical to the Downloads file of the same name. SHA256: `d91765e63c588fa646be268ca62d8fed8b69fd5bec99995f8bfb6a1c2c8a5baa`. This is the quotation authority available locally, not a verified latest revision-colored source.

No `RevThree` or `RevFive` text was found in readable loose `.tex`, `.md`, or `.txt` files under `/home/bardia` (apart from the new audit PLAN), or in `.tex` members of Downloads/latex-projects ZIPs, including nested ZIPs. The root agent reports no connected document-control sessions. The local `latex-projects/GFS-Spread-Sample-JSSAM/1- Measure.tex` is older (mtime 21 May, SHA256 `4246a88a4f90a5c280dbad33b7760df07e60c38489e687d530b0848a65a3ec21`) and must not be called current. `source_candidates.json` records the loose candidate hashes. ZIP metadata and mtimes alone do not establish latest authorial authority.

The specific current Comment 5 conditional proof cannot be quoted because its revision-colored source is absent. Do not invent it. The body fragments below are deliberately unwrapped; when integrating against the actual current source, retain its existing Comment 3 `RevThree` and Comment 5 `RevFive` groups, definitions, and closing braces. No original manuscript or response file was changed.

Two mutually exclusive choices are supplied. **A** changes the Section 3 and Algorithm 2 medoid and matching exponents to describe ordinary Euclidean objectives and replaces the cube kernel with the historical spherical kernel. The historical weighted k-means path still requires explicit disclosure relative to the modern OT definition; exponent/kernel edits alone do not reconcile that difference. **B** retains the modern squared/box definitions and explicitly labels retained historical figures as results from a different pipeline. It supplies no claim to have rerun or validated the modern pipeline.

The Section 4 replacement avoids automatically mapping all recovered block totals to published bars. The audit partition contains 138 Monte Carlo rows: 104 rows whose means and standard deviations link exactly to retained outputs with 2000 inferred DI replications, two rows whose means differ from those retained outputs, and 32 rows without logged replication counts. The 24 NMS/GMS rows are a separate weighted finite-design enumeration. Exact linkage of the 104 Monte Carlo rows does not mean exact enumeration. The main report refines the distinction between 5000 auxiliary-index replications and 2000 DI replications; neither a blanket 5000 nor blanket 2000 statement is warranted.

The prior experiment provides a separate conditional explanation, not existing manuscript Comment 5 text. Exact quotation from `DI_translation_equal_unequal_JSSAM_20261016/REPORT.md:42`:

> For a fixed reference and matching, shifting every selected point by delta adds delta to every internally translated population point. A distance-based KDE evaluated at the corresponding translated positions is unchanged; hence Dnet, eta and DI are unchanged. Squared matching has the same minimizing permutation set under this sample shift, because the change in its total cost is independent of the permutation. These conditions are satisfied by the observed reconstructed references here. Refitting IPnM does not in general guarantee those conditions for every population or initialization.

Historical implementation anchors: `DI_historical_implementation_audit_20261016/evidence/e71d6935b_density_disparity.py:42–48` (spherical Scott KDE), `:72–78` (weighted clustering path), `:93–95` (ordinary assignment), `:155–171` (ordinary weighted medoid and approximate tie handling). Section 2's squared clustering objective is a separate object and should not be changed merely because the medoid and matching objectives differ. The historical class applies KDE to upstream population coordinates; Population normalization must not be mistaken for covariance scaling in KDE.

An accessible author-response draft has related translation wording, but it is not established as the actual current Comment 5 source. Exact quotation from `/home/bardia/Downloads/JSSAM_measure_responses.tex:319–328`:

```latex
Thank you for this comment. We have clarified that the observed sample supplies
the initial centers for IP\(n\)M, whereas the resulting partition and medoids
are constructed using the coordinates and IPs of the entire population. This
sample-specific initialization is intentional: it produces a reference
configuration adapted to the observed sample rather than imposing one fixed
global \(n\)-means solution on every sample. A new figure illustrates how
different sample arrangements from the same population induce corresponding
sample-adaptive IP-balanced partitions and medoid reference configurations.
The statement about uniform translation is needed to clarify that DI assesses
relative configuration rather than absolute geographic position.
```

## section3_weighted_medoid

Decision: A: describe traced historical implementation. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:501–509`.

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

Proposed replacement, approval only:

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

Decision: A: describe traced historical implementation. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:545–556`.

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

Proposed replacement, approval only:

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
where \(\tau\) is a one-to-one mapping from clusters to sampled units.
```

Only exponent removal is substantive; retain any actual current revision-color wrapper.

## section3_historical_kernel

Decision: A: describe traced historical implementation. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:586–597`.

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

Proposed replacement, approval only:

```latex
The historical implementation used an isotropic scalar bandwidth
\(h=N^{-1/(d+4)}\), the Scott bandwidth returned by the fitted
\texttt{KernelDensity} estimator, with \(\mathbf H=h^2\mathbf I_d\).
The multivariate uniform kernel is uniform on the Euclidean unit ball:
\[
K(\bm u)=\frac{1}{v_d}\mathbf 1\{\|\bm u\|_2\leq1\},
\qquad
v_d=\frac{\pi^{d/2}}{\Gamma(1+d/2)},
\qquad \bm u\in\mathbb R^d.
\]
Applying the same estimator to the translated coordinates
```

Boundary convention should follow the fitted backend if exact boundary distances occur; formula states the standard ball-kernel convention. Existing H matrix KDE definition can remain once H=h^2I is specified. Source is KernelDensity(kernel="tophat",bandwidth="scott") on the coordinates supplied to the class; upstream Population normalization is separate. No covariance scaling is applied by this KDE class.

## section4_replications

Decision: A: describe traced historical implementation. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:977–978`.

Exact existing LaTeX:

```latex
\citep{sampling_pkg}. For each reported population, IP setting, sample size,
and competing design, \(5{,}000\) Monte Carlo samples were generated.
```

Proposed replacement, approval only:

```latex
\citep{sampling_pkg}. The retained records distinguish Monte Carlo evaluation
from exact enumeration. Where the Monte Carlo replication count can be
recovered, the retained records report \(2{,}000\) samples. Counts remain
unresolved for other blocks, so a common \(5{,}000\)-replication count is not
established for all reported comparisons.
```

Counts are cross-block historical-audit totals, not automatically one-to-one with every published bar. Before manuscript insertion, main audit must bind these totals to the reported figure groups. 24 NMS/GMS blocks belong to a separate comparison and are excluded. Prefer per-figure counts if their mappings are resolved.

## algorithm2_medoid

Decision: A: describe traced historical implementation. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1490–1505`.

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

Proposed replacement, approval only:

```latex
\State For each cluster \(U_i\), define the reference point \(\bm q_i\) by
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
For the medoid option, ties are broken by largest IP and then smallest label.
```

Changes only medoid exponent; centroid formula remains weighted mean.

## algorithm2_assignment

Decision: A: describe traced historical implementation. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1508–1518`.

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

Proposed replacement, approval only:

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

Decision: Both A and B. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:901–917`.

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

Proposed replacement, approval only:

```latex
    \item[(iii)] Conditional on a fixed partition, fixed reference points,
and a fixed matching, DI is invariant to a common translation of the matched
sample coordinates, provided the density estimator is translation equivariant.
Indeed, replacing each matched sample coordinate by
\(\bm c_{\tau(i)}+\bm a\) replaces every translated population coordinate
by \(\widetilde{\bm c}_\ell+\bm a\), leaving its density at the corresponding
translated location unchanged. In particular, if
\(\bm c_{\tau(i)}=\bm r_i+\bm a\) for all \(i\), then the transformed
population is a common translation of the original population and DI is zero.
This conditional property does not imply invariance when the sample alone is
shifted and its sample-adaptive partition, medoids, or matching are recomputed.
With ordinary Euclidean assignment, optimal rematching need not preserve the
original matching. Different spatial spread indices formalize spatial balance
from different viewpoints; DI evaluates departure from the sample-induced
IP\(n\)M reference configuration illustrated in
Figure~\ref{fig:sample_adaptive_reference}. Its rotation properties are discussed
in Appendix~\ref{Exm:rotation_4x5}.
```

Place inside existing RevFive braces when an authoritative colored source is recovered; do not replace or nest a macro speculatively. Finite-population admissibility of shifted sample coordinates is a separate requirement. Squared matching with a fixed reference has stronger translation behavior of its optimal permutation set, but sample-adaptive reclustering remains outside this theorem.

## section3_keep_squared_alternative

Decision: B: retain squared definition; explicitly separate historical pipeline. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:545–556`.

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

Proposed replacement, approval only:

```latex
The formal definition below uses squared Euclidean assignment and the
squared-distance representative specified above. The retained historical
results were produced by a distinct pipeline using ordinary Euclidean
assignment, weighted ordinary-distance medoids, and an isotropic spherical
\texttt{tophat} kernel. Those results therefore describe the historical
pipeline and are not numerical validation of the present squared-distance,
axis-aligned-kernel definition.

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
where \(\tau\) is a one-to-one mapping from clusters to sampled units.
```

Mutually exclusive with A. Retain squared equations in Section3 and Algorithm2. If retaining cube kernel too, label it as formal definition and retain explicit historical spherical-kernel distinction. This is not a claim to have rerun or validated a future squared pipeline.

## historical_clustering_disclosure

Decision: Both A and B; essential historical-method disclosure. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:488–489`.

Exact existing LaTeX:

```latex
\(\{\bm c_\ell:\ell\in s\}\) as initial centers, we apply the hard IP\(n\)M
procedure to obtain \(n\) spatially compact clusters
```

Proposed replacement, approval only:

```latex
\(\{\bm c_\ell:\ell\in s\}\) as initial centers, the formal definition applies
the hard IP\(n\)M procedure to obtain \(n\) spatially compact clusters.
The retained historical results used the then-current weighted \(k\)-means
initialization and refinement path rather than the entropic OT path described
in Section~2; results from that historical implementation are distinguished
from the formal OT definition.
```

Do not silently replace Section2/Algorithm1 with historical weighted k-means. Historical representative/matching/kernel corrections alone do not align the historical pipeline with the formal modern OT definition. Original e71d6935b DensityDisparity passes init_clust_method="weighted" at lines72-78; older d047278fc class used expanded and centroids. Scope specific figure/code path before universal attribution.

## abstract_translation

Decision: Both A and B. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:143–143`.

Exact existing LaTeX:

```latex
bounded and translation-invariant measure that distinguishes under-dispersion,
```

Proposed replacement, approval only:

```latex
bounded measure with a conditional translation property that distinguishes under-dispersion,
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## introduction_translation

Decision: Both A and B. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:233–235`.

Exact existing LaTeX:

```latex
interpreted as approximately well spread. The index is translation invariant, so shifting a sample without changing its internal geometry does not change its value. Thus, DI focuses on the relative spatial arrangement of the sample and
separates under-dispersion, optimal spread, and over-dispersion on a
common interpretable scale.
```

Proposed replacement, approval only:

```latex
interpreted as approximately well spread. With the reference partition and matching held fixed, a common translation of the matched sample coordinates leaves DI unchanged. This conditional property concerns the relative arrangement of the matched sample and does not guarantee invariance after sample-adaptive reclustering or rematching.
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## conclusion_translation

Decision: Both A and B. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1120–1121`.

Exact existing LaTeX:

```latex
The proposed index is bounded, translation invariant, and directly
interpretable. Negative values indicate under-dispersion or local clustering,
```

Proposed replacement, approval only:

```latex
The proposed index is bounded and directly interpretable, and has the conditional translation property described in Section~\ref{Sec:Measure}. Negative values indicate under-dispersion or local clustering,
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

## appendix_translation_example

Decision: Both A and B. Source: `/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex:1150–1151`.

Exact existing LaTeX:

```latex
In panel~(b), a uniform translation is applied to all sample units and DI
remains unchanged, confirming translation invariance.
```

Proposed replacement, approval only:

```latex
In panel~(b), a uniform translation is applied to all sample units and DI
remains unchanged for this example. This observation alone does not establish invariance after general sample-adaptive reclustering or rematching.
```

Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.

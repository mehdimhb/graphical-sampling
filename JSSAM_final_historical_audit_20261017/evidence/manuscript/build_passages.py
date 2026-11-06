from pathlib import Path
import json, hashlib, datetime, zipfile, io
out=Path(__file__).parent
source=Path('/home/bardia/projects/graphical-sampling/DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex')
lines=source.read_text().splitlines(keepends=True)
items=[]
def entry(key,start,end,replacement,decision='A: describe traced historical implementation',notes=''):
    items.append(dict(id=key,source=str(source),start_line=start,end_line=end,exact_existing_latex=''.join(lines[start-1:end]),proposed_replacement_latex=replacement,decision=decision,notes=notes))
entry('section3_weighted_medoid',501,509,r'''\[
m_i
\in
\arg\min_{u\in U_i}
\sum_{\ell\in U_i}\pi_\ell
\|\bm c_u-\bm c_\ell\|,
\qquad
\bm r_i=\bm c_{m_i}.
\]
''',notes='Only remove exponent 2. The clustering objective in Section 2 remains squared Euclidean; do not change it. Preserve the enclosing Comment 3 / RevThree group if a current colored source is recovered.')
entry('section3_assignment',545,556,r'''The sample units are then matched to the medoid reference points by solving the
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
''',notes='Only exponent removal is substantive; retain any actual current revision-color wrapper.')
entry('section3_historical_kernel',586,597,r'''The historical implementation used an isotropic scalar bandwidth
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
''',notes='Boundary convention should follow the fitted backend if exact boundary distances occur; formula states the standard ball-kernel convention. Existing H matrix KDE definition can remain once H=h^2I is specified. Source is KernelDensity(kernel="tophat",bandwidth="scott") on the coordinates supplied to the class; upstream Population normalization is separate. No covariance scaling is applied by this KDE class.')
entry('section4_replications',977,978,r'''\citep{sampling_pkg}. The retained records distinguish Monte Carlo evaluation
from exact enumeration. Where the Monte Carlo replication count can be
recovered, the retained records report \(2{,}000\) samples. Counts remain
unresolved for other blocks, so a common \(5{,}000\)-replication count is not
established for all reported comparisons.
''',notes='Counts are cross-block historical-audit totals, not automatically one-to-one with every published bar. Before manuscript insertion, main audit must bind these totals to the reported figure groups. 24 NMS/GMS blocks belong to a separate comparison and are excluded. Prefer per-figure counts if their mappings are resolved.')
entry('algorithm2_medoid',1490,1505,r'''\State For each cluster \(U_i\), define the reference point \(\bm q_i\) by
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
''',notes='Changes only medoid exponent; centroid formula remains weighted mean.')
entry('algorithm2_assignment',1508,1518,r'''\State Match the reference points to the sampled units by solving
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
''')
entry('comment5_translation_qualification',901,917,r'''    \item[(iii)] Conditional on a fixed partition, fixed reference points,
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
''','Both A and B',notes='Place inside existing RevFive braces when an authoritative colored source is recovered; do not replace or nest a macro speculatively. Finite-population admissibility of shifted sample coordinates is a separate requirement. Squared matching with a fixed reference has stronger translation behavior of its optimal permutation set, but sample-adaptive reclustering remains outside this theorem.')
entry('section3_keep_squared_alternative',545,556,r'''The formal definition below uses squared Euclidean assignment and the
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
''','B: retain squared definition; explicitly separate historical pipeline',notes='Mutually exclusive with A. Retain squared equations in Section3 and Algorithm2. If retaining cube kernel too, label it as formal definition and retain explicit historical spherical-kernel distinction. This is not a claim to have rerun or validated a future squared pipeline.')
entry('historical_clustering_disclosure',488,489,r'''\(\{\bm c_\ell:\ell\in s\}\) as initial centers, the formal definition applies
the hard IP\(n\)M procedure to obtain \(n\) spatially compact clusters.
The retained historical results used the then-current weighted \(k\)-means
initialization and refinement path rather than the entropic OT path described
in Section~2; results from that historical implementation are distinguished
from the formal OT definition.
''','Both A and B; essential historical-method disclosure',notes='Do not silently replace Section2/Algorithm1 with historical weighted k-means. Historical representative/matching/kernel corrections alone do not align the historical pipeline with the formal modern OT definition. Original e71d6935b DensityDisparity passes init_clust_method="weighted" at lines72-78; older d047278fc class used expanded and centroids. Scope specific figure/code path before universal attribution.')
# Blanket claims elsewhere need qualification if Comment5 is corrected.
for key,start,end,repl in [
 ('abstract_translation',143,143,'bounded measure with a conditional translation property that distinguishes under-dispersion,\n'),
 ('introduction_translation',233,235,'interpreted as approximately well spread. With the reference partition and matching held fixed, a common translation of the matched sample coordinates leaves DI unchanged. This conditional property concerns the relative arrangement of the matched sample and does not guarantee invariance after sample-adaptive reclustering or rematching.\n'),
 ('conclusion_translation',1120,1121,'The proposed index is bounded and directly interpretable, and has the conditional translation property described in Section~\\ref{Sec:Measure}. Negative values indicate under-dispersion or local clustering,\n'),
 ('appendix_translation_example',1150,1151,'In panel~(b), a uniform translation is applied to all sample units and DI\nremains unchanged for this example. This observation alone does not establish invariance after general sample-adaptive reclustering or rematching.\n')]:
 entry(key,start,end,repl,'Both A and B',notes='Follow-through needed to avoid retaining stronger blanket claims elsewhere; preserve existing revision groups.')
meta=dict(status='APPROVAL-ONLY FRAGMENTS; originals unchanged',source_status='Last designated current snapshot, NOT verified latest revision-colored source',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_lines=len(lines),download_duplicate='/home/bardia/Downloads/Measuring_Spatial_Spread_current_2026-10-12.tex',revision_wrapper_status='No RevThree or RevFive found in any readable loose .tex/.md/.txt under /home/bardia, or .tex members of Downloads/latex-projects ZIPs including nested ZIPs. Cannot quote absent macros or certify wrapper preservation against unseen current source.',passages=items)
(out/'passages.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
tex=['% APPROVAL ONLY. Do not input this file as a combined manuscript patch.\n% A and B are mutually exclusive decisions; only selected fragments may be applied.\n% No authoritative current RevThree/RevFive source was located.\n% Preserve actual current enclosing revision macros; never strip or invent them.\n']
for p in items:
 tex += [f'\n% {p["id"]} | {p["decision"]}\n% Existing source: {p["source"]}:{p["start_line"]}-{p["end_line"]}\n',p['proposed_replacement_latex']]
(out/'candidate_corrections.tex').write_text(''.join(tex))
# Record loose candidates and read-only ZIP search inventory for provenance.
ps=list(Path('/home/bardia/Downloads').glob('*Measure*.tex'))+list(Path('/home/bardia/Downloads/LaTeX').glob('*Measure*.tex'))+[Path('/home/bardia/latex-projects/GFS-Spread-Sample-JSSAM/1- Measure.tex'),source,Path('/home/bardia/projects/graphical-sampling/DI_historical_implementation_audit_20261016/evidence/submission_measure.tex')]
inventory=[]
for p in ps:
 s=p.read_text();inventory.append(dict(path=str(p),mtime=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),lines=len(s.splitlines()),RevThree=s.count('RevThree'),RevFive=s.count('RevFive')))
(out/'source_candidates.json').write_text(json.dumps(inventory,indent=2)+'\n')
for i in items:
 assert i['exact_existing_latex']==''.join(lines[i['start_line']-1:i['end_line']])
print('Wrote',len(items),'exact-quote/proposed-replacement pairs; verified line extraction; source SHA256',meta['source_sha256'])

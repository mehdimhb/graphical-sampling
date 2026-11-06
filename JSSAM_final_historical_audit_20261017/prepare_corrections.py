"""Prepare quoted approval-only fragments. Never edits a manuscript or response."""
from pathlib import Path
import hashlib,json

OUT=Path(__file__).resolve().parent
candidate=json.loads((OUT/'evidence/manuscript/passages.json').read_text())
SOURCE=Path(candidate['source'])
source=SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==candidate['source_sha256']
source_lines=source.splitlines(keepends=True)
passages=candidate['passages']
by_id={p['id']:p for p in passages}
for p in passages:
    assert ''.join(source_lines[p['start_line']-1:p['end_line']])==p['exact_existing_latex'],p['id']

# Keep exact adjacent sentence boundaries in exponent-only replacements.
for key,old,new in [
    ('section3_weighted_medoid',r'\|\bm c_u-\bm c_\ell\|^2',r'\|\bm c_u-\bm c_\ell\|'),
    ('section3_assignment',r'\|\bm c_{\tau(i)}-\bm r_i\|^2',r'\|\bm c_{\tau(i)}-\bm r_i\|'),
    ('algorithm2_assignment',r'\|\bm c_{\tau(i)}-\bm q_i\|^2',r'\|\bm c_{\tau(i)}-\bm q_i\|')]:
    p=by_id[key]
    assert old in p['exact_existing_latex']
    p['proposed_replacement_latex']=p['exact_existing_latex'].replace(old,new)

by_id['section3_historical_kernel']['proposed_replacement_latex']=r'''where, for the traced historical computations,
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
'''
by_id['section4_replications']['proposed_replacement_latex']=r'''\citep{sampling_pkg}. The reported summaries combine simulation stages.
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
'''
by_id['section4_replications']['decision']='Both A and B'
by_id['section4_replications']['notes']='The counts concern the 162-setting DI plotting input: 138 Monte Carlo settings and 24 finite-design settings. The ledger and final report give the population/design coverage; enumeration entries are not Monte Carlo replications.'
by_id['section3_historical_kernel']['notes']='The strict ball boundary follows the recovered backend; exact boundary distances can matter on a grid. The existing matrix KDE definition can remain with H=h^2I specified. Upstream Population normalization is separate from this scalar-bandwidth KDE.'
by_id['algorithm2_medoid']['proposed_replacement_latex']=by_id['algorithm2_medoid']['exact_existing_latex'].replace(
    r'\pi_\ell\|\bm c_u-\bm c_\ell\|^2',r'\pi_\ell\|\bm c_u-\bm c_\ell\|').replace(
    'For the medoid option, ties are broken by largest IP and then smallest label.',
    'In the traced historical medoid-update implementation, the first minimizing unit in cluster order was selected.')
by_id['algorithm2_medoid']['notes']='Changes the medoid exponent and historical tie convention; the centroid formula remains the weighted mean.'
by_id['section3_keep_squared_alternative']['proposed_replacement_latex']=r'''The formal definition below retains squared-distance representatives,
squared Euclidean matching, and the box kernel specified here. The traced
historical Monte Carlo computations instead used weighted preliminary
\(k\)-means followed by probability refinement, ordinary-distance medoids,
ordinary Euclidean matching, and a spherical \texttt{tophat} KDE.
The retained results must therefore be identified as results of that
historical variant, rather than numerical validation of the present
OT/squared-distance/box-kernel specification. Runtime provenance remains
unresolved for the other blocks identified in the reproducibility audit.

''' + by_id['section3_assignment']['exact_existing_latex']
by_id['historical_clustering_disclosure']['proposed_replacement_latex']=r'''\(\{\bm c_\ell:\ell\in s\}\) as initial centers, the formal definition applies
the hard IP\(n\)M procedure to obtain \(n\) spatially compact clusters.
The traced historical Monte Carlo implementation used weighted preliminary
\(k\)-means followed by open-TSP ordering and inclusion-probability refinement,
rather than the entropic OT preliminary stage described in Section~2.
Its retained results are distinguished from computations of the formal OT
variant; exact runtime attribution is unresolved for the other audited blocks.
'''
by_id['comment5_translation_qualification']['proposed_replacement_latex']=r'''    \item[(iii)] Conditional on a fixed hard partition, fixed reference
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
'''

def add(key,start,end,replacement,decision,notes):
    passages.append(dict(id=key,source=str(SOURCE),start_line=start,end_line=end,
        exact_existing_latex=''.join(source_lines[start-1:end]),
        proposed_replacement_latex=replacement,decision=decision,notes=notes))

add('section3_historical_medoid_ties',510,512,
    'In the traced historical medoid-update implementation, the first minimizing\nunit in cluster order was selected.\n',
    'A: describe the historical notebook tie convention',
    'Do not claim the historical notebook used today\'s np.isclose/largest-IP tie rule. If preserving the modern definition, retain its formal rule and disclose this historical difference instead. Keep any actual RevThree wrapper.')
add('algorithm2_historical_preliminary_stage',1485,1487,
    r'''\State For the historical DI variant, initialize weighted \(k\)-means
at the observed sample units and apply the historical open-TSP ordering and
inclusion-probability refinement to construct the reference partition.
\label{line:upbkm}
''',
    'A only: describe the historical pipeline; do not mix with the formal OT version',
    'Do not change the squared clustering loss in Section2. This fragment distinguishes algorithms, does not repair the historical n-inference defect, and does not certify original reference cardinalities. Keep Comment3/RevThree marking where applicable.')

conditional_proof=r'''% Optional Comment5 explanatory paragraph. Preserve the actual RevFive wrapper.
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
'''

header='''# Proposed LaTeX corrections — approval only

No manuscript or response has been changed. Exact quotations below are checked against the last designated 12 October source snapshot, not a verified latest colored revision. The actual current `RevThree`/`RevFive` source and exact current Comment 5 proof were not recovered. These bodies are deliberately unwrapped: preserve the real enclosing revision groups, definitions and closing braces when integrating. Do not invent or replace color definitions.

**Decision required:** A describes the traced historical weighted/Euclidean/spherical DI variant; B retains the modern OT/squared/box definition and explicitly separates the old numerical results. They are alternatives, not one combined patch. A does not repair historical reference-count defects or certify the untraced rows. Conditional translation wording and replication disclosure are needed under either choice. The squared clustering objective in Section 2 is separate from the medoid and matching objectives and must not be changed merely by deleting their exponents.

The supplied `.tex` is a collection of numbered review fragments, not an executable patch or a file to `\\input` wholesale. Complete provenance is in REPORT.md and the row-level ledger.
'''
md=[header]
tex=['% APPROVAL ONLY — NOT A COMBINED PATCH. Alternatives A and B are mutually exclusive.',
     '% Actual latest RevThree/RevFive wrappers were not recovered. Keep existing wrappers unchanged.',
     '% Fragments are deliberately unwrapped. Do not input this whole file into a manuscript.','']
for p in passages:
    md += [f"## {p['id']}",'',f"**{p['decision']}**. Source: `{p['source']}:{p['start_line']}–{p['end_line']}`.",'',
           'Exact existing LaTeX:','','```latex',p['exact_existing_latex'].rstrip(),'```','',
           'Proposed replacement:','','```latex',p['proposed_replacement_latex'].rstrip(),'```','',p.get('notes',''),'']
    tex += [f"% Fragment: {p['id']} — {p['decision']}",
            f"% Source lines {p['start_line']}–{p['end_line']} in the designated 12 October snapshot.",
            p['proposed_replacement_latex'].rstrip(),'']
md += ['## Optional Comment 5 proof clarification','',
       'The exact current conditional-proof passage is unavailable, so the following is proposed new text, not a claimed quotation of it. Retain its actual RevFive wrapper.','','```latex',conditional_proof.rstrip(),'```','']
tex += [conditional_proof]
(OUT/'PROPOSED_CORRECTIONS.md').write_text('\n'.join(md))
(OUT/'proposed_corrections.tex').write_text('\n'.join(tex))
payload=dict(source=str(SOURCE),source_sha256=candidate['source_sha256'],
             source_status=candidate['source_status'],passages=passages,
             optional_comment5_proof=conditional_proof,originals_modified=False)
(OUT/'proposed_passages.json').write_text(json.dumps(payload,indent=2)+'\n')
print(f'Prepared {len(passages)} exact-quote approval fragments; no manuscript edits.')

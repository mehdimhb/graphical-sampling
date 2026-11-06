"""Generate the short report and separate uninserted LaTeX proposals."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parent
read=lambda s:pd.read_csv(P/s,float_precision='round_trip')
f=lambda x:format(float(x),'.17g')
x=read('scored_references.csv').set_index('run_id');table=read('comparison_table.csv');pairs=read('reference_pairs.csv');groups=read('run_group_summary.csv')
spec=json.loads((P/'preselection.json').read_text())
large=pairs[(pairs.run_a=='s2_official_00')&(pairs.run_b=='s2_stress_04')].iloc[0]
robust=pairs[(pairs.run_a=='s2_official_02')&(pairs.run_b=='s2_stress_04')].iloc[0]
o0=x.loc['s2_official_00'];o2=x.loc['s2_official_02'];t4=x.loc['s2_stress_04']
md='| Sample/reference | Native OT loss J | Qsoft | Hard IP range | Dnet | eta | DI |\n|---|---:|---:|---:|---:|---:|---:|\n'
tex=[r'\textcolor{green}{%',r'\begin{tabular}{lrrrrrr}',r'\hline',r'Sample/reference & $J$ & $Q_{\rm soft}$ & Hard IP range & $D_{\rm net}$ & $\eta$ & DI \\',r'\hline']
for r in table.itertuples():
    short=r.sample+' / '+('O'+str(int(r.run_id[-2:])) if r.kind=='official' else 'T'+str(int(r.run_id[-2:])))
    md+=f'| {short} | {r.J_OT_native:.8g} | {r.Q_soft:.8g} | {r.hard_IP_min:.8g}–{r.hard_IP_max:.8g} | {r.Dnet:.8g} | {r.eta:.8g} | {r.DI:.8g} |\n'
    tex.append(f'{short} & {r.J_OT_native:.6f} & {r.Q_soft:.6f} & {r.hard_IP_min:.4f}--{r.hard_IP_max:.4f} & {r.Dnet:.6f} & {r.eta:.6f} & {r.DI:.6f} '+r'\\')
tex.extend([r'\hline',r'\end{tabular}%',r'}'])
(P/'comparison_table.tex').write_text('\n'.join(tex)+'\n')
report=f'''# Reviewer 1, Comment 1: reference sensitivity

Alternative references can change DI materially in this focused example under a stated 5% native-loss/compactness criterion. Variation from the official procedure's random TSP choices was smaller. The result is criterion-dependent and does not establish invariance or general instability across populations.

**Authority and fixed design.** The latest locally available Measure manuscript is the 12 October 2026 source saved in `sources/`. We used the existing clustered N=144 population, n=8, and the same unequal inclusion vector throughout. S1 IDs={spec['observed_samples_ids'][0]}; S2 IDs={spec['observed_samples_ids'][1]}. Both were fixed by seed20261013 before scoring. Each received ten official sample-initialized OT runs (TSP seeds0–9) and ten separately labeled alternative-initialization stress runs (centers preselected by seed918273; TSP seed0). All40 converged. The observed examples are fixed subsets, not draws used to estimate an UP-design probability.

We executed the actual package's OT updates, TSP, refinement, hardening, and original DI score function, using a separate manuscript-specific adapter for sample initialization, squared medoids/matching, and the box KDE. The unmodified public DI method differs from the manuscript; `discrepancies.md` documents each difference. Scoring uses one fixed covariance-based Scott matrix H=N^(-1/3)cov(original normalized coordinates,ddof=1), symmetric root, unweighted uniform box, and angular calibration pi/8. Scott's matrix interpretation is stated explicitly, not claimed uniquely specified. No package defaults silently replace manuscript choices.

**Suitability.** The stated criterion is the feasible preliminary entropic OT loss J=sum P*Cnorm+.01 sum P(log P−1), with Cnorm squared distances divided by their current median+1e-12. The manuscript specifies no global final-reference objective or numerical meaning of “nearly.” We therefore report J plus post-refinement Qsoft=sum pi*mu*distance² to the final soft weighted centroids, Qhard, hard IP totals, and medoid cost. Equal preliminary loss alone does not establish equality of the final partitions. For pairs, native relative J gap uses max absolute loss in the denominator, while Qsoft gap uses the smaller Qsoft. We show numerical equality,0.1%,1%,5%; eligibility uses losses, never DI. The full comparison is in `objective_tolerances.csv`, including OT-only and conservative joint criteria. No solution is claimed globally optimal.

O denotes an official TSP seed; T denotes a preselected stress initialization. The displayed table uses eight significant digits; CSV/JSON files retain full float64 precision and all eight hard-cluster totals and medoid IDs.

{md}
**Observed official variation.** The ten official runs produce two distinct references per sample after removing mere label permutations. Their preliminary J values are identical within each sample, but final Qsoft differs by about1.44%. DI ranges are 0.027924624250291685 for S1 and0.0043178696430701269 for S2. These are small relative to the predeclared descriptive material-change threshold0.05, but are not zero or evidence of universal robustness. The two TSP paths have length regret0 and about3.07% relative to an independently enumerated exact open-TSP optimum.

**Stress-test sensitivity.** For S2, O0 versus T4 changes DI from {f(o0.DI)} to {f(t4.DI)}, a difference {f(large.DI_absolute_difference)}. The native J relative gap is {f(large.native_OT_relative_gap)}, and the Qsoft relative gap {f(large.Qsoft_relative_gap)}; T4 has the lower loss on both. More tellingly, comparing T4 against O2, whose TSP path is optimal for its preliminary centroids, still gives DI difference {f(robust.DI_absolute_difference)}, with native J gap4.4524%, Qsoft gap2.6548%, and Qhard gap1.6066%. Both O2 and T4 use optimal TSP paths. The reference difference is substantive: optimal medoid-set RMS displacement {f(robust.medoid_set_RMS)} and IP-weighted hard-partition disagreement {f(robust.partition_IP_disagreement)};114 units change translated density. These are genuinely different references, not label changes. Hard masses are recorded rather than assumed exactly1.

The largest joint-5% DI differences are0.03529366358176067 for S1 and {f(large.DI_absolute_difference)} for S2. Numerical equality,0.1%, and1% joint rules admit no distinct pairs, so they provide no robustness evidence. OT-only equality admits the official pair with the modest differences above. Native-OT loss uses a reference-dependent median. With the official median held common only for loss comparison, T4's loss is20.855519683038062% lower, rather than within5%; requiring common-scale J closeness removes the substantial pair from the5% set. The result therefore depends on what “nearly suitable” means. T4 nevertheless remains a valid lower-loss alternative, not one chosen because of its DI.

All Dnet signs stay positive for S1 and negative for S2. The material S2 contrast has Dnet={f(o2.Dnet)} versus {f(t4.Dnet)}, well away from zero. It is reference sensitivity, not a near-zero sign reversal; the exact aggregation decomposition is retained in `reference_pairs.csv`. This task makes no proposal about other reviewer comments or about changing DI.

**Limited theory and qualification.** Consistent relabeling leaves every translation and hence DI unchanged; all assignments here are unique. For the fixed box bandwidth, if no translated pair crosses its support boundary, densities and DI also remain identical. A sufficient bound is2d<gamma, with d the maximum whitened displacement after removing a common shift and gamma the smallest original translated pair's margin from the box boundary. This is proved in `theory.md`. Neither equal OT losses nor close compactness values establish that condition. We can defend DI as conditional on a fully specified reference-construction convention, but not a claim that equally or nearly suitable alternative references always give the same or similar DI. The proposed wording makes that limitation explicit and restricts the numerical findings to these two samples.

`reference_comparison.pdf` and `.png` show the fixed S2 sample with actual medoids and density contributions. `verification.json` audits all40 runs independently from serialized state, all40320 possible squared assignments per run, source hashes, marginal constraints, exact densities, and replayed actual-package TSP/refinement. The manuscript and Sampling paper were not edited. `manuscript_clarification.tex` and `reviewer_response.tex` are separate proposals for review; the new manuscript wording is green.
'''
# Avoid a manually transcribed maximum: derive it from the tolerance table.
tol=read('objective_tolerances.csv');m1=tol[(tol['sample']=='S1')&(tol.scope=='all')&(tol.criterion=='native_OT_and_Qsoft')&(tol.relative_tolerance==.05)].iloc[0].max_DI_difference
report=report.replace('0.03529366358176067',f(m1))
(P/'report.md').write_text(report)
clarification=r'''% PROPOSAL ONLY. Requires xcolor. Do not overwrite existing manuscript revisions.
% Suggested insertion after the sample-adaptive reference paragraph in Section3,
% before Figure adaptive_ipnm_reference; alternatively replace only its broad
% "equally or nearly suitable" interpretation after author approval.
\textcolor{green}{The reference is conditional on the specified initialization
and numerical construction, rather than a unique globally optimal benchmark.
Consistent relabeling leaves DI unchanged, but equal preliminary OT losses do
not imply identical final partitions or DI after refinement and hardening.
In a focused check with two prespecified samples of size 8 from the clustered
population ($N=144$, unequal IPs), ten sample-initialized runs gave DI ranges
0.0279 and 0.0043. Alternative initialization, considered only as a sensitivity
stress test, yielded a reference with a 4.45\% native OT-loss gap and 2.65\%
refined soft-compactness gap, changing the second sample's DI from $-0.5822$
to $-0.4220$ without changing the sign of $D_{\mathrm{net}}$. Both losses were
lower for the alternative. These percentages use the stated per-run median
normalization; a common-scale OT comparison instead gives a 20.86\% improvement.
Thus the construction must be reported and fixed for reproducibility; we do
not assert invariance to alternative reference choices.}
'''
(P/'manuscript_clarification.tex').write_text(clarification)
response=r'''% PROPOSED response to Reviewer1, Comment1 ONLY. Original manuscript untouched.
\noindent\textbf{Reviewer1, Comment1.}
The sample-adaptive construction does not establish whether equally or nearly
suitable references can materially change DI for the same observed sample.

\medskip
\noindent\textbf{Proposed response.}
We agree that sample-adaptive initialization is not a proof of invariance to
alternative references. Consistent permutations of cluster labels leave all
translations and DI unchanged, but different partitions or medoid geometries
need not do so. Similar objective values alone do not establish a DI bound.

We performed a focused check on the existing clustered population with $N=144$,
$n=8$, unequal IPs, and two fixed observed samples chosen before inspecting
sensitivity. For each sample we used ten runs of the official sample-initialized
procedure and ten separately labeled alternative-initialization stress tests.
Population, samples, IPs, uniform box kernel, covariance-based Scott bandwidth,
and angular scaling were fixed. The full OT, refinement, hardening, squared
medoid and matching pipeline was recomputed; no DI values or densities were
fabricated. A separate audited adapter executes the package routines while
respecting the manuscript-specific choices where the public implementation
differs. The original DI formula was retained.

The manuscript states a preliminary entropic OT objective but no numerical
equivalence tolerance or final-reference minimization criterion. We therefore
report both its native normalized loss and refined soft compactness, actual
hard-cluster IP totals, and comparisons at 0.1\%,1\%, and 5\% loss tolerances.
The official runs produced two distinct references per sample with identical
preliminary losses; the DI ranges were 0.0279 and 0.0043. In the stress test, an
alternative with lower native OT loss and lower refined compactness loss had
gaps 4.45\% and 2.65\%, respectively, but changed the second sample's DI from
$-0.58223930146083458$ to$-0.42201965129016566$. Both compared TSP paths were
optimal for their preliminary centers, and $D_{\mathrm{net}}$ remained negative
and away from zero. This substantial contrast therefore does not arise from a
near-zero sign reversal or merely a poor TSP path.

The tolerance matters: the stricter joint 0.1\% and 1\% rules admitted no distinct
pairs in this set. Moreover, because the manuscript recomputes the cost median,
using a common median for objective comparison makes the alternative's OT loss
20.86\% lower, rather than nearly equal. We do not interpret the example as a
universal statement about arbitrary near-optimal references. It shows why fixing
the construction provides a reproducible convention, not a general robustness
guarantee. The proposed manuscript clarification therefore qualifies DI as
reference-conditional, explicitly distinguishes official randomness from the
initialization stress test, and restricts the numerical conclusion to the
examined population and samples. The compact table and one figure provide the
computed references and density evidence; full settings, seeds, coordinates,
and verification are supplied separately.
'''
(P/'reviewer_response.tex').write_text(response)
print(P/'report.md')

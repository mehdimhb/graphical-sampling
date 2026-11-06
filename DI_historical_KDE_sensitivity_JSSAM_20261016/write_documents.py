"""Generate separate author-review artifacts; never edit the original manuscript."""
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
LABEL={'C16':'Clustered','M10':'Meuse','S100':'Swiss'}
ORDER={'C16':0,'M10':1,'S100':2}


def main():
    verify=json.loads((OUT/'verification.json').read_text());assert verify['all_passed']
    data=pd.read_csv(OUT/'results.csv',float_precision='round_trip')
    baseline=pd.read_csv(OUT/'baselines.csv',float_precision='round_trip').set_index('scenario')
    summary=pd.read_csv(OUT/'summary.csv',float_precision='round_trip')
    summary=summary.sort_values(['scenario','experiment'],key=lambda c:c.map(ORDER) if c.name=='scenario' else c)
    baseline_lines=[];sd_lines=[];table_lines=[]
    for sid in ORDER:
        b=baseline.loc[sid]
        baseline_lines.append(f'| {LABEL[sid]} ({int(b.N)}, {int(b.n)}) | {b.old_box_DI:.9f} | {b.DI:.9f} | {b.h:.9f} |')
    for r in summary.itertuples():
        sd_lines.append(f'| {LABEL[r.scenario]} | {r.experiment} | {r.old_box_DI_sd:.9f} | {r.DI_sd:.9f} | [{r.DI_min:.6f}, {r.DI_max:.6f}] | {r.max_abs_delta_from_baseline:.6f} | {r.sign_differences_from_baseline} |')
        table_lines.append(f'{LABEL[r.scenario]} & {r.experiment} & {r.baseline_DI:.6f} & [{r.DI_min:.6f},{r.DI_max:.6f}] & {r.old_box_DI_sd:.6f} & {r.DI_sd:.6f} & {r.max_abs_delta_from_baseline:.6f} & {r.sign_differences_from_baseline}'+r'\\')
    min_dn=float(abs(data.Dnet).min());max_change=float(abs(data.DI_historical_minus_box).max())
    report=f'''# Historical-KDE rescoring of the reference-sensitivity study

**All 360 requested evaluations were rescored from verified saved reference configurations. No OT, clustering, medoid/matching optimization or TSP search was repeated.** C32 was excluded. Three saved baselines were rescored separately. These results use the historical KDE on the existing manuscript-oriented references; they do not reproduce the entire historical weighted-clustering/Euclidean-medoid pipeline or its Monte Carlo averages.

## Method and reuse

C16 (N=144,n=16), Meuse (N=155,n=10) and Swiss (N=959,n=100) retain their original fixed UP samples, seeds, 40 A/B/C outcomes and numerical hardening. A varies TSP search conditional on a fixed OT coupling and preliminary hard labels; B varies saved OT numerical routes under a deterministic TSP rule; C uses the corresponding B coupling with independent TSP seeds. All translated arrays were already saved and reconstruct bitwise as `coords + (coords[matched]-coords[medoids])[labels]`. State/summary fingerprints and reference hashes are recorded in preflight.json and results.csv; B/C coupling reuse and A's fixed coupling/preliminary labels were verified.

Normalization exactly matches the historical Population function: subtract the original per-coordinate minimum and divide all coordinates by the largest original coordinate span. Saved translated coordinates are **not normalized again**. Unweighted sklearn `KernelDensity(kernel="tophat", bandwidth="scott", rtol=1e-4)` uses Euclidean balls with scalar h=N^(-1/(d+4)); h is numerically identical for original and translated fits. The rtol setting matches the historical source. Default rtol=0 produced bitwise identical density arrays and scores for all 363 states.

Dnet and eta use the existing per-unit clipping, theta0=pi/8 and native-position density evaluations; DI=(1-eta)Dnet+eta sign(Dnet), with no added zero threshold. Neither DI definition nor reference construction was changed.

## Baselines

| Population (N,n) | Previous box DI | Historical-KDE DI | Scalar h |
| --- | ---: | ---: | ---: |
{chr(10).join(baseline_lines)}

## Procedure comparison

SD is the sample standard deviation of the 40 DI values (denominator39); it also equals the SD of differences from a fixed baseline. Max is the maximum absolute deviation from the historical-KDE baseline. Flips count np.sign differences from that baseline, including any zero transitions.

| Population | Procedure | Previous box SD | Historical-KDE SD | Historical-KDE DI range | Max | Flips |
| --- | --- | ---: | ---: | --- | ---: | ---: |
{chr(10).join(sd_lines)}

All 360 signs agree with their own recalculated baseline **and** their corresponding previous box-KDE evaluation: zero strict reversals, zero sign mismatches and zero zero-transitions. The retained 360 box-KDE evaluations already had zero baseline flips; absence of the old C32 reversal example results from excluding that scenario, not from an evaluated historical-KDE correction to C32. C32 was not rescored.

## Interpretation and numerical behavior

Observed spread decreases in every group with nonzero previous SD; Meuse B remains invariant over its saved outcomes. Baseline magnitudes nevertheless change substantially, especially Swiss; the largest paired change across the retained evaluations is {max_change:.9f}. This is a change of the **full KDE specification** (kernel geometry and bandwidth scaling), not an isolated kernel-shape experiment. It is not justified to transfer old effect sizes or describe the discrepancy as numerically immaterial.

Reference invariance is still contradicted conditionally. In C16, the baseline and A02 share exactly the saved OT coupling and preliminary labels, but differ in two aligned final hard assignments and one medoid. Their historical-KDE DIs are0.276619180 and0.283760614 (absolute difference0.007141434). Their open-TSP gaps are0% and1.891180% relative to the saved exact benchmark for their common centroid instance. Both have native/common OT gaps of1.041835%/0.617120% relative to best-observed saved objectives. They therefore qualify at the **2% screen, not1%**; OT optimality is not certified. This is a modest quantified dependence, not a demonstrated sign reversal or a universal stability bound. The unchanged quality screens and all7,380 comparisons are supplied separately.

Swiss variation retains the saved numerical hardening ambiguity around coincident/near-coincident centroids. KDE rescoring does not resolve it or establish that all saved numerical outcomes represent distinct mathematical OT optima. A comparisons remain conditional on the saved preliminary hardening. The new values do not inherit the numerical sizes of the earlier box-KDE tight-control effects.

No original or translated density is zero, and no nonfinite score occurs. Every density includes its own unit. The minimum observed absolute Dnet is {min_dn:.9f}. No evaluated Dnet is zero, and no within-scenario sign change is observed. Radial top-hat counts remain discontinuous at the radius boundary; no pair lies exactly on it (minimum distance-to-radius margin {verify['minimum_radius_boundary_margin']:.3g} in normalized units). Direct counts confirm all density arrays. These checks do not establish continuity under arbitrary perturbations.

These are descriptive conditional search summaries, not sampling standard errors or confidence intervals. Repeated references and dependent B/C trajectories are retained. Three fixed samples cannot establish population frequency, worst-case robustness, statistical consistency or stability of the manuscript's Monte Carlo averages.

## Verification and author action

All360 primary evaluations and3 baselines passed direct radial-count and original box-count checks, extended-precision DI checks, and independent individual-row summary verification. The largest score discrepancy was {verify['maximum_longdouble_score_error']:.3g}; no signs differed. Original source/study/package/manuscript hashes were unchanged for all{verify['protected_files_verified']:,} protected files. Independent review additionally recalculated optimization gaps, eligible-set counts and all pairwise flags from source summaries.

**Revise the Comment1 response and Appendix.** Replace the480/four-setting design, excluded C32 examples and box-specific effects with the360/three-setting design and the more limited conclusions above. The qualitative need to acknowledge reference dependence remains; the old stronger empirical example cannot be carried over. Proposed replacements are separate files awaiting author approval. No original manuscript, response, package, figure, dataset or saved study file was edited.

Files: results.csv (360 individual paired evaluations), baselines.csv (three controls), summary.csv, appendix_table.tex, appendix_discussion_proposed.tex, reviewer_comment1_recommendation.md, near_best_summary.csv, pairwise.csv, config.json, preflight.json, verification.json/csv and per-evaluation densities/summary records. HANDOFF.md contains inspection/resume commands.
'''
    replacements={'denominator39':'denominator 39','are0.':'are 0.','and0.':'and 0.','difference0.':'difference 0.','are0%':'are 0%', 'and1.':'and 1.', 'of1.':'of 1.', 'not1%':'not 1%', 'all7,380':'all 7,380','All360':'All 360','and3':'and 3','all5,':'all 5,','Comment1':'Comment 1','the480':'the 480','the360':'the 360'}
    for a,b in replacements.items():report=report.replace(a,b)
    (OUT/'REPORT.md').write_text(report)
    latex=r'''\begin{table}[htbp]
\centering\small
\caption{\textcolor{red}{Fixed-sample reference sensitivity after historical-KDE rescoring: Clustered $(N,n)=(144,16)$, Meuse $(155,10)$, and Swiss $(959,100)$. Each row contains40 saved evaluations;360 in total, with three separate baseline controls. $DI_0$ uses spherical top-hat KDE with scalar Scott bandwidth. SD uses denominator39. Max denotes $\max|DI-DI_0|$; flips compare signs with the corresponding baseline. Box SD refers to the previous covariance-scaled box KDE on exactly the same references. These are descriptive search summaries, not sampling uncertainty.}}
\label{tab:reference_sensitivity_historical_kde}
{\color{red}\resizebox{\linewidth}{!}{\begin{tabular}{llrllrrr}
\hline
Population & Proc. & $DI_0$ & DI range & SD(box) & SD(radial) & Max & Flips\\
\hline
'''+ '\n'.join(table_lines)+r'''
\hline
\end{tabular}}}
\end{table}
'''
    latex=latex.replace('contains40','contains 40').replace(';360','; 360').replace('denominator39','denominator 39')
    (OUT/'appendix_table.tex').write_text(latex)
    discussion=r'''{\color{red}
\paragraph{Sensitivity to the reference configuration.}
We fixed one UP sample in each of three settings: Clustered $(N,n)=(144,16)$,
Meuse $(155,10)$, and Swiss $(959,100)$. Procedure A varied TSP search while
fixing the OT coupling and preliminary labels; B varied numerical OT routes
under a deterministic TSP rule; C reused the corresponding B coupling with
independent TSP seeds. We retained40 outcomes per procedure and setting.
All360 saved reference configurations and translations were reused, without
repeating reference optimization. We recalculated only the densities and DI,
using a spherical uniform kernel and scalar Scott bandwidth
$h=N^{-1/(d+4)}$ in the original normalized coordinate system. The same
bandwidth was used after translation, without renormalizing the translated
coordinates. This is a KDE-only comparison on the saved references.

Table~\ref{tab:reference_sensitivity_historical_kde} summarizes the results.
The baseline DIs were0.276619, $-0.275822$, and $-0.085989$, respectively.
All signs agreed with the corresponding baseline. Reference invariance was
nevertheless not observed: in the Clustered setting, two outcomes sharing
exactly the OT coupling and preliminary labels gave DI0.276619 and0.283761,
an absolute difference of0.007141. Their open-TSP lengths were within1.892\%
of the saved exact benchmark for that common centroid instance. Their native
and common-scale OT gaps relative to the best observed saved values were
1.042\% and0.617\%, respectively; hence this comparison satisfies the2\%
screens, but not the1\% screen. Swiss variation includes ambiguity in numerical
hardening near coincident centroids and should not all be interpreted as
variation between distinct mathematical OT optima.

The historical-KDE rescoring reduced the observed spread and changed DI
magnitudes substantially relative to the previous box-KDE calculation.
These conditional, dependent search outcomes quantify a limitation of the
reference construction; they do not establish general reference invariance,
population-level frequencies, sampling uncertainty, or agreement with the
manuscript's Monte Carlo averages.
}
'''
    for a,b in [('retained40','retained 40'),('All360','All 360'),('were0.','were 0.'),('DI0.','DI 0.'),('and0.','and 0.'),('of0.','of 0.'),('within1.','within 1.'),('the2','the 2'),('the1','the 1')]:discussion=discussion.replace(a,b)
    (OUT/'appendix_discussion_proposed.tex').write_text(discussion)
    (OUT/'reviewer_comment1_recommendation.md').write_text('''# Recommendation for Reviewer1, Comment1

Revision is necessary. The original response describes four settings/480 evaluations and relies on an excluded C32 example. The revised Appendix contains three settings/360 evaluations, and historical-KDE magnitudes and variability differ substantially.

Use the recalculated table and distinguish numerical reference dependence from observed sign stability. All360 retained evaluations preserve both their baseline sign and their old box-KDE sign; the retained box outcomes also had no baseline reversals. Do not describe absence of reversals as a demonstrated consequence of changing KDE in C32: that scenario was not rescored.

A replacement focused example is C16 baseline/A02: identical OT coupling and preliminary labels, DI0.276619180 versus0.283760614, difference0.007141434. Both pass the2% native/common-OT and TSP screens, not1%; the open-TSP benchmark is saved and exact for their common centroid instance, while OT gaps are best-observed diagnostics. Describe this as modest conditional dependence, without declaring it negligible or materially large using an unstated threshold.

Retain qualifications about Swiss numerical hardening and dependent search trajectories. Replace box-specific kernel language with the historical radial/scalar specification. The theoretical non-invariance warning remains appropriate, but these three fixed samples provide no empirical sign-reversal example or general stability theorem. Historical-KDE rescoring does not reproduce the entire historical reference pipeline or validate manuscript Monte Carlo averages.

Separate proposed Appendix text is supplied for author review. No manuscript or original reviewer-response file has been edited.
'''.replace('Reviewer1','Reviewer 1').replace('Comment1','Comment 1').replace('All360','All 360').replace('DI0.','DI 0.').replace('versus0.','versus 0.').replace('difference0.','difference 0.').replace('the2','the 2').replace('not1','not 1'))
    preview=r'''\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{xcolor,graphicx,amsmath,amssymb}
\begin{document}
\input{appendix_discussion_proposed.tex}
\input{appendix_table.tex}
\end{document}
'''
    (OUT/'appendix_preview.tex').write_text(preview)
    print('Separate comparison report, table, proposed discussion and recommendation generated.')


if __name__=='__main__':main()

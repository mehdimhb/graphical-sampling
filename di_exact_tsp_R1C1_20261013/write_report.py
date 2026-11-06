from pathlib import Path
import json,re
import pandas as pd
P=Path(__file__).resolve().parent
read=lambda n:pd.read_csv(P/n,float_precision='round_trip')
fmt=lambda v:format(float(v),'.17g')
s=json.loads((P/'analysis_summary.json').read_text());design=json.loads((P/'frozen_design.json').read_text());table=read('02_Comparison.csv');orient=read('optimal_orientation_comparisons.csv')
md='| Sample | Ordering | Runs | Path length | Gap (%) | Dnet | eta | DI |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for r in table.itertuples():md+=f'| {r.sample} | {r.method} | {r.runs} | {r.TSP_length:.10g} | {r.TSP_gap_percent:.8g} | {r.Dnet:.10g} | {r.eta:.10g} | {r.DI:.10g} |\n'
e1=table[(table['sample']=='S1')&table.method.eq('Exact: deterministic path')].iloc[0]
long1=table[(table['sample']=='S1')&table.method.eq('Approximate: longer path')].iloc[0]
text=f'''# Does exact TSP remove the observed DI variation?

**Yes, for both fixed samples in this experiment.** Replacing only the randomized approximate open-TSP ordering by a deterministic exact optimum removes the previously observed seed variation: S1's DI range falls from {fmt(s['S1']['approximate_DI_range'])} to exactly0, and S2's from {fmt(s['S2']['approximate_DI_range'])} to exactly0. These zeros are observed bit-identical repeats, not rounded small ranges.

We retained the existing clustered population (N=144,n=8), UP vector, and samples S1={design['samples'][0]}, S2={design['samples'][1]}. Every OT initialization used the observed sample. The same manuscript-adapted package pipeline retained epsilon0.01, squared medoids/matching, pi/8 angular scaling, unweighted box kernel, fixed covariance-based Scott H, and numerical tolerances. Public-package discrepancies remain documented in `sources/discrepancies.md`; no defaults were silently substituted or source files changed.

Only the20 earlier official runs are compared; stress initializations are excluded. We recomputed the complete reference and fresh densities with exact ordering for seeds0–9 per sample, plus each optimum's reverse once:22 new runs.

## Compact comparison

{md}
Lengths use normalized coordinates. This table uses10 significant digits; CSVs retain17 and exact lengths80 decimal digits. The CSV additionally records medoids, hard IP ranges, and compactness. No figure is needed.

## Why the difference disappears

The heuristic returned either the optimal path or one about3.0746% longer. Their preliminary OT transports and centroids are bit-identical for each sample: the difference begins at ordering. The longer path changes7 hard memberships and replaces3 of8 medoids, altering translations and densities. Exact ordering recovers the shorter heuristic path's reference. Its medoid set is [37,61,99,104,116,121,123,132] for both samples, with sample-specific matching. Both Dnet signs remain unchanged; this is not a near-zero sign reversal. Full comparisons and geometry are saved.

## Optima and deterministic tie breaking

Exhaustive enumeration covers all40320 orientations, or20160 reversal classes. Lengths use80-digit Euclidean arithmetic, with a120-digit winning-path check. Independent Held–Karp dynamic programming agrees. Each sample has one undirected optimum: forward/reverse are its only two oriented optima. The next distinct path is about0.00909782 longer, far above numerical tie tolerance; full values are saved.

Tie rule: rank nodes lexicographically by(x,y), canonicalize each path against its reverse, then choose the smallest optimal rank sequence. Ties within1e-60 are checked at higher precision; none beyond reversal occurs here. The chosen zero-based rank path is [3,0,1,2,4,5,6,7]. Fresh reversal constructions yield identical hard references, matching, densities and DI; fractional shares differ by at most {fmt(orient.soft_membership_max_abs_difference.max())} from cumulative rounding. Reversal equivalence downstream was tested, not assumed.

## Randomness and what exact TSP does not optimize

Observed centers bypass randomized KMeans++ initialization, and the OT solver's internal seed is fixed. Sorting, splitting, hardening, declared medoid ties, unique matching, and density counts are deterministic here. TSP was the only effective random choice in the examined pipeline. With exact ordering, all intermediate arrays are bit-identical across seeds0–9, including OT, memberships, medoids, matching and densities. This establishes reproducibility in this environment, not platform-independent arithmetic.

Shorter TSP does **not** imply better final compactness. It reduces Qsoft from {fmt(long1.Q_soft)} to {fmt(e1.Q_soft)}, but increases hard weighted squared loss from {fmt(long1.Q_hard)} to {fmt(e1.Q_hard)}. Preliminary OT loss is unchanged. TSP optimizes path length, not hard compactness. Actual hard IP totals are retained without assuming exact balance.

Exact optimization eliminates this TSP-induced variation. It does not prove robustness to other OT solutions, populations, or genuinely different tied optimal paths. Exhaustive search is suitable for n=8, not a proposed large-n method. No manuscript edit or reviewer-response draft was made.

`verification.json` records independent checks of all22 new states,20 prior comparisons, fixed inputs, source hashes, optimal paths, medoids/matching, densities and the original DI identity. The ZIP includes code and all numerical results.
'''
text=re.sub(r'\b([A-Za-z]{3,})(?=\d)',r'\1 ',text)
for a,b in [('of8','of 8'),('or20160','or 20160'),('a120','a 120'),('once:22','once: 22'),('states,20','states, 20'),('by(x,y)','by (x,y)')]:text=text.replace(a,b)
(P/'01_Report.md').write_text(text)
print(P/'01_Report.md')

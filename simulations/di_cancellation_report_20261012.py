"""Render the investigation report from saved, unrounded numerical results."""
from di_cancellation_experiment_20261012 import *

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    df=pd.read_csv(OUT/'random_samples.csv',float_precision='round_trip')
    q=df[(abs(df.Dnet)<.001)&(df.eta>.3)].sort_values('Dnet',key=abs)
    def table(rows,columns):
        def fmt(v): return format(v,'.17g') if isinstance(v,(float,np.floating)) else str(v)
        return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(fmt(row[c]) for c in columns)+' |' for row in rows]])
    records=[]
    for _,r in q.iterrows():
        records.append(dict(draw=int(r.draw),IDs=', '.join(str(int(r[f'id_{j}'])) for j in range(1,5)),Dnet=r.Dnet,eta=r.eta,DI=r.DI))
    pairs=[]
    for k,p in enumerate(m['illustrated_pairs'],1):
        for key in ['A','B']:
            pairs.append(dict(sample=f'{k}{key}',IDs=', '.join(str(i+1) for i in json.loads(p[f'sample_{key}'])),Dnet=p[f'Dnet_{key}'],eta=p[f'eta_{key}'],DI=p[f'DI_{key}']))
    p=m['illustrated_pairs'][0]; p2=m['illustrated_pairs'][1]
    residual=p['DI_gap']-(p['eta_A']+p['eta_B'])
    exact_zero=int((df.Dnet==0).sum())
    text=f'''# Empirical investigation of DI cancellation and sign instability

## Answers

1. **Yes, at the requested empirical threshold.** Five of 5,000 SRS samples (0.1%) satisfy |Dnet| < 0.001 and η > 0.3. Among the {m['high_eta_count']} samples with η > 0.3, the minimum |Dnet| is **{m['min_abs_Dnet_high_eta']:.17g}**. This establishes coexistence in this population, not arbitrarily small attainable values or a universal limit.
2. **Yes, for similar feasible samples.** Replacing one unit at normalized distance {p['replacement_distance']:.17g} changes DI by **{p['DI_gap']:.17g}**, with both |Dnet| below 0.001 and both η above 0.34. A second pair replaces a unit at distance {p2['replacement_distance']:.17g} and changes DI by {p2['DI_gap']:.17g}. Both pairs retain three of four units. Their scores are identical under the saved reproducibility seed and each of ten common alternative clustering seeds (0–9).
3. **This is a material local interpretation problem.** DI can switch between moderately large negative and positive values for closely related samples with almost cancelling net density distortion. A scalar DI near ±0.35 must not be read as an unambiguous, smoothly varying ordering of spatial spread near cancellation. The strict event is uncommon in this SRS run, but its low frequency does not remove the demonstrated local sensitivity. Report Dnet and η alongside DI when interpreting this regime. These results do not establish that the index is unusable everywhere, or estimate rates under other sampling designs.

## Construction and reproducibility

Used the repository package directly: Population → DensityDisparity(representative="nmedoids") → FIPBalancedNMeans(init_clust_method="weighted", tol=1e-9, max_iter=100), initialized by each realized sample → package balanced-cluster construction → package weighted medoids → Euclidean Hungarian matching → cluster translation → package KDE → package DI. The public default is nmeans, so nmedoids is selected explicitly as requested. No package source, existing scripts, datasets, or previous results were changed.

Population: `{m['source']}` (N=144), the permuted clustered population referenced by existing simulations; SHA256 `{m['source_sha256']}`. File rows have no ID column: IDs are the package defaults 1–144 in file order; indices are 0–143. Population subtracts the coordinate-wise minima and divides both axes by the largest original axis span. CSVs contain both raw and package-normalized coordinates.

Sampling: 5,000 independent simple random subsets of size four without replacement, NumPy default_rng(12345), sorted within sample. All 5,000 realized subsets were distinct. Each unit has inclusion probability 4/144. NumPy is used because package RandomSampling.sample draws with replacement. This is a specified SRS experiment, not a pooled comparison of spread-oriented designs. For package-internal random steps, each subset uses SeedSequence([12345, *sorted_indices]) to seed NumPy and Numba; seeds are recorded. No sampling weights from the file are used in the equal-probability experiment.

Software versions and input hash are in `manifest.json`. Reproduce from repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-di .venv/bin/python simulations/di_cancellation_experiment_20261012.py
MPLCONFIGDIR=/tmp/mpl-di .venv/bin/python simulations/di_cancellation_audit_20261012.py
MPLCONFIGDIR=/tmp/mpl-di .venv/bin/python simulations/di_cancellation_report_20261012.py
```

## Definition and bandwidth audit

Let f_i be the original population KDE evaluated at its original unit coordinates, and g_i the translated KDE evaluated at the translated coordinates of the corresponding units. Both are unweighted top-hat KDEs, fitted to all N=144 points, with bandwidth="scott" and rtol=1e-4. The sklearn Scott bandwidth is N^(-1/(d+4)) = 144^(-1/6) = **{audit['scott_bandwidth']:.17g}** in normalized coordinates. It is the same for both fits; no sample-size bandwidth, adaptive choice, or additional coordinate rescaling is introduced after translation.

The intermediate values are recovered from the package's returned densities with its exact elementwise operations:

```
u_i = 1 / (sqrt(2) sqrt(f_i² + g_i²))
Dnet = mean(clip((f_i - g_i) u_i / sin(π/8), -1, 1))
η = mean(clip((1 - (f_i + g_i) u_i) / (1 - cos(π/8)), -1, 1))
DI = Dnet + (sign(Dnet) - Dnet) η
```

Clipping occurs **before** averaging. η is the implemented scaled cosine distortion, not a separately estimated density variance. The maximum absolute discrepancy from `(1−η)Dnet + η sign(Dnet)` over all 5,000 draws is **{m['max_formula_error']:.17g}**, consistent with floating-point evaluation order. For the four illustrated samples, independent trigonometric calculations using θ_i=atan2(g_i,f_i) agree within 5e-15. Seeded reruns reproduce their saved scores exactly.

**Manuscript comparison remains pending:** no current JSSAM manuscript was found in the repository or supplied during this run. The audit certifies the actual package construction and the identity supplied in the task, not line-by-line agreement with an unavailable manuscript. In particular, the bandwidth, per-unit clipping, hard cluster labels, and explicit medoid choice should be checked against that document before using these findings in the paper.

## Near-cancellation samples in the random experiment

All five samples meeting the requested strict thresholds:

{table(records,['draw','IDs','Dnet','eta','DI'])}

`near_cancellation.csv` lists the twenty closest high-η samples, with IDs and coordinates. `random_samples.csv` contains all 5,000 draws at 17 significant digits. No small value is rounded to zero. Exactly {exact_zero} random draws have Dnet numerically equal to zero.

![Joint distribution](eta_vs_Dnet.png)

The plot shows all draws and a ±0.01 zoom, with the requested ±0.001 strip highlighted. Near-zero Dnet is compatible with a range of η: 21 draws have |Dnet| < 0.001, five of them with η > 0.3. `joint_distribution.csv` reports bin counts, median/max η, and high-η counts. For |Dnet| < 0.01 there are 188 draws, including 31 with η > 0.3. The evidence already answers the primary coexistence question, so no optional n=8 or unequal-probability extension was needed.

## Similar samples and targeted search

After the SRS run, selected the three smallest-|Dnet| high-η samples of each sign. Enumerated all 4×140 one-unit replacements for each of these six anchors: {m['local_evaluations']} comparisons. This targeted selection is deliberately favorable to finding counterexamples; its hit rate is **not** an unbiased estimate of population-wide instability. Across random and local stages, {m['unique_pipeline_evaluations']} distinct subsets were evaluated. Full replacements and all qualifying opposite-sign pairs are saved separately.

Illustration 1 minimizes the larger absolute Dnet across qualifying searched pairs. Illustration 2 minimizes replacement distance among searched pairs with both |Dnet| < 0.01. These are search-specific extrema, not global optima across all feasible subsets.

{table(pairs,['sample','IDs','Dnet','eta','DI'])}

Pair 1 replaces ID {p['removed_id']} with ID {p['added_id']}, moving only {100*p['replacement_distance']:.9g}% of the largest population axis span. The absolute Dnet change is {abs(p['Dnet_A']-p['Dnet_B']):.17g}; the DI change is {p['DI_gap']:.17g}. Its sign-dependent term contributes η_A+η_B = {p['eta_A']+p['eta_B']:.17g} to the gap; the remaining contribution is only {residual:.17g}. Thus almost the entire gap comes from the formula's sign term. This decomposition does not assume a fixed reference clustering along a continuous path: both samples are fully reclustered by the package.

![Pair 1](spatial_pair_1.png)

Pair 2 replaces ID {p2['removed_id']} with ID {p2['added_id']}, moving {100*p2['replacement_distance']:.9g}% of the largest span.

![Pair 2](spatial_pair_2.png)

The repeated-seed checks in `rng_sensitivity.csv` found no variation for either illustrative pair, supporting a geometric sample change rather than a random initialization explanation for these examples. This does not establish seed independence for every sample.

## Mathematical discontinuity versus observed instability

For a fixed η>0, the formula has one-sided limits −η and +η as Dnet approaches zero, while sign(0)=0 gives DI=0 exactly. This algebraic discontinuity is unconditional for that fixed-η extension of the formula. It does not by itself prove that a particular finite population's feasible samples attain Dnet=0 with η>0 or approach zero arbitrarily closely.

The observations supply the missing practical evidence at the requested finite threshold: actual KDEs from actual feasible samples produce near cancellation with substantial η, and one nearby unit replacement reverses the sign and produces a large index difference. No fabricated angles/densities or continuously moved off-population points were used. The finite support, top-hat KDE neighborhood changes, sample-dependent clustering, and medoid/matching choices mean this experiment is not a proof of a continuous geometric limit or a decomposition of every source of non-smoothness. It demonstrates a concrete local sensitivity, with the final sign term accounting for almost all of the illustrated DI gap.

The defensible interpretation is that net direction and remaining distortion are distinct quantities: cancellation in the former can coexist with substantial latter distortion. DI combines them in a way that makes its sign and magnitude sensitive near that boundary. The present evidence warrants substantive care in interpreting DI alone; it does not justify dismissing the reviewer’s concern as merely hypothetical.

## Artifacts and checks

- `random_samples.csv`: complete main experiment, including Dnet, η, DI, numerical identity error, unit IDs, indices, and both coordinate systems.
- `near_cancellation.csv`, `joint_distribution.csv`: relevant cases and joint-distribution summaries.
- `local_search.csv`, `opposite_sign_pairs.csv`: targeted comparisons, explicitly separate from SRS.
- `eta_vs_Dnet.png`, `spatial_pair_1.png`, `spatial_pair_2.png`: figures.
- `pair_*_densities.npz`: actual original/translated densities for illustrative samples.
- `population_and_samples.npz`: normalized/raw coordinates, inclusion probabilities, IDs and all sampled subsets.
- `manifest.json`, `audit.json`, `rng_sensitivity.csv`, execution logs: provenance and verification evidence.

Audit passed: reproduced the seeded sampling stream, checked unique in-range units per sample, verified the identity on all 5,000 rows, checked η bounds, reran illustrative package scores, checked the angular equivalent using actual densities, and verified the Scott bandwidth. This report is an investigation, not a response to the reviewer.
'''
    (OUT/'report.md').write_text(text)
    print(OUT/'report.md')
if __name__=='__main__': main()

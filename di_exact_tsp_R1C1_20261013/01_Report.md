# Does exact TSP remove the observed DI variation?

**Yes, for both fixed samples in this experiment.** Replacing only the randomized approximate open-TSP ordering by a deterministic exact optimum removes the previously observed seed variation: S1's DI range falls from 0.027924624250291685 to exactly 0, and S2's from 0.0043178696430701269 to exactly 0. These zeros are observed bit-identical repeats, not rounded small ranges.

We retained the existing clustered population (N=144,n=8), UP vector, and samples S1=[13, 16, 33, 50, 59, 75, 97, 127], S2=[10, 12, 13, 26, 66, 122, 123, 144]. Every OT initialization used the observed sample. The same manuscript-adapted package pipeline retained epsilon 0.01, squared medoids/matching, pi/8 angular scaling, unweighted box kernel, fixed covariance-based Scott H, and numerical tolerances. Public-package discrepancies remain documented in `sources/discrepancies.md`; no defaults were silently substituted or source files changed.

Only the 20 earlier official runs are compared; stress initializations are excluded. We recomputed the complete reference and fresh densities with exact ordering for seeds 0–9 per sample, plus each optimum's reverse once: 22 new runs.

## Compact comparison

| Sample | Ordering | Runs | Path length | Gap (%) | Dnet | eta | DI |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | Approximate: optimal path | 8 | 1.671658552 | 0 | 0.09329787958 | 0.2056985713 | 0.2798052104 |
| S1 | Approximate: longer path | 2 | 1.723055623 | 3.0746154 | 0.04457095233 | 0.2169806688 | 0.2518805861 |
| S1 | Exact: deterministic path | 10 | 1.671658552 | 0 | 0.09329787958 | 0.2056985713 | 0.2798052104 |
| S2 | Approximate: longer path | 4 | 1.723055622 | 3.0746154 | -0.3367501365 | 0.3766409137 | -0.5865571711 |
| S2 | Approximate: optimal path | 6 | 1.671658551 | 0 | -0.3347211841 | 0.3720517044 | -0.5822393015 |
| S2 | Exact: deterministic path | 10 | 1.671658551 | 0 | -0.3347211841 | 0.3720517044 | -0.5822393015 |

Lengths use normalized coordinates. This table uses 10 significant digits; CSVs retain 17 and exact lengths 80 decimal digits. The CSV additionally records medoids, hard IP ranges, and compactness. No figure is needed.

## Why the difference disappears

The heuristic returned either the optimal path or one about 3.0746% longer. Their preliminary OT transports and centroids are bit-identical for each sample: the difference begins at ordering. The longer path changes 7 hard memberships and replaces 3 of 8 medoids, altering translations and densities. Exact ordering recovers the shorter heuristic path's reference. Its medoid set is [37,61,99,104,116,121,123,132] for both samples, with sample-specific matching. Both Dnet signs remain unchanged; this is not a near-zero sign reversal. Full comparisons and geometry are saved.

## Optima and deterministic tie breaking

Exhaustive enumeration covers all 40320 orientations, or 20160 reversal classes. Lengths use 80-digit Euclidean arithmetic, with a 120-digit winning-path check. Independent Held–Karp dynamic programming agrees. Each sample has one undirected optimum: forward/reverse are its only two oriented optima. The next distinct path is about 0.00909782 longer, far above numerical tie tolerance; full values are saved.

Tie rule: rank nodes lexicographically by (x,y), canonicalize each path against its reverse, then choose the smallest optimal rank sequence. Ties within 1e-60 are checked at higher precision; none beyond reversal occurs here. The chosen zero-based rank path is [3,0,1,2,4,5,6,7]. Fresh reversal constructions yield identical hard references, matching, densities and DI; fractional shares differ by at most 2.0276835765997703e-13 from cumulative rounding. Reversal equivalence downstream was tested, not assumed.

## Randomness and what exact TSP does not optimize

Observed centers bypass randomized KMeans++ initialization, and the OT solver's internal seed is fixed. Sorting, splitting, hardening, declared medoid ties, unique matching, and density counts are deterministic here. TSP was the only effective random choice in the examined pipeline. With exact ordering, all intermediate arrays are bit-identical across seeds 0–9, including OT, memberships, medoids, matching and densities. This establishes reproducibility in this environment, not platform-independent arithmetic.

Shorter TSP does **not** imply better final compactness. It reduces Qsoft from 0.082045533579329616 to 0.080883500327871505, but increases hard weighted squared loss from 0.081731745255015656 to 0.083171129048574655. Preliminary OT loss is unchanged. TSP optimizes path length, not hard compactness. Actual hard IP totals are retained without assuming exact balance.

Exact optimization eliminates this TSP-induced variation. It does not prove robustness to other OT solutions, populations, or genuinely different tied optimal paths. Exhaustive search is suitable for n=8, not a proposed large-n method. No manuscript edit or reviewer-response draft was made.

`verification.json` records independent checks of all 22 new states, 20 prior comparisons, fixed inputs, source hashes, optimal paths, medoids/matching, densities and the original DI identity. The ZIP includes code and all numerical results.

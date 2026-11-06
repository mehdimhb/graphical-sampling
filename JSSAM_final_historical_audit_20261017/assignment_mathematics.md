# Translation and the matching objective

No Monte Carlo, reference fitting, or DI recomputation is needed for this diagnostic. Run `assignment_counterexample.py` to verify all costs, the SciPy assignments, and the exact integer squared-cost increment.

## Fixed medoids, sample-only translation

Let fixed medoids be r1=(0,0), r2=(2,0), with original sample s1=(0,0), s2=(1,1). Translate both sample units by delta=(3,0), giving s1'=(3,0), s2'=(4,1). The identity pairs r1→s1, r2→s2; the swap pairs r1→s2, r2→s1.

| Objective | Original identity | Original swap | Shifted identity | Shifted swap |
| --- | ---: | ---: | ---: | ---: |
| Sum of Euclidean distances | √2 = 1.414214 | 2+√2 = 3.414214 | 3+√5 = 5.236068 | 1+√17 = 5.123106 |
| Sum of squared distances | 2 | 6 | 14 | 18 |

Both Euclidean optima are unique: identity before translation, swap afterwards. The shifted difference (3+√5)−(1+√17)=0.112962351882129... is positive; the script checks it with 70-digit Decimal arithmetic. This is not a tie-breaking or floating-point artifact. Dividing all coordinates and delta by five gives the same strict reversal with all medoids and both samples in the unit square.

The two original cluster translations are (0,0) and (−1,1). Under the new Euclidean-optimal assignment they are (4,1) and (1,0). Their changes are (4,1) and (2,−1), rather than one common delta. Consequently the usual common-translation proof for the transformed population cannot be invoked. This example proves failure of assignment invariance, **not that every rematching changes DI**: a particular KDE/population can still yield the same DI by coincidence or symmetry.

## Why squared matching is different

For Q_delta(tau)=sum_i ||s_tau(i)+delta−r_i||²,

Q_delta(tau) = Q_0(tau) + 2 delta·(sum_i s_i−sum_i r_i) + n||delta||².

The increment is independent of the permutation. Hence the entire exact minimizing-permutation set is unchanged. In this example the common increment is 12 and the unique squared optimum stays the identity. If minima are tied, a consistent tie rule is additionally needed to claim the selected permutation, rather than its set, is unchanged.

This proof assumes an unweighted, full bijection between two sets of the same size n. It does not automatically apply to historical rectangular K=n−1 matching: different assignments can then select different sample subsets, so the sum of matched sample coordinates need not be constant.

For ordinary distances L_delta(tau)=sum_i ||s_tau(i)+delta−r_i||, there is no analogous general permutation-independent expansion. The counterexample disproves the universal assignment claim in dimension two.

Two qualifications:

- Jointly translating **both** medoids and sample leaves every cost unchanged for either metric. That is a different operation from moving the sample within a fixed population/reference.
- If the sample is itself a uniformly translated copy of the medoid pattern, the corresponding natural pairing remains Euclidean-optimal: its cost n||v+delta|| attains the triangle-inequality lower bound for every assignment. Additional Euclidean ties may appear, so this does not guarantee an unchanged minimizing set or selected permutation. It explains why particularly regular reference-pattern examples can remain invariant without proving the general sample-only claim.

## The valid conditional DI statement

Fix the hard population partition, its medoids, and the medoid-to-sample pairing. Translating the sample by delta changes each transformed population point from T(c_l) to T(c_l)+delta. A fixed-bandwidth translation-invariant KDE, evaluated at the corresponding transformed point as in the historical package, has the same translated-density value for each unit. The original density values are fixed, so the full DI formula is unchanged.

This statement holds for either spherical or box KDE and does not require squared matching **when the pairing is explicitly held fixed**. With squared-optimal matching it also holds after rematching provided the reference is fixed and ties are handled consistently. With Euclidean-optimal rematching, assignment stability is an additional condition to verify; the counterexample shows it is not automatic.

Recomputing a sample-adaptive partition or medoids can invalidate the fixed-reference condition under either matching objective. The recent EP/UP translation examples used explicitly squared matching, independently refitted references, and saved zero membership changes. Their observed invariance remains valid for those saved configurations, but does not establish a universal result or reproduce the historical Euclidean pipeline. The older Figure 9 examples and their grid-projected rotations must also remain empirical illustrations; rounded DI=0.00 is not a proof of exact equality.

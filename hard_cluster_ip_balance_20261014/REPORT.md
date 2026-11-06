# Hard-cluster IP balance in the Measure populations

14 October 2026. Balance-only, sample-initialized OT IPnM experiment; no DI/KDE or full manuscript Monte Carlo simulation. Existing manuscript, code and results were preserved.

**Coverage:** 106/180 prespecified sample cases converged, covering 39/60 settings and 1,720 hard clusters. 30 settings have three completed replicates. 32 attempted cases reached a convergence/time limit; 42 additional cases were not attempted after an expensive first replicate. All requested settings and issues remain in [summary.csv](summary.csv). Missingness may depend on initialization, so these are descriptive results from the completed cases.

| Population (N) | IP | Settings / 6 | Runs / 18 | Observed mass range | Largest setting MAE | Worst cluster error |
|---|---|---:|---:|---|---:|---:|
| Gridded (144) | EP | 5 | 10 | 0.6944–1.1111 | 0.1111 | 0.3056 |
| Gridded (144) | UP | 5 | 15 | 0.3686–1.6267 | 0.1758 | 0.6314 |
| Random (144) | EP | 5 | 12 | 0.6944–1.1111 | 0.1111 | 0.3056 |
| Random (144) | UP | 4 | 11 | 0.7581–1.2960 | 0.0867 | 0.2960 |
| Clustered (144) | EP | 4 | 12 | 0.8889–1.1111 | 0.1111 | 0.1111 |
| Clustered (144) | UP | 4 | 12 | 0.5996–1.3497 | 0.1108 | 0.4004 |
| Meuse (155) | EP | 3 | 9 | 0.9032–1.0323 | 0.0484 | 0.0968 |
| Meuse (155) | UP | 3 | 9 | 0.8641–1.1234 | 0.0482 | 0.1359 |
| Swiss (959) | EP | 3 | 8 | 0.9802–1.0010 | 0.0020 | 0.0198 |
| Swiss (959) | UP | 3 | 8 | 0.9622–1.0349 | 0.0158 | 0.0378 |

MAE means mean absolute error from mass 1. The table uses available converged runs; min/max and worst errors span those runs. The 60-row [summary.csv](summary.csv) contains all eight requested statistics, probability validity and replicate counts. [Cluster measurements](cluster_measurements.csv) give every cardinality, mass and error. Statistics pool clusters within each setting; SD uses ddof=0.

**Answers to Reviewer 1, Comment 4**

1. **How close to one?** At n/N ≤ .10, the largest setting MAE is 0.0323 under EP and 0.0223 under UP; worst individual errors are 0.0323 and 0.0490. This fraction split is descriptive. For simulated EP populations, n=4,8,16 gives exactly unit masses in mathematical arithmetic; n=32 gives .8889/1.1111.
2. **Does imbalance increase with n/N?** Larger deviations appear at higher tested fractions, especially under UP, but the relation is not monotone. Gridded EP n=32→50 has MAE .1111→.0733 while the maximum error rises .1111→.3056. High-fraction conclusions remain limited: no simulated n=100, Meuse n≥50 or Swiss n≥50 OT case converged within budget. Separate [exact EP predictions](EP_arithmetic_predictions.csv) apply independently of OT convergence; for Meuse n=100→150, predicted MAE falls .3194→.0624 while maximum error rises .3548→.9355. These predictions are not empirical measurements.
3. **EP versus UP?** EP error is bounded by n/N and is zero when n divides N. UP is governed by max(pi), which may be much larger than n/N. Gridded UP n=50 has pooled MAE 0.1758, maximum error 0.6314, and 20.7% of clusters beyond .25. The first tested, converged simulated UP settings with errors >.25 are Gridded: n=32, n/N=0.222; Random: n=32, n/N=0.222; Clustered: n=32, n/N=0.222. These are observed crossings, not estimated universal onset thresholds. Small sampling fractions alone cannot guarantee UP balance.
4. **Is “usually approximately IP-balanced” supported?** The completed low-probability settings support a qualified description of close balance. An unconditional statement across all EP/UP vectors or fractions is unsupported; three samples and incomplete convergence cannot establish a general “usually” frequency.
5. **Suggested manuscript wording:** “The soft formulation yields unit IP mass in every cluster. After maximum-membership hard assignment, the cluster-mass error is bounded by the largest individual inclusion probability. In our evaluated populations, hard-cluster masses were generally close to one when individual inclusion probabilities were small, while larger deviations occurred in some settings. Approximate balance therefore depends on the IP vector and border assignments, rather than being guaranteed by the sampling fraction alone.”

**Actual UP construction and invalid proportional vectors**

The preserved Measure code uses stored `prob` weights, Meuse Cu, and **Swiss clip(Area,5,100)**, followed by its existing certainty-unit helper. All 60 primary vectors satisfy 0 < pi ≤ 1; the largest recorded sum-to-n rounding residual is 1.42e-14. No new capping, probability normalization, weight generation or site deletion was introduced. Code evidence is preserved in `sources/probability_construction_evidence.json`.

Simple proportional normalization is invalid in six UP settings: Gridded n=100 (max pi 1.717); Random n=100 (max pi 1.219); Clustered n=100 (max pi 1.389); Meuse n=50 (max pi 1.024); Meuse n=100 (max pi 2.048); Meuse n=150 (max pi 3.072). [probability_audit.csv](probability_audit.csv) reports these vectors; none was passed to OT. Primary vectors at those settings follow the original code's **certainty adjustment**, so they cease to be strictly proportional. Raw Swiss Area also has zero weights. The manuscript should explicitly document its actual Area clipping and certainty handling. No N=144 setting uses n=150.

**Procedure and checks**

Three initial samples per setting were fixed before measuring balance: reproducible SRS for EP and random-order pivotal sampling for UP, including certainty units. Six prior clustered EP n16/n32 couplings were reused. Fresh OT uses the existing epsilon=.01, squared-distance scaling, 1e-10 marginal tolerance, 1e-7 outer tolerance and ≤2e-7 final centroid residual. Each fresh OT case has a 240-process-CPU-second budget and the documented iteration limits. A limit establishes nonconvergence within this budget, not impossibility. Some small-n initializations also failed.

Fixed TSP seed 20261014, 15 documented approximate restarts, canonical orientation and the current corrected exact-overlap hardening are used throughout. No TSP sensitivity is tested. Empty clusters are retained for measurement. [README.md](README.md) gives complete methods and reproduction commands; [CODE_REVIEW.md](CODE_REVIEW.md) records the independent review and corrections.

All completed states passed independent exact overlap/assignment, mass, cardinality, threshold-count and OT checks. **Bound violations: zero. Empty hard clusters: 0.** Largest soft-mass residual: 3.33e-15. All 62 protected source/input artifacts retain their hashes. Four full fixed-seed repeats reproduced all 11 stored arrays bit for bit. Evidence: [verification.json](verification.json), [verification.csv](verification.csv), [repeatability.json](repeatability.json).

**Files and resource use**

- [Mass distributions](mass_distributions.png): target 1, shaded ±.10/±.25; unavailable settings marked ×.
- [Imbalance versus fraction](imbalance_vs_fraction.png): mean and maximum error, EP/UP distinguished; missing settings break curves. Both figures also have PDFs.
- [Reproducible code](experiment.py), source/data snapshots and frozen states accompany the CSVs and this report in the ZIP.

At most eight single-threaded, lowered-priority workers ran on 16 logical CPUs. CPU and memory were monitored every two seconds; concurrency and suspension adapted to competing load. Recorded median total CPU was 50.1%; maximum 30-second rolling CPU was 56.9%, below 60%. Peak machine memory use was 33.9%. Both resource logs are retained. The first controller sometimes underutilized CPU; the improved replicate-phase controller persistently suspends excess workers.

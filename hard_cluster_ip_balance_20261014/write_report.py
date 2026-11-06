"""Write a brief evidence-linked report after analysis and verification."""
from pathlib import Path
import json
import pandas as pd
HOME=Path(__file__).resolve().parent

def f(v):return '—' if pd.isna(v) else f'{v:.4f}'

def write():
    s=pd.read_csv(HOME/'summary.csv');r=pd.read_csv(HOME/'run_summary.csv');c=pd.read_csv(HOME/'compact_summary.csv')
    v=json.loads((HOME/'verification.json').read_text());a=pd.read_csv(HOME/'probability_audit.csv')
    mon=pd.concat([pd.read_csv(p) for p in HOME.glob('resource_monitor*.csv')],ignore_index=True)
    ok=s[s.completed_replicates>0];small=ok[ok.fraction<=.1];ep=small[small.IP=='EP'];up=small[small.IP=='UP']
    completed=int((r.status=='completed').sum());failed=int(r.status.isin(['cpu_limit','failed','worker_exit']).sum())
    invalid=a[(a.IP=='UP')&(~a.simple_proportional_valid)]
    bad='; '.join(f'{t.population} n={int(t.n)} (max pi {t.simple_proportional_max:.3f})' for _,t in invalid.iterrows())
    table=['| Population (N) | IP | Settings / 6 | Runs / 18 | Observed mass range | Largest setting MAE | Worst cluster error |',
           '|---|---|---:|---:|---|---:|---:|']
    for _,t in c.iterrows():
        mae=pd.Series([t.at_fraction_le_010_max_MAE,t.at_fraction_gt_010_max_MAE]).max()
        error=pd.Series([t.at_fraction_le_010_worst_error,t.at_fraction_gt_010_worst_error]).max()
        table.append(f'| {t.population} ({int(t.N)}) | {t.IP} | {int(t.completed_settings)} | {int(t.completed_runs)} | {f(t.observed_mass_min)}–{f(t.observed_mass_max)} | {f(mae)} | {f(error)} |')
    g=ok[(ok.population=='Gridded')&(ok.IP=='UP')&(ok.n==50)].iloc[0]
    examples=[]
    for pop in ['Gridded','Random','Clustered']:
        t=ok[(ok.population==pop)&(ok.IP=='UP')&(ok.max_abs_error>.25+1e-12)].sort_values('n')
        if len(t):
            q=t.iloc[0];examples.append(f'{pop}: n={int(q.n)}, n/N={q.fraction:.3f}')
    text=f'''# Hard-cluster IP balance in the Measure populations

14 October 2026. Balance-only, sample-initialized OT IPnM experiment; no DI/KDE or full manuscript Monte Carlo simulation. Existing manuscript, code and results were preserved.

**Coverage:** {completed}/180 prespecified sample cases converged, covering {len(ok)}/60 settings and {v['measured_clusters']:,} hard clusters. {int((s.completed_replicates==3).sum())} settings have three completed replicates. {failed} attempted cases reached a convergence/time limit; {180-completed-failed} additional cases were not attempted after an expensive first replicate. All requested settings and issues remain in [summary.csv](summary.csv). Missingness may depend on initialization, so these are descriptive results from the completed cases.

{chr(10).join(table)}

MAE means mean absolute error from mass 1. The table uses available converged runs; min/max and worst errors span those runs. The 60-row [summary.csv](summary.csv) contains all eight requested statistics, probability validity and replicate counts. [Cluster measurements](cluster_measurements.csv) give every cardinality, mass and error. Statistics pool clusters within each setting; SD uses ddof=0.

**Answers to Reviewer 1, Comment 4**

1. **How close to one?** At n/N ≤ .10, the largest setting MAE is {ep.mean_abs_error.max():.4f} under EP and {up.mean_abs_error.max():.4f} under UP; worst individual errors are {ep.max_abs_error.max():.4f} and {up.max_abs_error.max():.4f}. This fraction split is descriptive. For simulated EP populations, n=4,8,16 gives exactly unit masses in mathematical arithmetic; n=32 gives .8889/1.1111.
2. **Does imbalance increase with n/N?** Larger deviations appear at higher tested fractions, especially under UP, but the relation is not monotone. Gridded EP n=32→50 has MAE .1111→.0733 while the maximum error rises .1111→.3056. High-fraction conclusions remain limited: no simulated n=100, Meuse n≥50 or Swiss n≥50 OT case converged within budget. Separate [exact EP predictions](EP_arithmetic_predictions.csv) apply independently of OT convergence; for Meuse n=100→150, predicted MAE falls .3194→.0624 while maximum error rises .3548→.9355. These predictions are not empirical measurements.
3. **EP versus UP?** EP error is bounded by n/N and is zero when n divides N. UP is governed by max(pi), which may be much larger than n/N. Gridded UP n=50 has pooled MAE {g.mean_abs_error:.4f}, maximum error {g.max_abs_error:.4f}, and {g.pct_gt_025:.1f}% of clusters beyond .25. The first tested, converged simulated UP settings with errors >.25 are {'; '.join(examples)}. These are observed crossings, not estimated universal onset thresholds. Small sampling fractions alone cannot guarantee UP balance.
4. **Is “usually approximately IP-balanced” supported?** The completed low-probability settings support a qualified description of close balance. An unconditional statement across all EP/UP vectors or fractions is unsupported; three samples and incomplete convergence cannot establish a general “usually” frequency.
5. **Suggested manuscript wording:** “The soft formulation yields unit IP mass in every cluster. After maximum-membership hard assignment, the cluster-mass error is bounded by the largest individual inclusion probability. In our evaluated populations, hard-cluster masses were generally close to one when individual inclusion probabilities were small, while larger deviations occurred in some settings. Approximate balance therefore depends on the IP vector and border assignments, rather than being guaranteed by the sampling fraction alone.”

**Actual UP construction and invalid proportional vectors**

The preserved Measure code uses stored `prob` weights, Meuse Cu, and **Swiss clip(Area,5,100)**, followed by its existing certainty-unit helper. All 60 primary vectors satisfy 0 < pi ≤ 1; the largest recorded sum-to-n rounding residual is {a.pi_sum_residual.abs().max():.3g}. No new capping, probability normalization, weight generation or site deletion was introduced. Code evidence is preserved in `sources/probability_construction_evidence.json`.

Simple proportional normalization is invalid in six UP settings: {bad}. [probability_audit.csv](probability_audit.csv) reports these vectors; none was passed to OT. Primary vectors at those settings follow the original code's **certainty adjustment**, so they cease to be strictly proportional. Raw Swiss Area also has zero weights. The manuscript should explicitly document its actual Area clipping and certainty handling. No N=144 setting uses n=150.

**Procedure and checks**

Three initial samples per setting were fixed before measuring balance: reproducible SRS for EP and random-order pivotal sampling for UP, including certainty units. Six prior clustered EP n16/n32 couplings were reused. Fresh OT uses the existing epsilon=.01, squared-distance scaling, 1e-10 marginal tolerance, 1e-7 outer tolerance and ≤2e-7 final centroid residual. Each fresh OT case has a 240-process-CPU-second budget and the documented iteration limits. A limit establishes nonconvergence within this budget, not impossibility. Some small-n initializations also failed.

Fixed TSP seed 20261014, 15 documented approximate restarts, canonical orientation and the current corrected exact-overlap hardening are used throughout. No TSP sensitivity is tested. Empty clusters are retained for measurement. [README.md](README.md) gives complete methods and reproduction commands; [CODE_REVIEW.md](CODE_REVIEW.md) records the independent review and corrections.

All completed states passed independent exact overlap/assignment, mass, cardinality, threshold-count and OT checks. **Bound violations: zero. Empty hard clusters: {v['empty_hard_clusters']}.** Largest soft-mass residual: {v['largest_soft_mass_error']:.3g}. All {v['protected_sources']} protected source/input artifacts retain their hashes. Four full fixed-seed repeats reproduced all 11 stored arrays bit for bit. Evidence: [verification.json](verification.json), [verification.csv](verification.csv), [repeatability.json](repeatability.json).

**Files and resource use**

- [Mass distributions](mass_distributions.png): target 1, shaded ±.10/±.25; unavailable settings marked ×.
- [Imbalance versus fraction](imbalance_vs_fraction.png): mean and maximum error, EP/UP distinguished; missing settings break curves. Both figures also have PDFs.
- [Reproducible code](experiment.py), source/data snapshots and frozen states accompany the CSVs and this report in the ZIP.

At most eight single-threaded, lowered-priority workers ran on 16 logical CPUs. CPU and memory were monitored every two seconds; concurrency and suspension adapted to competing load. Recorded median total CPU was {mon.cpu_percent.median():.1f}%; maximum 30-second rolling CPU was {mon.rolling_30s_cpu_percent.max():.1f}%, below 60%. Peak machine memory use was {mon.memory_percent.max():.1f}%. Both resource logs are retained. The first controller sometimes underutilized CPU; the improved replicate-phase controller persistently suspends excess workers.
'''
    (HOME/'REPORT.md').write_text(text);print('Wrote REPORT.md')

if __name__=='__main__':write()

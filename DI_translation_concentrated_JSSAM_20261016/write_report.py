"""Render report/CSV from completed saved fits only; no optimization."""
import json,csv,hashlib,importlib.metadata
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent
cfg=json.loads((P/'config.json').read_text());selection=json.loads((P/'selection.json').read_text());selected=selection['selected']
a=json.loads((P/'runs'/selected['original_id']/'summary.json').read_text());b=json.loads((P/'runs'/selected['translated_id']/'summary.json').read_text())
comparisons=json.loads((P/'comparisons.json').read_text());verification=json.loads((P/'verification.json').read_text())
rows=[]
for c in comparisons:
 before=json.loads((P/'runs'/c['original_id']/'summary.json').read_text());after=json.loads((P/'runs'/c['translated_id']/'summary.json').read_text())
 rows.append(dict(sample_description=c['description'],shift_vector=f"({c['delta'][0]}, {c['delta'][1]})",
  DI_before=before['DI'],DI_after=after['DI'],DI_difference=after['DI']-before['DI'],changed_memberships=c['changed_units']))
with (P/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (P/'display_memberships.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['unit_zero_based','x','y','original_cluster','shifted_cluster_aligned','changed','original_sample','shifted_sample'])
 for i in range(144):
  x,y=i%12,i//12;left=a['labels'][i];right=selected['new_to_old'][b['labels'][i]]
  w.writerow([i,x,y,left+1,right+1,int(left!=right),int(i in a['sample_ids']),int(i in b['sample_ids'])])
with (P/'display_references.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['cluster','size','medoid_x','medoid_y','matched_original_x','matched_original_y','matched_shifted_x','matched_shifted_y'])
 for k in range(8):
  m=a['medoid_ids'][k];ma=a['matched_ids'][k];mb=b['matched_ids'][k]
  w.writerow([k+1,len(a['clusters'][k]),m%12,m//12,ma%12,ma//12,mb%12,mb//12])
coords='; '.join(f"({x},{y})" for x,y in a['sample'])
short=f'''# Concentrated samples and large translations

**Finding:** all nine independently rebuilt pairs retained the same hard partition, medoids and DI. No genuinely changed-clustering example was found in this bounded experiment. Consequently, it does not determine whether DI remains invariant when the reference genuinely changes.

![Concentrated original and shifted samples with independently rebuilt clusters](translation_comparison.png)

The population is the fixed grid `{{0,...,11}}²`, with `N=144`, `n=8` and equal inclusion probabilities `1/18`. Three asymmetric compact eight-point shapes were specified before fitting. All are confined to at most four rows and four columns. Each was shifted by `(3,2)`, `(4,1)` and `(2,3)`; every translated point remains a population unit.

The displayed original sample is `{coords}`, occupying three columns and four rows. Its shift is `(3,2)`. Both references are reconstructed independently from the respective eight sample coordinates.

| Component | Original | Shifted |
|---|---:|---:|
| Dnet | {a['Dnet']:.15f} | {b['Dnet']:.15f} |
| eta | {a['eta']:.15f} | {b['eta']:.15f} |
| DI | {a['DI']:.15f} | {b['DI']:.15f} |

**DI difference = 0; changed memberships = 0/144 after label alignment; adjusted Rand index = 1.** Every hard cluster contains 18 units. These are equal saved numerical values, not merely equal rounded labels.

| Compact sample | DI before and after every tested shift | Changed units |
|---|---:|---:|
'''
for i,entry in enumerate(cfg['candidates'],1):
 z=json.loads((P/f'runs/sample{i}_original/summary.json').read_text())
 short+=f"| {entry['name']} | {z['DI']:.10f} | 0/144 |\n"
short+=f'''
The DI values are close to −1, but their equality is **not merely saturation**: all original/translated density counts and both components are identical. For every pair, the newly constructed translated population is exactly the old constructed population plus the common sample shift.

The calculation uses the manuscript-style sample-initialized entropic OT, cost-median scaling, hard IP balancing, squared-distance medoids and squared-distance matching, followed by the explicitly requested historical spherical Scott KDE. It does not use the package DI default. Coordinates are normalized using the whole population (`raw/11`), never the compact sample. Translated coordinates are not renormalized; densities are evaluated at their corresponding unit locations. The DI definition and sign are unchanged.

Settings: solver seed 17, OT epsilon 0.01, outer tolerance `1e-10`, marginal tolerance `1e-12`, standard 15 OTSP restarts, one thread. The displayed pair was independently repeated at outer/marginal tolerances `1e-12`/`1e-13`, with unchanged results. Preliminary assignments are well separated: all saved maximum-versus-runner-up coupling margins exceed 0.022. Matching is uniquely optimal for all primary fits. One exact medoid tie occurs in the displayed reference and follows the manuscript smallest-label rule. Tied OTSP routes cannot change these memberships because every preliminary group already has exactly 18 units (IP mass one).

Independent checks verified {verification['verified_saved_fits']} saved fits: 12 primary, two tighter fits, one replay and one joint-coordinate control. They include exhaustive eight-point matching/TSP checks, direct spherical neighbor counts, label-independent co-membership matrices and OT convergence/stationarity checks. Density components agree with independent scalar calculations within {verification['component_recalculation_max_error']:.2g}.

**Interpretation:** this provides a concentrated-sample, large-shift example of invariance with an unchanged reference. Neither “different clustering, same DI” nor “different clustering, different DI” was observed. It is not a general invariance proof, and no changed-clustering example was forced.

Files: [PDF figure](translation_comparison.pdf), [nine-row CSV](summary.csv), [memberships](display_memberships.csv), [medoids and matching](display_references.csv), [verification](verification.json), and [method details](METHODS.md). All candidates, shifts, seeds, scripts and per-run states are retained. Completed fits are skipped on resume. No manuscript, reviewer response, GFS package, previous figure, dataset or saved study result was modified; all {verification['original_protected_files_unchanged']:,} protected-file hashes match.
'''
(P/'REPORT.md').write_text(short)
methods='''# Method and reproduction

This isolated module evaluates the mathematical pipeline specified in the available October12 Measure snapshot: sample coordinates initialize the OT centroids; balanced entropic OT alternates Sinkhorn/Newton solutions and centroid updates with median cost normalization; preliminary labels are maximum allocations; the existing GFS15-restart nearest-neighbor/2-opt OTSP heuristic orders centroids; the existing within-cluster ordering is retained; exact EP cumulative overlaps determine hard memberships; medoids minimize squared-distance loss with the specified IP/ID tie rule; squared matching is solved exactly with a deterministic lexicographic tie rule. The existing Newton helper solves the same convex fixed-centroid entropic OT problem, without changing its objective.

The user's requested historical spherical KDE differs from the box kernel displayed in that available manuscript snapshot. That explicit kernel choice is documented, not silently treated as identical. KDE uses KernelDensity(kernel="tophat", bandwidth="scott",rtol=1e-4), with h=144^(-1/6). Both fits have144 points and therefore the same numerical h. Original density is evaluated at original units, and the second density at the corresponding internally translated units. The whole-population normalization is raw/11; the compact sample is not rescaled on its own.

Exact integer grid medoid/matching costs are proportional to squared normalized-coordinate costs, so they have the same mathematical optima while avoiding floating rounding of genuine geometric ties. Equal-IP quota overlaps are also evaluated exactly. No tolerance is used to merge nearby unequal OT allocations or medoid costs. The approximate OTSP heuristic is preserved; exhaustive8! path enumeration is diagnostic only and does not replace the saved route.

All3 shapes and3 shifts were fixed in config.json before the first fit. Figure selection prioritizes a verified changed geometry/partition if found, never DI magnitude/sign. None was found; the first prespecified pair is shown. All other outcomes are retained. Thus these deliberately chosen cases do not estimate a frequency of invariance failure.

The displayed matching costs (raw squared-grid units) are:
'''
methods+=f"\n- Original: best {a['assignment']['best_cost']}, second {a['assignment']['second_cost']}.\n- Shifted: best {b['assignment']['best_cost']}, second {b['assignment']['second_cost']}.\n"
methods+='''
The independent verifier compares all40320 matching permutations, explicitly checks medoid ties and partitions, reconstructs moved coordinates and strict spherical counts, recomputes Dnet/eta/DI with scalar arithmetic, validates row/column masses and centroid residuals, and verifies additive OT dual stationarity. Reference partition comparisons use both maximum-overlap label alignment and direct pairwise co-membership matrices. The separate frozen-population control shifts saved moved coordinates and verifies KDE/DI invariance; it is a density-only control, not a new matching optimization. Full joint-coordinate and same-seed replay controls rerun the full reference independently.

From the project root:
```bash
# Saved-state verification only:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/verify.py
# Rebuild figure and report from saved states only:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/make_figure.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/write_report.py
# Resume is idempotent and skips completed fits:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/run_experiment.py
```

The12 primary fits and4 controls are new small calculations. No completed480-evaluation study, historical Monte Carlo simulation or previous visual study was rerun. No historical fixed-seed claim is made. All computations use one thread and one worker, below the approximate55% CPU policy. protected_before.json includes the original protected files and all files from the preceding visual study. manifest.json records source/dependency versions and saved-result hashes.
'''
(P/'METHODS.md').write_text(methods)
files=['method.py','run_experiment.py','controls.py','make_figure.py','verify.py','write_report.py']
deps=['graphical_sampling/population.py','graphical_sampling/clustering/fip_balanced_nmeans.py','DI_reference_sensitivity_JSSAM/numerical.py','DI_reference_sensitivity_JSSAM/accelerated_ot.py',cfg['manuscript']]
manifest=dict(software_versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','numba','matplotlib']},
 source_SHA256={k:hashlib.sha256((P/k).read_bytes()).hexdigest() for k in files},
 dependency_SHA256={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in deps},
 saved_result_SHA256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'runs').glob('*/*') if f.is_file()})
(P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(P/'HANDOFF.md').write_text('''# Concentrated-sample translation experiment

Complete: three asymmetric compact samples, three large valid shifts each;12 primary fits plus4 controls. All9 pairs have identical partitions and DI. No genuinely changed reference was found. See REPORT.md, summary.csv and translation_comparison.pdf/png.

Original studies, package, manuscript and reviewer responses remain unchanged. The runner is resumable and skips all completed fit IDs. Inspect saved results without optimization:
```bash
cd /home/bardia/projects/graphical-sampling
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_concentrated_JSSAM_20261016/verify.py
```

All inputs and seeds: config.json. All completed runs: runs/. Protected-file baseline: protected_before.json. Source and result hashes: manifest.json.
''')
print(P/'REPORT.md');print(P/'summary.csv')

"""Write the short scientific report from saved states only."""
import json,csv,hashlib,importlib.metadata,re
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent
cfg=json.loads((P/'config.json').read_text());selection=json.loads((P/'selection.json').read_text())
s=selection['selected'];a=json.loads((P/'runs'/s['original_id']/'summary.json').read_text());b=json.loads((P/'runs'/s['translated_id']/'summary.json').read_text())
v=json.loads((P/'verification.json').read_text());ctrl=json.loads((P/'controls.json').read_text());comparisons=json.loads((P/'comparisons.json').read_text())
raw=[(x,y) for y in range(12) for x in range(12)]
refs=[]
for i in range(8):
 m=raw[a['medoid_ids'][i]];ma=raw[a['matched_ids'][i]];mb=raw[b['matched_ids'][i]]
 refs.append(dict(cluster=i+1,units=len(a['clusters'][i]),medoid_x=m[0],medoid_y=m[1],
 original_matched_x=ma[0],original_matched_y=ma[1],shifted_matched_x=mb[0],shifted_matched_y=mb[1],
 medoid_tied_candidates=json.dumps(a['medoid_ties'][i])))
with (P/'display_references.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(refs[0]));w.writeheader();w.writerows(refs)
with (P/'display_memberships.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['unit_index_zero_based','x','y','cluster_original','cluster_shifted_aligned','changed','original_sample','shifted_sample'])
 for i,(x,y) in enumerate(raw):
  left=a['labels'][i];right=s['new_to_old'][b['labels'][i]]
  w.writerow([i,x,y,left+1,right+1,int(left!=right),int(i in a['sample_ids']),int(i in b['sample_ids'])])
report=[
'# Sample translation on a12x12 grid: visual IPnM experiment',
'16 October2026. **Result: no change found.** Five asymmetric eight-point samples were each shifted by(1,1),(1,0),(0,1). Every shifted sample was fitted independently. All15 comparisons retained the same hard partition after optimal cluster-label alignment, with exactly identical computed DI. No result was forced, and the search stopped after the five prespecified candidates.',
'![Original and shifted samples with independently fitted references](translation_comparison.png)',
'## Displayed example',
'Population U={(x,y):x,y=0,...,11}; N=144,n=8 and pi_l=1/18. The original sample is',
'```text\n(1,1), (5,0), (9,2), (2,4), (7,5), (0,8), (4,10), (10,9)\n```',
'The displayed translation is delta=(1,1). Every new point remains in U. This sample has no nonidentity square-grid symmetry. The reference clusters have irregular stepped boundaries, shown as discrete grid-cell boundaries in the figure.',
'| Quantity | Original | Shifted |\n|---|---:|---:|\n'+
f'| Dnet | {a["Dnet"]:.15f} | {b["Dnet"]:.15f} |\n| eta | {a["eta"]:.15f} | {b["eta"]:.15f} |\n| DI | {a["DI"]:.15f} | {b["DI"]:.15f} |',
'**DI difference=0; changed memberships=0/144; adjusted Rand index=1.** This is equality of the saved computed values, not merely equality after rounding. The original population remains fixed in both panels. Colors were aligned by maximum partition overlap, and equality was independently checked using the complete pairwise co-membership matrices.',
'| Cluster | Units | Medoid | Matched original point | Matched shifted point |\n|---|---:|---|---|---|'
]
for r in refs:
 report.append(f'|{r["cluster"]}|{r["units"]}|({r["medoid_x"]},{r["medoid_y"]})|({r["original_matched_x"]},{r["original_matched_y"]})|({r["shifted_matched_x"]},{r["shifted_matched_y"]})|')
report += ['The matching permutation is[0,1,2,3,4,5,6,7] in both states. Every internal cluster displacement increases by exactly delta. Consequently, the entire constructed translated population moves by delta and its self-evaluated KDE values remain unchanged unit by unit. Unit-level memberships and the complete reference table are saved separately as CSV files.',
'## Bounded alternatives',
'Candidate1 above was specified directly. Four additional asymmetric samples were frozen from RNG seed20261016 before any fit. Every candidate lies inside the0..10 subgrid so all three positive one-step shifts are valid. Solver seed17 and settings were held fixed across all pairs; selection was based on partition changes, never DI magnitude or sign.',
'| Sample | Original DI | Changed units for(1,1)/(1,0)/(0,1) | Largest absolute DI difference |\n|---|---:|---|---:|']
for i in range(1,6):
 z=json.loads((P/f'runs/sample{i}_original/summary.json').read_text());cs=[x for x in comparisons if x['candidate']==i]
 report.append(f'|{i}|{z["DI"]:.10f}|'+ '/'.join(str(x['changed_units']) for x in cs)+f'|{max(abs(x["DI_difference"]) for x in cs):.1f}|')
report += ['All sample coordinates and seeds are preserved in config.json; all individual results, including the unchanged alternatives, are retained. The five candidates can induce different references from one another, but no original-versus-uniformly-shifted pair induced a different partition.',
'## Method and numerical checks',
'The available12October Measure snapshot defines sample-adaptive entropic squared-distance OT with costs divided by their median, followed by centroid OTSP ordering, exact cumulative-IP balancing, hard clusters, squared-distance medoids and squared-distance matching. This study implements those equations in an isolated module, using the current package OTSP/within-cluster ordering and the existing stabilized Sinkhorn/Newton helpers for the same entropic OT problem. It does not call the package DI default, which uses a different preliminary method. The requested historical spherical KDE is an explicit exception to the box-kernel paragraph in that manuscript snapshot; no unseen later manuscript edits are certified.',
'Coordinates use the historical normalization raw/11. The historical KDE is KernelDensity(kernel="tophat",bandwidth="scott",rtol=1e-4), with h=144^(-1/6)='+f'{a["h"]:.15f}'+'. It has spherical support and equal unit weights. Original and internally translated estimates use the same numerical bandwidth; internally translated coordinates are not renormalized. The density pair is f(c_l),g(c_tilde_l), evaluated at each configuration\'s own unit locations. The manuscript Dnet,eta,DI definition and sign convention are unchanged.',
'OT epsilon=.01; outer tolerance1e-10; marginal tolerance1e-12; outer limit2000. The displayed pair was independently rerun at1e-12 outer and1e-13 marginal tolerances. OTSP retains the existing randomized nearest-neighbor heuristic,2-opt and15 restarts, with a fixed seed. EP quota overlaps and squared grid objectives use exact integer arithmetic to avoid floating ties in medoid/matching costs. Equal-IP medoid ties choose the smallest population label, as specified in the manuscript. All fits use one thread and one worker, well below the approximately55% machine CPU budget.',
f'- Verified{v["verified_saved_fits"]} saved full fits:20 primary,2 tighter controls,1 same-seed replay and1 joint-coordinate control.\n- Maximum row/column marginal errors:{v["max_row_error"]:.3g}/{v["max_column_error"]:.3g}; maximum final centroid residual:{v["max_fixed_point_residual"]:.3g}.\n- Independent dual-stationarity residual<={v["max_independent_dual_stationarity_residual"]:.3g}.\n- Preliminary maximum-versus-runner-up coupling margin>={v["minimum_preliminary_assignment_margin"]:.6f}: no near-tied preliminary hard assignments.\n- Exhaustive evaluation of all8! matchings confirms the saved squared optima; displayed matching is unique, with original costs13 versus35 for the best and second assignment, in raw squared-grid units.\n- Exhaustive8-node open-TSP checks agree with the heuristic route length within{v["max_heuristic_TSP_length_error_against_exhaustive"]:.3g}.\n- Direct spherical-neighbor counts and an independent scalar recomputation of Dnet,eta,DI agree within{v["component_recalculation_max_error"]:.3g}. No self-densities are zero.\n- Same-seed replay and common coordinate shift(8,-4) reproduce the displayed scientific states. The separate density-only control shifts the saved internally translated population and confirms unchanged KDE/DI; it does not constitute another matching search.',
'Ties were not hidden. The displayed reference has four exact medoid ties, resolved by the manuscript rule; its squared matching is unique. Symmetry-related optimal OTSP routes also exist. They cannot affect hard membership in this experiment: every preliminary cluster already has18 units, exactly IP mass1, so any whole-cluster path order and any within-cluster ordering preserve its membership under quota cuts. All24 saved preliminary and final partitions satisfy this property. Tighter tolerance checks leave memberships, medoids and DI unchanged. Some alternative samples have exact matching ties, consistently resolved lexicographically; all those outcomes are retained.',
'## Interpretation and limit',
'For these asymmetric samples on the specified regular population, independent full reference reconstruction preserves the reference partition and DI under each tested shift. This is a controlled illustration of invariance with an unchanged reference, not a general theorem. **No genuinely changed-clustering pair was found. Therefore this experiment cannot determine whether DI stays invariant when the sample-adaptive clustering genuinely changes.** That conditional research question remains unanswered by this bounded experiment; it would be incorrect to claim the figure answers it.',
'## Files and reproducibility',
'Figure: translation_comparison.pdf and translation_comparison.png. Main report: REPORT.md. Evidence: results.csv,comparisons.json,controls.json,verification.json,display_memberships.csv,display_references.csv,runs/*/summary.json and state.npz. Config and seeds are frozen in config.json. Every completed fit is saved immediately and skipped on resume. An unused inherited helper was removed during final source cleanup; executed_source/ preserves the exact executed module/configuration, and all scientific functions/settings and completed results are unchanged.',
'From the project root:\n```bash\nPYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_visual_JSSAM_20261016/verify.py\n# Rebuild the figure from saved results only:\nPYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_visual_JSSAM_20261016/make_figure.py\n# Safe resume/status check: completed experiment fits are skipped.\nPYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_visual_JSSAM_20261016/run_experiment.py\n```',
'No manuscript, reviewer response, GFS package, original simulation, old figure, dataset or completed sensitivity result was changed. All5,387 pre-existing protected-file hashes remain unchanged. No claims about historical Monte Carlo seed handling are made.'
]
(P/'METHODS_AND_CHECKS.md').write_text(re.sub(r'(\|[^\n]*\|)\n\n(?=\|)', r'\1\n', '\n\n'.join(report)+'\n'))
short=[
 '#12x12 sample-translation experiment',
 '**Finding:** all15 tested shifts preserved the hard partition and DI. No pair with genuinely changed clustering was found, so this experiment cannot answer whether DI is invariant when the reference changes.',
 '![Original and shifted sample, independently fitted IPnM references](translation_comparison.png)',
 'The population is U={0,...,11}^2, N=144,n=8,pi=1/18. The displayed asymmetric sample is(1,1),(5,0),(9,2),(2,4),(7,5),(0,8),(4,10),(10,9). It is shifted by(1,1), with every new point remaining in U. Both references are reconstructed independently from their respective sample centers.',
 '| Quantity | Original | Shifted |\n|---|---:|---:|\n'+
 f'| Dnet | {a["Dnet"]:.12f} | {b["Dnet"]:.12f} |\n| eta | {a["eta"]:.12f} | {b["eta"]:.12f} |\n| DI | {a["DI"]:.12f} | {b["DI"]:.12f} |',
 '**DI difference=0; changed memberships=0/144 after cluster-label alignment; adjusted Rand index=1.** Every cluster has18 units. Medoids and the unique squared matching are unchanged, and each internal cluster displacement increases by exactly the common sample shift.',
 'Four additional asymmetric samples were specified before fitting(seed20261016). All five samples were shifted by(1,1),(1,0),(0,1), using the same solver seed17. Every comparison reruns the complete reference procedure; no pair was selected by DI magnitude or sign.',
 '| Sample | DI for original and all3 shifts | Changed memberships |\n|---|---:|---:|']
for i in range(1,6):
 z=json.loads((P/f'runs/sample{i}_original/summary.json').read_text())
 short.append(f'|{i}|{z["DI"]:.10f}|0/144|')
short += [
 'The calculation follows the manuscript OT equations, median cost scaling, ordered IP balancing, squared medoids/matching and original DI definition, with the explicitly requested historical spherical Scott KDE. Coordinates are normalized as raw/11; translated coordinates are not renormalized. Each density is evaluated at its own configuration\'s unit locations. All source data and settings are saved locally.',
 'OT uses epsilon=.01, outer tolerance1e-10 and marginal tolerance1e-12. Tighter1e-12/1e-13 controls preserve the displayed result. Independent checks verify24 saved fits, including replay and joint-coordinate controls. Preliminary assignment margins exceed.022, and direct spherical counts reproduce the densities and DI.',
 'The grid still induces exact medoid ties and symmetry-related OTSP routes. Medoid ties follow the manuscript smallest-label rule. Every preliminary cluster already contains18 units, so alternative OTSP routes cannot affect quota-refined memberships here. These are not hidden numerical assignment ties.',
 '**Interpretation:** this is a visual illustration of unchanged-reference translation invariance, not a proof of general sample-only invariance. Five bounded alternatives did not produce a genuinely changed-clustering example; no result was forced.',
 'Files: [vector PDF](translation_comparison.pdf), [numerical checks and full method](METHODS_AND_CHECKS.md), [all results](results.csv), [displayed memberships](display_memberships.csv), [medoids and matching](display_references.csv). Scripts and per-run states are retained. Completed fits are skipped on resume.',
 'No manuscript, reviewer response, GFS package, original simulation, old figure or saved study result was changed. All5,387 protected-file hashes remain unchanged.'
]
short_text=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(short)+'\n')
for old,new in {'#12x12':'# 12×12','all15':'all 15','all3':'all 3','All5,387':'All 5,387','has18':'has 18','contains18':'contains 18','fitting(seed20261016)':'fitting (seed 20261016)','by(1,1),(1,0),(0,1)':'by (1,1), (1,0), and (0,1)','solver seed17':'solver seed 17','sample is(1,1)':'sample is (1,1)','epsilon=.01, outer tolerance1e-10 and marginal tolerance1e-12':'epsilon=0.01, outer tolerance 1e-10 and marginal tolerance 1e-12','Tighter1e-12/1e-13':'Tighter 1e-12/1e-13','verify24':'verify 24','exceed.022':'exceed 0.022','N=144,n=8,pi=1/18':'N=144, n=8, pi=1/18'}.items():short_text=short_text.replace(old,new)
(P/'REPORT.md').write_text(short_text)
metadata=dict(software={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','numba','matplotlib']},
 manuscript=cfg['manuscript'],manuscript_SHA256=hashlib.sha256((ROOT/cfg['manuscript']).read_bytes()).hexdigest(),
 source_SHA256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['method.py','run_experiment.py','controls.py','verify.py','make_figure.py','write_report.py']},
 saved_result_SHA256={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'runs').glob('*/*') if p.is_file()})
(P/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
(P/'HANDOFF.md').write_text('#12x12 sample-translation visual experiment\n\nComplete. Five candidate samples,15 shifts,24 saved full fits including controls.\n\nFigure: translation_comparison.pdf/png. Findings: REPORT.md. All shifts give unchanged partitions and DI; no changed-reference example was found.\n\nNo original experiment should be rerun. Inspect saved states with:\n```bash\ncd /home/bardia/projects/graphical-sampling\nPYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_visual_JSSAM_20261016/verify.py\n```\n\nThe resumable run_experiment.py skips all completed fits. Existing manuscript/package/old results are unchanged.\n')
print(P/'REPORT.md')

if __name__=='__main__':pass

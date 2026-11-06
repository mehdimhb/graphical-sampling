"""Render the verified saved states; never fits a reference or reruns OTSP."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/jssam-comment5-equal-unequal/mpl'
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment

OUT=Path(__file__).resolve().parent
cfg=json.loads((OUT/'config.json').read_text())
verification=json.loads((OUT/'verification.json').read_text());assert verification['all_checks_passed']
comparisons=json.loads((OUT/'comparisons.json').read_text())
raw=np.array([(x,y) for y in range(12) for x in range(12)])
colors=plt.get_cmap('tab10')(np.arange(8))


def load(name):return json.loads((OUT/'runs'/name/'summary.json').read_text())


def aligned_map(a,b):
    A=np.asarray(a['labels']);B=np.asarray(b['labels'])
    table=np.array([[sum((A==i)&(B==j)) for j in range(8)] for i in range(8)])
    r,c=linear_sum_assignment(-table);mapping=np.empty(8,int);mapping[c]=r
    return mapping


fig,axes=plt.subplots(2,4,figsize=(23,13),sharex=True,sharey=True)
records=[];reference_rows=[];unit_rows=[]
for rowindex,row in enumerate(cfg['rows']):
    zlist=[load(f"{row['id']}_{kind}_{where}") for kind in ['equal','unequal'] for where in ['original','shifted']]
    mappings=[np.arange(8),aligned_map(zlist[0],zlist[1]),aligned_map(zlist[0],zlist[2])]
    mappings.append(mappings[2][aligned_map(zlist[2],zlist[3])])
    for col,(ax,z,mapping) in enumerate(zip(axes[rowindex],zlist,mappings)):
        lab=np.array(z['labels']);shown=mapping[lab];grid=shown.reshape(12,12)
        ax.pcolormesh(np.arange(-.5,12.5),np.arange(-.5,12.5),grid,cmap=ListedColormap(colors),
            vmin=0,vmax=7,alpha=.115,shading='flat',rasterized=False)
        pi=np.array(z['pi'])
        ax.scatter(raw[:,0],raw[:,1],s=750*pi,c=colors[shown],linewidths=0,zorder=2)
        boundaries=[]
        for y in range(12):
            for x in range(12):
                if x<11 and grid[y,x]!=grid[y,x+1]:boundaries.append([(x+.5,y-.5),(x+.5,y+.5)])
                if y<11 and grid[y,x]!=grid[y+1,x]:boundaries.append([(x-.5,y+.5),(x+.5,y+.5)])
        ax.add_collection(LineCollection(boundaries,colors='#747474',linewidths=.7,alpha=.6,zorder=3))
        med=raw[z['medoid_ids']];matched=raw[z['matched_ids']];sample=np.array(z['sample'])
        for m,t in zip(med,matched):ax.plot([m[0],t[0]],[m[1],t[1]],color='#555',lw=.85,ls=(0,(3,2)),alpha=.8,zorder=4)
        if col%2:
            old=np.array(zlist[col-1]['sample'])
            ax.scatter(old[:,0],old[:,1],s=133,facecolors='none',edgecolors='#777',linewidths=1,alpha=.65,zorder=4)
            a=old.mean(0);b=sample.mean(0)
            ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#a32626',lw=1.7,alpha=.85,shrinkA=2,shrinkB=2),zorder=5)
        ax.scatter(sample[:,0],sample[:,1],s=145,facecolors='none',edgecolors='#be2525',linewidths=2.2,zorder=6)
        ax.scatter(med[:,0],med[:,1],s=75,color='black',marker='x',linewidths=1.9,zorder=7)
        kind='Equal FIPs' if col<2 else 'Unequal FIPs'
        title=f"({chr(97+4*rowindex+col)}) {kind} · {'shifted' if col%2 else 'original'}"
        subtitle=f"DI = {z['DI']:+.9f}"
        if col%2:
            c=next(c for c in comparisons if c['row']==row['id'] and c['FIP']==z['kind'])
            subtitle+=f"\nΔDI = {c['DI_difference']:+.9f}; changed {c['changed_memberships']}/144"
        else:subtitle+='\n'
        subtitle+=f"\nSample occupies {z['sample_clusters_occupied']}/8 clusters"
        ax.set_title(title+'\n'+subtitle,fontsize=12.6,pad=11,linespacing=1.25)
        ax.set_aspect('equal');ax.set_xlim(-.6,11.6);ax.set_ylim(-.6,11.6)
        ax.set_xticks(range(0,12,2));ax.set_yticks(range(0,12,2));ax.tick_params(labelsize=10)
        if rowindex==1:ax.set_xlabel('Grid x',fontsize=12)
        if col==0:ax.set_ylabel(('Dispersed sample' if rowindex==0 else 'Concentrated sample')+'\nGrid y',fontsize=14)
        for k,m in enumerate(z['medoid_ids']):
            reference_rows.append(dict(panel=chr(97+4*rowindex+col),run_id=z['run_id'],cluster=k,display_cluster=int(mapping[k]),
                medoid_id=m,medoid_x=int(raw[m,0]),medoid_y=int(raw[m,1]),matched_id=z['matched_ids'][k],
                matched_x=int(matched[k,0]),matched_y=int(matched[k,1]),
                cluster_size=len(z['clusters'][k]),hard_cluster_IP_total=z['hard_cluster_IP_totals'][k]))
        for i,(x,y) in enumerate(raw):
            unit_rows.append(dict(run_id=z['run_id'],unit_id=i,x=int(x),y=int(y),pi=z['pi'][i],
                cluster=z['labels'][i],display_cluster=int(shown[i]),selected=i in z['sample_ids'],
                translated_x=z['moved'][i][0],translated_y=z['moved'][i][1],original_density=z['f'][i],translated_density=z['g'][i]))
    for kind in ['equal','unequal']:
        c=next(c for c in comparisons if c['row']==row['id'] and c['FIP']==kind)
        a=load(c['original_id']);b=load(c['shifted_id'])
        records.append(dict(sample_description=row['id'],FIP=kind,shift_x=row['shift'][0],shift_y=row['shift'][1],
            DI_before=a['DI'],DI_after=b['DI'],DI_difference=c['DI_difference'],changed_memberships=c['changed_memberships'],
            ARI=c['ARI'],Dnet_before=a['Dnet'],Dnet_after=b['Dnet'],eta_before=a['eta'],eta_after=b['eta'],
            occupied_clusters_before=a['sample_clusters_occupied'],occupied_clusters_after=b['sample_clusters_occupied'],
            same_density_counts=c['same_translated_density_counts'],
            fixed_reference_DI_difference=c['fixed_reference_DI_difference']))

handles=[Line2D([],[],color='none',marker='o',markerfacecolor='#666',markeredgecolor='none',markersize=7,label='Population unit; area ∝ FIP'),
    Line2D([],[],color='none',marker='o',markerfacecolor='none',markeredgecolor='#be2525',markeredgewidth=2,markersize=10,label='Selected sample unit'),
    Line2D([],[],color='none',marker='x',markeredgecolor='black',markeredgewidth=2,markersize=8,label='Medoid'),
    Line2D([],[],color='#555',lw=1,ls=(0,(3,2)),label='Medoid–sample match'),
    Line2D([],[],color='none',marker='o',markerfacecolor='none',markeredgecolor='#777',markersize=9,label='Sample before shift')]
fig.suptitle('Does shifting the sample change DI? Equal versus unequal inclusion probabilities',fontsize=21,y=.985)
fig.text(.5,.943,r'$12\times12$ fixed population · $N=144$, $n=8$ · all eight references independently fitted',ha='center',fontsize=13)
fig.text(.5,.909,'Top row: shift (1, 1)     |     Bottom row: large shift (9, 8)     |     FIPs stay attached to population units',ha='center',fontsize=12,color='#333')
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=5,frameon=False,fontsize=11.5,columnspacing=1.5)
fig.text(.5,.04,'Equal FIPs: π = 1/18. Unequal FIPs: 0.0234375–0.0869141, summing to 8.\nColors show hard clusters and are aligned by overlap; unchanged colors are an observed result, not imposed reuse of the reference.',
    ha='center',fontsize=11,color='#333',linespacing=1.6)
fig.subplots_adjust(left=.055,right=.99,bottom=.15,top=.83,wspace=.15,hspace=.36)
fig.savefig(OUT/'equal_unequal_translation.png',dpi=230,bbox_inches='tight',facecolor='white')
fig.savefig(OUT/'equal_unequal_translation.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
for filename,rows in [('summary.csv',records),('display_references.csv',reference_rows),('display_units.csv',unit_rows)]:
    with (OUT/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

lines=['# Grid sample translation: equal versus unequal inclusion probabilities','',
    '**Finding:** DI is unchanged under all four illustrated sample translations. Each original and shifted sample received a new, independent complete IPnM fit. All four hard partitions are unchanged after label alignment; this experiment does not demonstrate invariance when the reference itself changes.','',
    '![Eight independently fitted panels](equal_unequal_translation.png)','',
    '| Sample | FIPs | Shift | DI before | DI after | Difference | Changed memberships |',
    '| --- | --- | --- | ---: | ---: | ---: | ---: |']
for r in records:
    lines.append(f"| {r['sample_description']} | {r['FIP']} | ({r['shift_x']},{r['shift_y']}) | {r['DI_before']:.12f} | {r['DI_after']:.12f} | {r['DI_difference']:.12f} | {r['changed_memberships']}/144 |")
lines+=['',
    'The first row reuses the **sample geometry** from the earlier dispersed example: `(1,1),(5,0),(9,2),(2,4),(7,5),(0,8),(4,10),(10,9)`, shifted by `(1,1)`. Its equal-FIP original and shifted samples occupy all eight reference clusters. With unequal FIPs, the original occupies seven clusters and the shifted sample eight.',
    '',
    'The second row uses the earlier compact asymmetric shape: `(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,3)`, shifted by `(9,8)` into the top-right corner. The original occupies one cluster; the shifted sample occupies two, under both FIP settings. These occupancy counts describe where the sample falls in the population partition; they do not mean that the partition changed.',
    '',
    '## Inclusion probabilities',
    '',
    'The population is `{0,...,11}²`, with 144 units and sample size eight. Equal FIPs are `1/18`. A **single unequal vector** is frozen for both sample shapes and all their shifts. Seed 20261016 generates integer weights in `[24,89]`, followed by a deterministic sum correction to 8192; `pi_i=w_i/1024`. The resulting FIPs range from 0.0234375 to 0.0869140625, sum **exactly to eight** in binary arithmetic, and all lie strictly between zero and one. The construction was fixed before any fit and was not selected to produce invariance.',
    '',
    '`population_FIPs.csv` gives all 144 coordinates, equal/unequal FIPs and integer weights. Point area in the main figure is proportional to FIP on a common scale. FIPs remain attached to fixed population units: a moved sample point acquires the FIP of its new unit. The illustrated samples are deliberately chosen configurations, not random draws whose selection frequencies estimate these FIPs.',
    '',
    '## Procedure and numerical controls',
    '',
    'All **eight primary references were fitted anew**, including the four equal-FIP panels; no prior partition was reused. Starting centers are the eight observed sample coordinates. The isolated manuscript-style method uses sample-initialized entropic OT (epsilon 0.01, squared costs divided by their median), seeded open-TSP ordering with 15 restarts, the existing stable within-cluster order, cumulative-IP border splitting, hard assignment to the larger share, weighted **squared-distance** medoids, and optimal **squared-distance** matching. Exact medoid ties select highest FIP, then smallest population ID. Exact hard-share ties select the smallest persistent cluster index; none occurs in these fits.',
    '',
    'Equal-FIP overlap is computed exactly using weights 1/18. Unequal-FIP overlap is computed exactly using integer numerators and denominator 1024. The soft clusters have IP mass exactly one; hard cluster totals can differ from one after border hardening. These totals, medoids and matching are saved in `display_references.csv`. Unequal FIPs affect clustering, quota refinement and weighted medoids; they are **not used as KDE weights**.',
    '',
    'Density calculations use the historical unweighted spherical top-hat KDE and scalar Scott bandwidth `h=144^(-1/6)` on population-normalized coordinates `raw/11`. The same numerical bandwidth is used before and after internal cluster translations, with no renormalization of the internally translated configuration. Each density is evaluated at its corresponding unit locations. The existing Dnet, eta and DI definitions and signs are unchanged.',
    '',
    'Primary outer/marginal tolerances are 1e-10/1e-12. All four unequal-FIP panels were independently refitted at 1e-12/1e-13; hard partitions, medoids, matching, density counts and DI remain unchanged. A same-seed unequal-FIP replay and a full joint-coordinate translation control also agree. Total: eight primary fits plus six controls, one thread and one worker. No original Monte Carlo or completed 480-run sensitivity experiment was rerun.',
    '',
    'Independent checks verified all 14 saved states: exact rational IP overlaps; exhaustive 8! squared matchings and open paths; weighted medoid objectives/ties; OT marginal, fixed-point and dual-stationarity residuals; label-independent co-memberships; scalar spherical-neighbor counts; and Dnet/eta/DI. Every saved TSP path attains the exhaustive shortest length for its eight saved centroids. Preliminary OT assignment margins exceed '+f"{verification['minimum_preliminary_assignment_gap']:.6g}"+'. Unequal-FIP medoids and hard shares have no exact ties. The equal grid retains some exact medoid/route ties, resolved by the recorded conventions; the four equal-panel results agree with their earlier saved counterparts.',
    '',
    'There are no zero self-evaluated densities. The top-hat estimator is discontinuous at its support boundary, but every original and shifted pair has identical density counts and both components, so equality is not merely rounding or saturation near DI=-1. In each pair the internally translated populations differ by exactly the common sample shift. The frozen-reference control also gives zero DI difference.',
    '',
    '## Interpretation',
    '',
    'Unequal inclusion probabilities change the reference configurations and the absolute DI values relative to equal probabilities. They **do not break sample-translation invariance in these two fixed examples**. The dispersed sample remains positive; the concentrated sample remains strongly negative. None of the four shifts changes the induced hard partition, so these examples cannot settle the harder question of what happens when IPnM selects a genuinely different reference. They are deterministic illustrations, not estimates of the prevalence of invariance or a universal proof.',
    '',
    'For a fixed reference and matching, shifting every selected point by delta adds delta to every internally translated population point. A distance-based KDE evaluated at the corresponding translated positions is unchanged; hence Dnet, eta and DI are unchanged. Squared matching has the same minimizing permutation set under this sample shift, because the change in its total cost is independent of the permutation. These conditions are satisfied by the observed reconstructed references here. Refitting IPnM does not in general guarantee those conditions for every population or initialization.',
    '',
    '## Files and reproducibility',
    '',
    '- `equal_unequal_translation.pdf/png`: requested 2-row × 4-column figure.',
    '- `summary.csv`: four primary before/after comparisons, components and membership changes.',
    '- `population_FIPs.csv`: frozen probabilities for all population units.',
    '- `display_units.csv`, `display_references.csv`: all primary hard memberships, translated coordinates, densities, medoids and matches.',
    '- `runs/`: complete summaries and arrays for all 14 fits; `results.csv`, `progress.json`, `run.log`, `verification.json` and source scripts preserve the work.',
    '',
    'All '+str(verification['protected_files_unchanged'])+' protected original/source/result files retain their hashes. Existing package, manuscript, reviewer responses, datasets, old figures and studies were not edited. The current Git branch was not changed or pushed by this experiment.',
    '',
    'From the project root (completed fits are skipped):','',
    '```bash',
    'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_equal_unequal_JSSAM_20261016/run_experiment.py',
    'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_equal_unequal_JSSAM_20261016/verify.py',
    'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_equal_unequal_JSSAM_20261016/make_outputs.py',
    '```','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'equal_unequal_translation.png');print(OUT/'REPORT.md')

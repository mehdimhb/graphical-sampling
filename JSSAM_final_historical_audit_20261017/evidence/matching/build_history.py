"""Build a reviewable source timeline without executing scorer/notebook code."""
from pathlib import Path
import json,csv,subprocess,re
D=Path(__file__).resolve().parent
manifest=json.loads((D/'manifest.json').read_text());rows=[]
def git(*a):return subprocess.check_output(['git',*a],text=True)
for e in manifest:
 p=e['path'];ref=e['ref'];raw=git('show',ref+':'+p)
 if p.endswith('.py'):cells=[{'source':[raw]}]
 else:cells=json.loads(raw)['cells']
 for ci,c in enumerate(cells):
  s=''.join(c.get('source',[]));ls=s.splitlines()
  if not (p.endswith('.py') or 'class DensityDisparityFlexible' in s or (re.search(r'linear_sum_assignment\s*\(',s) and 'cdist' in s)):
   continue
  anch=[]
  for i,l in enumerate(ls,1):
   if any(x in l for x in ['init_clust_method','cost = cdist','linear_sum_assignment(','D = cdist','objective = D @ w']):anch.append(f'C{ci}:L{i:04d} {l.strip()}')
  if p.endswith('.py'):
   assoc='Source history only; package presence alone does not attribute results.'
  elif 'measure_calculation_nmedoids_clean' in p:
   assoc='106 saved MC rows associated with source: 104 exact legacy means+SDs; 2 mean discrepancies. 32 MC and 24 design rows excluded from scorer attribution.'
  elif '/01_measure' in p:
   assoc='Retained small test: cell15 execution9, 32 parsed EP result rows, Iterations10, zero exact legacy mean+SD matches; cell17 2000-run block has no execution/output.'
  elif '/handy_jupyters/' in p or p.endswith('/measure_calculation.ipynb'):
   assoc='Renamed/archived scorer source; independent row-level attribution not asserted.'
  else:assoc='Later illustration/comparison source; not provenance of legacy 162-row main table.'
  method=re.findall(r'init_clust_method\s*=\s*[\'\"](\w+)',s)
  if 'IPBalancedOTNMeans(' in s:method.append('IPBalancedOTNMeans direct class')
  rows.append(dict(record_type='package_module' if p.endswith('.py') else 'notebook_matching_override',git_commit=ref,ancestor_of_audited_current=subprocess.run(['git','merge-base','--is-ancestor',ref,'af36c9e3ff36f957243188ade01a1e01400f3709'],stdout=subprocess.DEVNULL).returncode==0,commit_datetime=git('show','-s','--format=%ai',ref).strip(),path=p,cell=ci if p.endswith('.ipynb') else '',execution_count=c.get('execution_count'),retained_outputs=len(c.get('outputs',[])),preliminary_method=';'.join(method),medoid_objective='weighted Euclidean sum' if '_cluster_medoids' in s or 'weighted_medoids' in s else 'nmeans/no medoid implementation in this block',matching_objective='Euclidean sum (cdist default; no squaring)',source_anchors=' | '.join(anch),extracted_source=e['extraction'],result_association=assoc,limitations='Saved source/output association does not pin every historical kernel binding, dependency, or import identity.'))
for e in manifest:
 if '/03_variance' in e['path'] or '/04_paper' in e['path']:
  is03='/03_' in e['path']
  rows.append(dict(record_type='notebook_no_matching_override',git_commit=e['ref'],ancestor_of_audited_current=subprocess.run(['git','merge-base','--is-ancestor',e['ref'],'af36c9e3ff36f957243188ade01a1e01400f3709'],stdout=subprocess.DEVNULL).returncode==0,commit_datetime=git('show','-s','--format=%ai',e['ref']).strip(),path=e['path'],cell='',execution_count='',retained_outputs='',preliminary_method='',medoid_objective='',matching_objective='no local override',source_anchors='No linear_sum_assignment or sqeuclidean in notebook source',extracted_source=e['extraction'],result_association='Variance-estimation/HT calculation; no DI scorer definition or DI scoring path identified.' if is03 else 'Plots existing summaries; illustrative cell12 imports graphical_sampling.index.DensityDisparity directly and scores individual samples, without local matching override.',limitations='Imports and saved plotting outputs do not by themselves establish historical scorer runtime for main table.'))
rows.sort(key=lambda r:(r['commit_datetime'],r['path'],str(r['cell'])))
with (D/'matching_history.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Preserve numbered full class source for the June5 OT sample-initialization caveat.
p='graphical_sampling/clustering/fip_balanced_nmeans.py';ref='82a7eca7d';full=git('rev-parse',ref).strip();src=git('show',ref+':'+p)
(D/'82a7eca7d__fip_balanced_nmeans.txt').write_text(f'Git {full}:{p}\n'+''.join(f'L{i:04d} {l}\n' for i,l in enumerate(src.splitlines(),1)))
print(f'{len(rows)} timeline entries')

p='graphical_sampling/clustering/ip_balanced_ot_nmeans.py';ref='247d9422e';full=git('rev-parse',ref).strip();src=git('show',ref+':'+p)
(D/'247d9422e__ip_balanced_ot_nmeans.txt').write_text(f'Git {full}:{p}\n'+''.join(f'L{i:04d} {l}\n' for i,l in enumerate(src.splitlines(),1)))

"""Read preserved Git JSON/CSV and bytes only; never execute scientific source."""
from pathlib import Path
import subprocess,json,csv,io,re,hashlib,collections
from decimal import Decimal
D=Path(__file__).resolve().parent
REV='e71d6935bfe824e72a7b722067a24775ed160610'
def git(*a):return subprocess.check_output(['git',*a])
def blob(p,r=REV):return git('show',r+':'+p)
def readcsv(raw):return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
def normalize(r):
 r=dict(r)
 r['Population']={'clust_N144_perm':'Clustered','grid_N144_perm':'Gridded','rand_N144_perm':'Random'}.get(r['Population'],r['Population'])
 r['Probability']={'Equal':'EP','Unequal':'UP'}.get(r['Probability'],r['Probability'])
 r['Method']={'MaxEnt':'MAX','Maxe':'MAX','SPC':'SCP','Scps':'SCP','Lopi':'LPV','Wave':'WAV','LPM':'LPV'}.get(r['Method'],r['Method'])
 return r
K=['Population','Probability','Method','n','N']
def key(r):return tuple(str(r[k]) for k in K)
def dicts(rows):
 out={key(normalize(r)):normalize(r) for r in rows}
 assert len(out)==len(rows)
 return out
def savedtext(o):
 t=o.get('text',o.get('data',{}).get('text/plain',''))
 return ''.join(t) if isinstance(t,list) else t
nb=json.loads(blob('simulations/jupyters/measure_calculation_nmedoids_clean.ipynb'))
printed={}
pat=re.compile(r'^\s*\d+\s+(\S+)\s+(\d+)\s+(\d+)\s+(MaxEnt|SPC|LPV|WAV)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+nmedoids\s*$',re.M)
for ci,inc in [(19,'UP'),(20,'EP')]:
 for oi,o in enumerate(nb['cells'][ci].get('outputs',[])):
  for pop,N,n,m,dm,ds,M in pat.findall(savedtext(o)):
   r=normalize(dict(Population=pop,N=N,n=n,Method=m,Dm=dm,Ds=ds,Probability=inc,Iterations=M,cell=ci,output=oi))
   if key(r) in printed:assert all(printed[key(r)][c]==r[c] for c in ['Dm','Ds','Iterations'])
   printed[key(r)]=r
# Design values are saved but scorer's import identity is not pinned.
designed={}
pat2=re.compile(r'^\s*\d+\s+(meuse|swiss)\s+(EP|UP)\s+(NMS|GMS)\s+(initial|best)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$',re.M)
for oi,o in enumerate(nb['cells'][22]['outputs']):
 for pop,pr,m,des,N,n,dm,ds in pat2.findall(savedtext(o)):
  r=dict(Population=pop,Probability=pr,Method=m,N=N,n=n,Dm=dm,Ds=ds,cell=22,output=oi)
  if key(r) in designed:assert all(designed[key(r)][c]==r[c] for c in ['Dm','Ds'])
  designed[key(r)]=r
plotinput=dicts(readcsv(blob('simulations/jupyters/results/DI_nmedoids_selected_populations.csv')))
master=dicts(readcsv(blob('simulations/results/master_summary_DI_nmedoids_updated.csv')))
clean=dicts(readcsv(blob('simulations/results/DI_nmedoids_selected_populations_clean.csv')))
current=dicts(readcsv(Path('simulations/results/All_DI_nmedoids_all_desings.csv').read_bytes()))
assert len(printed)==106 and len(designed)==24 and len(plotinput)==162 and len(clean)==138
pairs=lambda r:(Decimal(r['Dm']),Decimal(r['Ds']))
checks={'history_commit':REV,'primary_notebook':'simulations/jupyters/measure_calculation_nmedoids_clean.ipynb','legacy_all_rows':len(plotinput),'MC_rows':len(clean),'saved_MC_rows':len(printed),'design_saved_rows':len(designed),'CSV_checks':[]}
for name,rows in [('June1 updated master',master),('current All_DI table',current),('June1 clean Monte Carlo table',clean)]:
 overlap=set(rows)&set(plotinput)
 assert overlap==set(clean) if name=='June1 clean Monte Carlo table' else overlap==set(plotinput)
 assert all(pairs(rows[k])==pairs(plotinput[k]) for k in overlap)
 checks['CSV_checks'].append({'name':name,'exact_mean_SD_rows':len(overlap),'total_source_rows':len(rows)})
rows=[]
for k,r in plotinput.items():
 p=printed.get(k) or designed.get(k)
 if k in printed:status='saved_MC_exact' if pairs(r)==pairs(p) else 'saved_MC_DI_mean_discrepancy'
 elif k in designed:status='saved_design_exact_import_unresolved' if pairs(r)==pairs(p) else 'saved_design_discrepancy'
 else:status='MC_retained_output_missing'
 rows.append({**{c:r[c] for c in K},'legacy_Dm':r['Dm'],'legacy_Ds':r['Ds'],'retained_Dm':p['Dm'] if p else '', 'retained_Ds':p['Ds'] if p else '', 'Iterations':p.get('Iterations','') if p else '', 'source_cell':p['cell'] if p else '', 'source_output':p['output'] if p else '', 'status':status,'matching_provenance':'Euclidean unsquared in associated scorer source; historical kernel not immutable' if k in printed else 'unresolved imported scorer' if k in designed else 'unresolved historical execution','June1_master_exact':True,'current_All_DI_exact':True})
with (D/'row_provenance.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks['row_status_counts']=dict(collections.Counter(r['status'] for r in rows))
checks['differences']=[r for r in rows if 'discrepancy' in r['status']]
checks['unresolved_MC_groups']=dict(collections.Counter('/'.join(r[c] for c in ['Population','Probability']) for r in rows if r['status']=='MC_retained_output_missing'))
checks['MC_iterations']=sorted(set(r['Iterations'] for r in rows if r['Iterations']))
# Later OT notebook retained rows: small test, not source-only 2000-run cell.
later=json.loads(blob('simulations/jupyters/01_measure_calculation_spreadness_indices.ipynb','82a7eca7d'))
laterrows={}
for oi,o in enumerate(later['cells'][15]['outputs']):
 for pop,N,n,m,dm,ds,M in pat.findall(savedtext(o)):
  r=normalize(dict(Population=pop,N=N,n=n,Method=m,Dm=dm,Ds=ds,Probability='EP',Iterations=M))
  laterrows[key(r)]=r
checks['June5_OT_test']={'retained_cell':15,'execution_count':later['cells'][15]['execution_count'],'rows_parsed':len(laterrows),'Iterations':sorted(set(r['Iterations'] for r in laterrows.values())),'exact_mean_SD_matches_to_legacy':sum(pairs(r)==pairs(plotinput[k]) for k,r in laterrows.items()),'2000_run_cell':17,'2000_run_cell_execution_count':later['cells'][17]['execution_count'],'2000_run_cell_outputs':len(later['cells'][17]['outputs'])}
# Byte continuity for figures, not inferred current-code provenance.
checks['figure_continuity']=[]
for name in ['simulated_result_measure_medoid.pdf','meuse_balance_indices_measure_medoid.pdf','swiss_balance_indices_measure_medoid.pdf']:
 p='simulations/plots/'+name
 h=blob(p);now=Path(p).read_bytes();assert h==now
 checks['figure_continuity'].append({'path':p,'June1_SHA256':hashlib.sha256(h).hexdigest(),'current_byte_identical':True})
# Exact historical identity of later cleaned notebooks.
checks['later_notebook_blobs']=[]
for basename in ['01_measure_calculation_spreadness_indices','03_variance_estimation_NHT','04_paper_plots_and_tables']:
 p='simulations/jupyters/'+basename+'.ipynb'
 hashes={r:hashlib.sha256(blob(p,r)).hexdigest() for r in ['82a7eca7d','7c61dbeafa6808e9eafa0b1fc678a25c077109cc','af36c9e3ff36f957243188ade01a1e01400f3709']}
 checks['later_notebook_blobs'].append({'path':p,'hashes':hashes,'all_identical':len(set(hashes.values()))==1})
(D/'provenance_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k not in ['figure_continuity','later_notebook_blobs']},indent=2))

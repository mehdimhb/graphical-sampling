"""Read committed JSON/CSV/source as data only. Never import project/notebook code."""
from pathlib import Path
import csv,io,json,re,subprocess,hashlib,collections
ROOT=Path('/home/bardia/projects/graphical-sampling'); OUT=ROOT/'JSSAM_final_historical_audit_20261017/evidence/replications'; REV='e71d6935bfe824e72a7b722067a24775ed160610'
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def blob(p,rev=REV):return git('show',rev+':'+p)
def rows(raw):return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
manifest=[]
def save_original(path,tag,nb=False,rev=REV):
 raw=blob(path,rev); manifest.append(dict(commit=rev,path=path,git_blob=git('rev-parse',rev+':'+path).decode().strip(),sha256=hashlib.sha256(raw).hexdigest()))
 if nb:
  data=json.loads(raw); lines=[]
  for i,c in enumerate(data['cells']):
   lines.append(f'CELL {i} execution_count={c.get("execution_count")} type={c["cell_type"]}\n')
   lines.extend(''.join(c.get('source',[])).splitlines(keepends=True));lines.append('\nEND_SOURCE\n')
   for j,o in enumerate(c.get('outputs',[])):
    lines.append(f'OUTPUT {j} type={o.get("output_type")}\n');
    if o.get('output_type')=='error':lines.append(json.dumps({k:o.get(k) for k in ['ename','evalue','traceback']})+'\n')
    lines.append(''.join(o.get('text',o.get('data',{}).get('text/plain',[])))+'\n')
   lines.append('END_CELL\n\n')
  (OUT/tag).write_text(''.join(lines));return data
 (OUT/tag).write_bytes(raw);return raw
nb=save_original('simulations/jupyters/measure_calculation_nmedoids_clean.ipynb','historical_medoid_notebook.txt',True)
plot=save_original('simulations/jupyters/paper_plots.ipynb','historical_plot_notebook.txt',True)
old=save_original('simulations/jupyters/store/measurs.ipynb','historical_5000_notebook.txt',True)
save_original('simulations/jupyters/measurs.ipynb','initial_March15_5000_notebook.txt',True,rev='1f376e9fc1394feac1d02888b0ba0053fb48bceb')
save_original('simulations/jupyters/measurs.ipynb','May23_5000_notebook.txt',True,rev='4d056408b5167f17d4669bd8233bd16ebacc3d17')
for p in ['graphical_sampling/index/density_disparity.py','graphical_sampling/design.py']:
 try:save_original(p,'historical_'+Path(p).name)
 except subprocess.CalledProcessError:pass
clean=rows(save_original('simulations/results/DI_nmedoids_selected_populations_clean.csv','historical_MC_clean.csv'))
allrows=rows(save_original('simulations/jupyters/results/DI_nmedoids_selected_populations.csv','historical_plot_DI.csv'))
exact=rows(save_original('simulations/jupyters/results/DI_saved_designs_meuse_swiss_nmedoids.csv','historical_saved_design_DI.csv'))
master=rows(save_original('simulations/results/master_summary_DI_nmedoids_updated.csv','historical_master.csv'))
paths=git('ls-tree','-r','--name-only',REV).decode().splitlines()
(OUT/'historical_git_file_inventory.txt').write_text('\n'.join(paths)+'\n')
def norm(r):
 r={k:v.strip() if isinstance(v,str) else v for k,v in r.items()};
 for col in ['n','N']:
  if col in r:r[col]=str(int(float(r[col])))
 r['Population']={'clust_N144_perm':'Clustered','grid_N144_perm':'Gridded','grided_N144_perm':'Gridded','rand_N144_perm':'Random','clust':'Clustered','grid':'Gridded','rand':'Random'}.get(r['Population'],r['Population']);r['Probability']={'Equal':'EP','Unequal':'UP'}.get(r['Probability'],r['Probability']);r['Method']={'MaxEnt':'MAX','Maxe':'MAX','SPC':'SCP','Scps':'SCP','Lopi':'LPV','Wave':'WAV','GRZ':'GMS'}.get(r['Method'],r['Method']);return r
def key(r):return tuple(str(r[k]) for k in ['Population','Probability','Method','n','N'])
parsed={}
pat=re.compile(r'^\s*\d+\s+(\S+)\s+(\d+)\s+(\d+)\s+(MaxEnt|SPC|LPV|WAV)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+nmedoids\s*$',re.M)
for ci,prob in [(19,'UP'),(20,'EP')]:
 for o in nb['cells'][ci].get('outputs',[]):
  text=''.join(o.get('text',o.get('data',{}).get('text/plain',[])))
  for pop,N,n,method,dm,ds,M in pat.findall(text):
   r=norm(dict(Population=pop,Probability=prob,Method=method,N=N,n=n,Dm=dm,Ds=ds,Iterations=M,cell=ci));parsed[key(r)]=r
orig=[]
for p in paths:
 if re.match(r'simulations/results/(equal|unequal)_M=5000/summary_.*\.csv$',p):
  rr=rows(blob(p));manifest.append(dict(commit=REV,path=p,git_blob=git('rev-parse',REV+':'+p).decode().strip(),sha256=hashlib.sha256(blob(p)).hexdigest()))
  for source_row_line,r in enumerate(rr,2):
   r['source_row_line']=source_row_line
   m=re.match(r'summary_(clust_N144_perm|grid_N144_perm|rand_N144_perm|meuse|swiss)_n=',Path(p).name)
   if not m:continue
   r=norm(dict(r,Population=m[1],Probability='EP' if '/equal_' in p else 'UP',source_path=p))
   # paper_plots cell33 discards Maxe EP and uses Rand renamed MAX.
   if r['Probability']=='EP':
    if r['Method']=='MAX':continue
    if r['Method']=='Rand':r['Method']='MAX'
   elif r['Method']=='Rand':continue
   orig.append(r)
orig_by=collections.defaultdict(list)
for r in orig:orig_by[key(r)].append(r)
master_by={key(norm(r)):norm(r) for r in master}
exact_by={key(norm(r)):norm(r) for r in exact}
source_lines={}
cell=None
for line_no,line in enumerate((OUT/'historical_medoid_notebook.txt').read_text().splitlines(),1):
 cm=re.match(r'CELL (\d+) ',line)
 if cm:cell=int(cm[1])
 if cell in [19,20]:
  match=pat.fullmatch(line)
  if match:
   pop,N,n,method,dm,ds,M=match.groups();pr=norm(dict(Population=pop,Probability='UP' if cell==19 else 'EP',Method=method,n=n,N=N));source_lines.setdefault(key(pr),line_no)
ledger=[]
for plot_csv_line,ar in enumerate(allrows,2):
 r=norm(ar); k=key(r); p=parsed.get(k); ex=exact_by.get(k); older=orig_by.get(k,[]); mr=master_by.get(k,{})
 eq=lambda a,b,cols: all(a.get(c,'')!='' and b.get(c,'')!='' and abs(float(a[c])-float(b[c]))<1e-12 for c in cols)
 record={k:r[k] for k in ['Population','Probability','Method','n','N','Dm','Ds']}
 record.update(source_commit=REV,medoid_source_config_requested=2000 if not ex else '',setting_requested_count_in_saved_output=2000 if p else '',saved_output_Iterations=p['Iterations'] if p else '',saved_output_cell=p['cell'] if p else 22 if ex else '',saved_DI_mean_SD_match=eq(r,p,['Dm','Ds']) if p else all(round(float(ex[c]),6)==round(float(r[c]),6) for c in ['Dm','Ds']) if ex else '',completed_count_direct_observation='not independently logged',DI_statistics_used_count=ex.get('Support','') if ex else p['Iterations'] if p and eq(r,p,['Dm','Ds']) else '',DI_count_status='weighted full design support; count is support size' if ex else '2000 inferred from allocating/generating all requested samples, no drop logic, successful saved statistics' if p and eq(r,p,['Dm','Ds']) else 'saved output labelled 2000 but final mean differs; final count unresolved' if p else 'count unknown; source configuration 2000 is not setting-specific execution evidence',exact_support=ex.get('Support','') if ex else '',old_5000_source_count=len(older),old_ite_values=';'.join(sorted({a.get('ite','') for a in older})),old_source_paths=';'.join(a['source_path'] for a in older),old_source_row_lines=';'.join(str(a['source_row_line']) for a in older),plot_input_evidence='historical_plot_DI.csv:'+str(plot_csv_line),saved_output_evidence='historical_medoid_notebook.txt:'+str(source_lines[k]) if p else 'historical_saved_design_DI.csv:'+str(exact.index(next(a for a in exact if key(norm(a))==k))+2) if ex else '',saved_output_Dm=p['Dm'] if p else ex['Dm'] if ex else '',saved_output_Ds=p['Ds'] if p else ex['Ds'] if ex else '',aux_VI_LB_match_old=any(eq(mr,a,['Vm','Vs','Lm','Ls']) for a in older),aux_VI_LB_match_after_documented_clipping=any(eq(mr,dict(a,Lm=max(-1.5,min(1.5,float(a['Lm']))),Ls=max(-1.5,min(1.5,float(a['Ls'])))),['Vm','Vs','Lm','Ls']) for a in older),aux_MI_match_old=any(eq(mr,a,['Mm','Ms']) for a in older),old_DI_equals_new=any(eq(r,a,['Dm','Ds']) for a in older),stats_count_caveat='Iterations is assigned requested count, not len(samples); no retained MC raw samples/count logs located' if not ex else 'len(probs), normalized probability weighted population SD, not Monte Carlo replicates')
 ledger.append(record)
with (OUT/'per_setting_replication_ledger.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
with (OUT/'original_5000_summary_rows.csv').open('w') as f:
 fields=sorted(set().union(*(r.keys() for r in orig)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(orig)
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
summary=dict(rows=len(ledger),MC_rows=sum(not r['exact_support'] for r in ledger),MC_printed_rows=len(parsed),MC_printed_exact_matches=sum(bool(r['saved_DI_mean_SD_match']) for r in ledger if not r['exact_support']),exact_support_rows=sum(bool(r['exact_support']) for r in ledger),counts_by_group=[dict(Population=a,Probability=b,rows=len(v),printed_MC=sum(bool(r['saved_output_Iterations']) for r in v),exact_support_rows=sum(bool(r['exact_support']) for r in v)) for (a,b),v in ((g,[r for r in ledger if (r['Population'],r['Probability'])==g]) for g in sorted({(r['Population'],r['Probability']) for r in ledger}))],aux_VI_LB_matches=sum(r['aux_VI_LB_match_old'] for r in ledger),aux_VI_LB_matches_after_clipping=sum(r['aux_VI_LB_match_after_documented_clipping'] for r in ledger),aux_MI_matches=sum(r['aux_MI_match_old'] for r in ledger),new_DI_equals_old=sum(r['old_DI_equals_new'] for r in ledger),new_samples=0,notebook_code_executed=False)
(OUT/'replication_checks.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

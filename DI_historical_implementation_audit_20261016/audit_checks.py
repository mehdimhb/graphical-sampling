"""Read-only provenance and arithmetic checks. No sampling, clustering, TSP or DI evaluations."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import __future__,ast,collections,hashlib,io,json,re,subprocess,sys,zipfile,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent;EV=OUT/'evidence';REV='e71d6935b'
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def blob(path,rev=REV):return git('show',rev+':'+path)
def norm(d):
 d=d.copy();d['Population']=d.Population.replace({'clust_N144_perm':'Clustered','grid_N144_perm':'Gridded','rand_N144_perm':'Random','grided_N144_perm':'Gridded'});d['Probability']=d.Probability.replace({'Equal':'EP','Unequal':'UP'});d['Method']=d.Method.replace({'MaxEnt':'MAX','Maxe':'MAX','SPC':'SCP','Scps':'SCP','Lopi':'LPV','Wave':'WAV','LPM':'LPV'});return d
KEY=['Population','Probability','Method','n','N']
def compare(a,b,tag,require_exact=True):
 a=norm(a);b=norm(b);assert not a.duplicated(KEY).any() and not b.duplicated(KEY).any()
 m=a.merge(b,on=KEY,suffixes=('_a','_b'));err=max(abs(m.Dm_a-m.Dm_b).max(),abs(m.Ds_a-m.Ds_b).max());assert len(m)==len(a),(tag,len(a),len(m));
 if require_exact:assert err<1e-12,(tag,err)
 mismatches=m[(abs(m.Dm_a-m.Dm_b)>1e-12)|(abs(m.Ds_a-m.Ds_b)>1e-12)]
 if not require_exact:mismatches.to_csv(EV/'saved_output_transcription_differences.csv',index=False)
 return dict(comparison=tag,matched_rows=len(m),maximum_absolute_DI_mean_SD_difference=float(err),exact_matching_rows=len(m)-len(mismatches),different_rows=len(mismatches))
def main():
 EV.mkdir(exist_ok=True);results={};inputs={}
 nbpath='simulations/jupyters/measure_calculation_nmedoids_clean.ipynb';plotpath='simulations/jupyters/paper_plots.ipynb'
 old=json.loads(blob(nbpath));plot=json.loads(blob(plotpath));new=json.loads((ROOT/'simulations/jupyters/01_measure_calculation_spreadness_indices.ipynb').read_text())
 clean=pd.read_csv(io.BytesIO(blob('simulations/results/DI_nmedoids_selected_populations_clean.csv')))
 histinput=pd.read_csv(io.BytesIO(blob('simulations/jupyters/results/DI_nmedoids_selected_populations.csv')))
 master=pd.read_csv(io.BytesIO(blob('simulations/results/master_summary_DI_nmedoids_updated.csv')))
 allnow=pd.read_csv(ROOT/'simulations/results/All_DI_nmedoids_all_desings.csv')
 results['CSV_links']=[compare(clean,allnow,'June-1 clean CSV -> current All_DI'),compare(histinput,allnow,'June-1 plotting input -> current All_DI'),compare(histinput,master,'June-1 plotting input -> June-1 updated master'),compare(clean,pd.read_csv(ROOT/'simulations/Store/DI_nmedoids_selected_populations_clean.csv'),'June-1 clean CSV -> current Store copy')]
 parsed=[]
 pattern=re.compile(r'^\s*\d+\s+(\S+)\s+(\d+)\s+(\d+)\s+(MaxEnt|SPC|LPV|WAV)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+nmedoids\s*$',re.M)
 for ci,inc in [(19,'UP'),(20,'EP')]:
  for o in old['cells'][ci].get('outputs',[]):
   text=''.join(o.get('text',o.get('data',{}).get('text/plain',[])))
   for pop,N,n,method,dm,ds,M in pattern.findall(text):parsed.append(dict(Population=pop,Probability=inc,Method=method,N=int(N),n=int(n),Dm=float(dm),Ds=float(ds),Iterations=int(M),source_cell=ci))
 printed=norm(pd.DataFrame(parsed)).drop_duplicates(KEY);printed.to_csv(EV/'matched_saved_notebook_outputs.csv',index=False)
 results['saved_execution_outputs']=compare(printed,allnow,'June-1 retained notebook execution outputs -> All_DI',require_exact=False);results['saved_execution_outputs'].update(replications=sorted(printed.Iterations.unique().tolist()),by_probability=printed.Probability.value_counts().to_dict(),source_cells=[19,20],unmatched_MC_cells=len(clean)-len(printed))
 assert len(clean)==138 and len(histinput)==162 and len(printed)==106
 versions=[]
 for rev in ['d047278fc','a6102e6cc','e71d6935b','v1.0.0','HEAD']:
  source=blob('graphical_sampling/index/density_disparity.py',rev).decode();methods=re.findall(r'init_clust_method\s*=\s*[\'\"](\w+)',source);versions.append(dict(revision=rev,commit=git('rev-parse',rev).decode().strip(),preliminary_methods=methods,nmedoids_option='nmedoids' in source))
 results['package_history']=versions
 for name,notebook in [('June1',old),('June5_HEAD',new)]:
  for i,c in enumerate(notebook['cells']):
   s=''.join(c.get('source',[]))
   if 'class DensityDisparityFlexible' in s:results[name+'_scorer']=dict(cell=i,execution_count=c.get('execution_count'),methods=re.findall(r'init_clust_method\s*=\s*[\'\"](\w+)',s))
 # Arithmetic-only reconstruction from historical functions and installed helper, not a historical execution claim.
 from package_sampling.utils import inclusion_probabilities
 env={'np':np}
 classnode=next(n for n in ast.parse(blob('graphical_sampling/population.py').decode()).body if isinstance(n,ast.ClassDef) and n.name=='Population');exec(compile(ast.Module(body=[classnode],type_ignores=[]),'historical_population','exec',flags=__future__.annotations.compiler_flag),env)
 P=env['Population']
 for c in old['cells']:
  s=''.join(c.get('source',[]))
  if 'def force_sum_to_n' in s:
   node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='force_sum_to_n');exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_force_sum','exec'),env)
 rows=[]
 specs=[('Clustered','simulations/populations/simulated/clust_N144_perm.csv',[4,8,16,32]),('Gridded','simulations/populations/simulated/grid_N144_perm.csv',[4,8,16,32]),('Random','simulations/populations/simulated/rand_N144_perm.csv',[4,8,16,32]),('meuse','simulations/populations/simulated/meuse.csv',[5,10,20]),('swiss','simulations/populations/simulated/swiss.csv',[50,100,150])]
 for pop,path,ns in specs:
  d=pd.read_csv(ROOT/path);x=d[['x','y']].to_numpy(float);N=len(x)
  for inc in ['EP','UP']:
   for n in ns:
    weights=np.ones(N) if inc=='EP' else d['prob'].to_numpy(float) if pop in ['Clustered','Gridded','Random'] else d.copper.to_numpy(float) if pop=='meuse' else np.clip(d.AREA.to_numpy(float),5,100)
    raw=inclusion_probabilities(weights.copy(),n);pi=env['force_sum_to_n'](raw,n);p=P(x,pi,variable=np.ones(N))
    rows.append(dict(Population=pop,Probability=inc,N=N,target_n=n,raw_sum=repr(float(raw.sum())),sum_after_historical_force=repr(float(pi.sum())),inferred_n=p.n,shortfall=n-p.n,mechanism_only=True))
 pd.DataFrame(rows).to_csv(EV/'historical_n_inference_reconstruction.csv',index=False)
 results['n_inference_reconstruction']=dict(settings=len(rows),shortfall_settings=sum(r['shortfall']>0 for r in rows),numpy=np.__version__,no_actual_historical_cluster_counts_logged=True)
 # Figure files and plotting-input paths link actual historical artifacts, not package presence alone.
 figurematches=[]
 for name in ['simulated_result_measure_medoid.pdf','meuse_balance_indices_measure_medoid.pdf','swiss_balance_indices_measure_medoid.pdf']:
  raw=blob('simulations/plots/'+name);local=(ROOT/'simulations/plots'/name).read_bytes();assert raw==local
  figurematches.append(dict(name=name,historical_commit=REV,SHA256=hashlib.sha256(raw).hexdigest(),current_plot_byte_identical=True))
 results['committed_figure_links']=figurematches
 # Capture exact source anchors; do not execute notebook cells.
 anchors=[]
 for tag,nb in [('June1_measure',old),('June1_plots',plot),('HEAD_measure',new)]:
  for i,c in enumerate(nb['cells']):
   s=''.join(c.get('source',[]))
   for needle in ['class DensityDisparityFlexible','init_clust_method="weighted"','init_clust_method="ot"','"D": DensityDisparityFlexible','def exact_di_for_design','SAMPLE_CNT_TEST = 2000','master_summary.loc[replace_mask, "Dm"]','simulated_result_measure.pdf','nmedoids_selected_populations.csv']:
    if needle in s:anchors.append(dict(source=tag,cell=i,execution_count=c.get('execution_count'),anchor=needle))
 results['notebook_anchors']=anchors
 for p in [ROOT/'DI_reference_sensitivity_JSSAM/AUDIT.md',ROOT/'DI_reference_sensitivity_JSSAM/REPORT.md',ROOT/'DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex',Path('/home/bardia/latex-projects/GFS-Spread-Sample-JSSAM/1- Measure.tex'),Path('/home/bardia/latex-projects/GFS-Spread-Sample-JSSAM/0- FullPaperGFSS.tex'),Path('/home/bardia/Downloads/Code and Notebooks/measure_calculation_nmedoids_clean.ipynb')]:inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 results.update(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_git_HEAD=git('rev-parse','HEAD').decode().strip(),new_sampling_runs=0,new_DI_evaluations=0,new_OT_or_TSP_searches=0,input_SHA256=inputs)
 (OUT/'checks.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({k:v for k,v in results.items() if k not in ['notebook_anchors','input_SHA256']},indent=2))
if __name__=='__main__':main()

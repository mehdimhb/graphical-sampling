"""Additive balance-only Measure experiment; see plan.md and REPORT.md."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-hard-ip-balance')
from pathlib import Path
import sys,json,time,math,argparse,multiprocessing as mp,collections,platform,hashlib
HOME=Path(__file__).resolve().parent
sys.path.insert(0,str(HOME/'sources'))
import numpy as np
import pandas as pd
import psutil
from scipy.spatial.distance import cdist
from graphical_sampling.population import Population
from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.clustering.fip_balanced_nmeans import _Clustering
import ot_adapter
from adapter import refine
from corrected_tsp import solve_corrected
from probability_helper import inclusion_probabilities
BASE_SEED=20261014
TSP_SEED=20261014
RESTARTS=15
LIMIT_CPU=240.0
SPECS=[('Gridded','grid_N144_perm','prob',144),('Random','rand_N144_perm','prob',144),('Clustered','clust_N144_perm','prob',144),('Meuse','meuse','copper',155),('Swiss','swiss','AREA',959)]

def csv(rows,path):
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')

def balance_metrics(pi,labels,n):
    size=np.bincount(labels,minlength=n); mass=np.bincount(labels,weights=pi,minlength=n)
    error=np.abs(mass-1); maxpi=float(np.max(pi))
    return dict(size=size,mass=mass,abs_error=error,min_mass=float(mass.min()),max_mass=float(mass.max()),
        mean_abs_error=float(error.mean()),max_abs_error=float(error.max()),sd_mass=float(np.std(mass,ddof=0)),
        pct_gt_010=float(100*np.mean(error>.10+1e-12)),pct_gt_025=float(100*np.mean(error>.25+1e-12)),
        max_pi=maxpi,empty_clusters=np.flatnonzero(size==0).tolist(),bound_pass=bool(error.max()<=maxpi+1e-10))

def pivotal_sample(pi,rng,n):
    """Random-order pivotal draw with prescribed first-order inclusion probabilities."""
    p=pi.copy(); order=rng.permutation(len(p)); active=[int(i) for i in order if 0<p[i]<1]
    while len(active)>=2:
        i,j=active[:2]; a,b=p[i],p[j]; s=a+b; u=rng.random()
        if s<=1:
            p[i],p[j]=(s,0.) if u<a/s else (0.,s)
        else:
            prob=(1-b)/(2-s)
            p[i],p[j]=(1.,s-1) if u<prob else (s-1,1.)
        active=[k for k in (i,j) if 0<p[k]<1]+active[2:]
    if active:
        i=active[0]
        if min(p[i],1-p[i])>1e-10:raise ValueError('Pivotal remainder is not an integer within rounding.')
        p[i]=float(p[i]>.5)
    selected=np.flatnonzero(p==1)
    assert len(selected)==n and len(np.unique(selected))==n
    return selected

def prepare_design():
    if (HOME/'design.json').exists():return json.loads((HOME/'design.json').read_text())
    audit=[]; cases=[]; probabilities=[]; coordinates=[]
    root=HOME.parent; prior=root/'di_ipnm_numerical_correction_20261014'
    old=json.loads((prior/'design.json').read_text()) if (prior/'design.json').exists() else None
    for index,(name,stem,col,N) in enumerate(SPECS):
        raw=pd.read_csv(HOME/f'populations/{stem}.csv'); assert len(raw)==N
        x=raw[['x','y']].to_numpy(float); weights=raw[col].to_numpy(float)
        w=np.clip(weights,5,100) if name=='Swiss' else weights.copy()
        for i in range(N):coordinates.append(dict(population=name,unit_id=i+1,x=x[i,0],y=x[i,1],raw_weight=weights[i],used_weight=w[i]))
        sizes=[4,8,16,32,50,100] if N==144 else [5,10,20,50,100,150]
        for n in sizes:
            strict=n*w/w.sum(); rawprop=n*weights/weights.sum()
            for setting in ['EP','UP']:
                pi=np.full(N,n/N) if setting=='EP' else inclusion_probabilities(w.copy(),n)
                valid=bool(np.all((pi>0)&(pi<=1)) and abs(math.fsum(pi)-n)<=1e-10)
                row=dict(population=name,N=N,n=n,IP=setting,fraction=n/N,pi_min=float(pi.min()),pi_max=float(pi.max()),pi_sum=math.fsum(pi),
                    pi_sum_residual=math.fsum(pi)-n,valid=valid,certainty_units=int(np.sum(pi==1)),
                    simple_proportional_min=float(strict.min()),simple_proportional_max=float(strict.max()),
                    simple_proportional_valid=bool(np.all((strict>0)&(strict<=1))),
                    raw_weight_proportional_min=float(rawprop.min()),raw_weight_proportional_max=float(rawprop.max()),
                    raw_weight_proportional_valid=bool(np.all((rawprop>0)&(rawprop<=1))),
                    construction='EP n/N' if setting=='EP' else ('existing helper on clip(Area,5,100)' if name=='Swiss' else f'existing helper on {col}'),
                    changed_by_existing_certainty_helper=bool(setting=='UP' and np.any(strict>1)))
                audit.append(row)
                for i,p in enumerate(pi):probabilities.append(dict(population=name,N=N,n=n,IP=setting,unit_id=i+1,pi=p,simple_proportional_pi=strict[i] if setting=='UP' else n/N))
                seed=BASE_SEED+10000*index+100*n+(0 if setting=='EP' else 1)
                rng=np.random.default_rng(seed)
                for rep in range(1,4):
                    sample=np.sort(rng.choice(N,n,replace=False)) if setting=='EP' else pivotal_sample(pi,rng,n)
                    cache=None
                    if name=='Clustered' and setting=='EP' and n in (16,32) and old is not None:
                        prev=next(c for c in old['cases'] if c['n']==n and c['sample']==f'S{rep}')
                        sample=np.array(prev['zero_based_indices']); seed_used=prev['selection_seed']
                        source=prior/f'preliminary/n{n}_S{rep}'
                        if (source/'frozen.npz').exists():
                            dest=HOME/f'cached_OT/n{n}_S{rep}';dest.mkdir(parents=True,exist_ok=True)
                            import shutil
                            for f in ['frozen.npz','diagnostics.json']:shutil.copy2(source/f,dest/f)
                            cache=str(dest.relative_to(HOME))
                    else:seed_used=seed
                    cases.append(dict(case_id=f'{name}_n{n}_{setting}_r{rep}',population=name,stem=stem,N=N,n=n,IP=setting,replicate=rep,
                        sample=sample.tolist(),sample_seed=seed_used,sampling_design='SRS' if setting=='EP' else 'random-order pivotal',cache=cache,valid=valid))
    csv(audit,HOME/'probability_audit.csv');csv(probabilities,HOME/'inclusion_probabilities.csv');csv(coordinates,HOME/'population_coordinates.csv')
    design=dict(seed=BASE_SEED,tsp_seed=TSP_SEED,tsp_restarts=RESTARTS,case_cpu_limit_seconds=LIMIT_CPU,cases=cases)
    (HOME/'design.json').write_text(json.dumps(design,indent=2));return design

def population(case):
    raw=pd.read_csv(HOME/f'populations/{case["stem"]}.csv');x=raw[['x','y']].to_numpy(float)
    col=next(s[2] for s in SPECS if s[0]==case['population']);w=raw[col].to_numpy(float)
    if case['population']=='Swiss':w=np.clip(w,5,100)
    pi=np.full(len(raw),case['n']/len(raw)) if case['IP']=='EP' else inclusion_probabilities(w.copy(),case['n'])
    assert np.all((pi>0)&(pi<=1)) and abs(math.fsum(pi)-case['n'])<=1e-10
    # n is supplied to FIPBalancedNMeans explicitly; Population receives already constructed pi.
    return Population(x,pi),pi

def run_case(case):
    start=time.perf_counter(); cpu_start=time.process_time(); folder=HOME/'runs'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
    base={k:case[k] for k in ['case_id','population','N','n','IP','replicate','sample_seed','sampling_design']};base['fraction']=case['n']/case['N']
    solver_refs=[]
    try:
        pop,pi=population(case);x=pop.coords;s=np.array(case['sample']);n=case['n'];N=case['N']
        if case['cache']:
            state=np.load(HOME/case['cache']/'frozen.npz');diag=json.loads((HOME/case['cache']/'diagnostics.json').read_text())
            assert np.array_equal(state['sample'],s) and np.array_equal(state['coords'],x) and np.array_equal(state['pi'],pi)
            P=state['P'];centers=state['centers'];prelim=state['labels'];base['reused_OT']=True
        else:
            # Same package updates and tolerances; enforce a computation budget between batches.
            class BudgetOT(ot_adapter.ConvergedOT):
                def __init__(self,K):
                    super().__init__(K)
                    solver_refs.append(self)
                def _sinkhorn_log(self,C,pi,b,f,g):
                    for batch in range(ot_adapter.MAX_SINKHORN_UPDATES//100):
                        if time.process_time()-cpu_start>LIMIT_CPU:raise TimeoutError(f'OT exceeded {LIMIT_CPU:g} process CPU seconds')
                        f,g=_Clustering._sinkhorn_log(self,C,pi,b,f,g)
                        P=np.exp((f[:,None]+g[None,:]-C)/self.epsilon)
                        row=float(abs(P.sum(axis=1)-pi).max());col=float(abs(P.sum(axis=0)-b).max())
                        self.latest_row_error=row;self.latest_col_error=col
                        if max(row,col)<=1e-10:
                            self.total_sinkhorn+=(batch+1)*100;self.max_updates=max(self.max_updates,(batch+1)*100)
                            self.worst_row_error=max(self.worst_row_error,row);self.worst_col_error=max(self.worst_col_error,col)
                            return f,g
                    raise RuntimeError('OT inner iteration limit exceeded')
            ot_adapter.ConvergedOT=BudgetOT
            part=ot_adapter.ManuscriptPartition(n=n,init_clust_method='ot',tol=1e-7,max_iter=2000,n_init=1)
            # Keep class reference for failure diagnostics without changing sample initialization.
            prelim,centers=part._get_labels_centroids(x,pi,x[s]);P=part.transport;diag=part.ot_diagnostics;base['reused_OT']=False
        base.update({k:v for k,v in diag.items() if k in (
            "outer_iterations","outer_shift","final_fixed_point_residual","total_sinkhorn_updates","max_updates_per_solve",
            "row_marginal_error","relative_row_marginal_error","column_marginal_error","converged")})
        if case["cache"]:base["cached_source_case_id"]=diag.get("case_id")
        path,tsp,_,_=solve_corrected(centers,prelim,pop.ids,pop.ids[s],TSP_SEED,RESTARTS);base.update(tsp)
        state=refine(path,prelim,x,pi,pop.ids,centers,pop.ids[s],mode='corrected')
        mu=state['mu'];labels=state['labels'];metrics=balance_metrics(pi,labels,n)
        base.update({k:v for k,v in metrics.items() if k not in ('mass','size','abs_error')})
        base.update(status='completed',soft_mass_max_error=float(abs(pi@mu-1).max()),membership_row_error=float(abs(mu.sum(axis=1)-1).max()),
                    exact_half_ties=state['exact_half_ties'],ordering_key_ties=state['ordering_key_ties'],share_arithmetic=state['share_arithmetic'])
        assert base['soft_mass_max_error']<1e-9 and base['membership_row_error']<1e-12
        assert metrics['bound_pass']
        np.savez_compressed(folder/'state.npz',coords=x,pi=pi,sample=s,P=P,centers=centers,preliminary_labels=prelim,
            path=state['path'],omega=state['omega'],mu=mu,labels=labels,initial_centers=x[s])
        csv([dict(cluster=k,initial_center_unit_id=int(pop.ids[s[k]]),size=int(metrics['size'][k]),mass=float(metrics['mass'][k]),abs_error=float(metrics['abs_error'][k]),soft_mass=float((pi@mu)[k])) for k in range(n)],folder/'clusters.csv')
    except Exception as e:
        base.update(status='cpu_limit' if isinstance(e,TimeoutError) else 'failed',issue=f'{type(e).__name__}: {e}')
        if solver_refs:
            solver=solver_refs[-1]
            base.update(outer_iterations_at_stop=solver.outer_calls,completed_sinkhorn_updates_at_stop=solver.total_sinkhorn,
                        last_row_error=getattr(solver,'latest_row_error',None),last_col_error=getattr(solver,'latest_col_error',None))
    base.update(cpu_seconds=time.process_time()-cpu_start,wall_seconds=time.perf_counter()-start)
    (folder/'summary.json').write_text(json.dumps(base,indent=2));return base

def worker(case):
    os.nice(10)
    r=run_case(case)
    print(case['case_id'],r['status'],round(r['cpu_seconds'],2),'CPU s',r.get('issue',''),flush=True)

def worker_control(target,cap,cpu,rolling,memory,last_full_pause):
    if cpu>=58 or rolling>=57.5 or memory>=78:
        target=max(1,target-1)
    elif not last_full_pause and cpu<48 and rolling<48 and target<cap:
        target+=1
    full=bool(cpu>=60 or rolling>=57.5 or memory>=78)
    return target,full

def schedule(cases,jobs):
    queue=collections.deque(cases);active={};history=collections.deque(maxlen=15)
    cap=min(jobs,max(1,int(psutil.cpu_count()*.55)));target=cap;ctx=mp.get_context('fork')
    log=HOME/'resource_monitor_v2.csv';new=not log.exists();last_full_pause=False;paused_pids=set()
    with log.open('a',buffering=1) as f:
        if new:f.write('utc_epoch,cpu_percent,rolling_30s_cpu_percent,memory_percent,available_GiB,experiment_RSS_GiB,active_workers,target_workers,throttle,runnable_workers,suspended_workers\n')
        psutil.cpu_percent(interval=None);last_status=0
        while queue or active:
            for pid,(proc,case) in list(active.items()):
                if not proc.is_alive():
                    proc.join();del active[pid];paused_pids.discard(pid)
                    path=HOME/'runs'/case['case_id']/'summary.json'
                    if not path.exists():
                        path.parent.mkdir(parents=True,exist_ok=True)
                        path.write_text(json.dumps({**case,'status':'worker_exit','issue':f'exit code {proc.exitcode}'}))
            cpu=psutil.cpu_percent(interval=None);history.append(cpu);rolling=float(np.mean(history));mem=psutil.virtual_memory()
            target,full_pause=worker_control(target,cap,cpu,rolling,mem.percent,last_full_pause)
            if not full_pause:
                while queue and len(active)<target:
                    case=queue.popleft();p=ctx.Process(target=worker,args=(case,));p.start();active[p.pid]=(p,case)
            runnable=set() if full_pause else set(list(active)[:target])
            rss=0
            for pid in active:
                try:
                    p=psutil.Process(pid);rss+=p.memory_info().rss
                    if pid in runnable and pid in paused_pids:p.resume();paused_pids.discard(pid)
                    elif pid not in runnable and pid not in paused_pids:p.suspend();paused_pids.add(pid)
                except psutil.NoSuchProcess:pass
            f.write(f'{time.time()},{cpu},{rolling},{mem.percent},{mem.available/2**30},{rss/2**30},{len(active)},{target},{int(full_pause or bool(paused_pids))},{len(runnable)},{len(paused_pids)}\n')
            if time.monotonic()-last_status>30:
                print('MONITOR',len(queue),'queued,',len(active),'active,',len(runnable),'runnable; CPU',cpu,'30s',round(rolling,1),'RAM',mem.percent,'target',target,flush=True);last_status=time.monotonic()
            last_full_pause=full_pause
            time.sleep(2)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','first','replicates','case'],default='first');parser.add_argument('--jobs',type=int,default=8);parser.add_argument('--case-id');args=parser.parse_args()
    design=prepare_design();cases=[c for c in design['cases'] if c['valid']]
    # Compile actual approximate TSP functions once; fork workers inherit machine code.
    solve_corrected(np.array([[0.,0.],[.4,.3],[.8,.4],[.9,.9]]),np.arange(4),np.arange(1,5),np.arange(1,5),TSP_SEED,RESTARTS)
    if args.phase=='case':
        c=next(c for c in cases if c['case_id']==args.case_id);print(json.dumps(run_case(c),indent=2));return
    if args.phase=='pilot':cases=[next(c for c in cases if c['case_id']=='Clustered_n4_EP_r1')]
    elif args.phase=='first':cases=[c for c in cases if c['replicate']==1]
    elif args.phase=='replicates':
        feasible=[]
        for c in cases:
            p=HOME/'runs'/c['case_id'].replace(f'_r{c["replicate"]}','_r1')/'summary.json'
            if c['replicate']>1 and (c['cache'] or c['n']<=20 or (p.exists() and json.loads(p.read_text())['status']=='completed')):feasible.append(c)
        cases=feasible
    cases=[c for c in cases if not (HOME/'runs'/c['case_id']/'summary.json').exists()]
    schedule(cases,args.jobs)

if __name__=='__main__':main()

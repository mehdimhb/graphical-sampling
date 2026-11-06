"""One fixed-sample manuscript reference and DI, with exact declared ties."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-jssam-reference')
from pathlib import Path
import sys,json,math,time,hashlib
HOME=Path(__file__).resolve().parent
sys.path.insert(0,str(HOME/'sources'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from numerical import FixedPopulation,exact_medoids,exact_match,exact_open_tsp
from ot_solver import diagnostic,atomic_json
from adapter import refine
from corrected_tsp import solve_corrected
from probability_helper import inclusion_probabilities
from density_helper import BoxKDE
from graphical_sampling.index import DensityDisparity
PROTOCOL=json.loads((HOME/'protocol.json').read_text())

def population(case):
    raw=pd.read_csv(HOME/f'populations/{case["stem"]}.csv');x=raw[['x','y']].to_numpy(float)
    if case['IP']=='EP':pi=np.full(len(raw),case['n']/len(raw))
    else:
        weights=raw['copper'].to_numpy(float) if case['population']=='Meuse' else np.clip(raw['AREA'].to_numpy(float),5,100)
        pi=inclusion_probabilities(weights.copy(),case['n'])
    return raw,FixedPopulation(x,pi,case['n'])

class ManuscriptScore(DensityDisparity):
    def __init__(self,pop):
        self.box=BoxKDE(pop.coords,'box_covariance')
        super().__init__(pop,n_jobs=1,representative='nmedoids',kde_rtol=0)
    def _density(self,coords):return self.box.density(coords)

def OT_folder(case,index):return HOME/'ot'/f'{case["scenario"]}_{index:02d}'

def baseline_state(case):
    reference=HOME/'baselines'/f'{case["scenario"]}.json'
    if not reference.exists():return None
    row=json.loads(reference.read_text())
    return HOME/row['ot_folder']

def benchmark(case,folder,ot):
    path=folder/'TSP_benchmark.json'
    if path.exists():return json.loads(path.read_text())
    sample=np.asarray(case['sample']);ids=np.arange(1,case['N']+1)
    route,meta,raw,costs=solve_corrected(ot['centers'],ot['labels'],ids,ids[sample],PROTOCOL['TSP_quality_benchmark_seed'],100)
    info=dict(heuristic_length=meta['TSP_length'],heuristic_path=route.tolist(),seed=PROTOCOL['TSP_quality_benchmark_seed'],restarts=100,exact=False)
    # Baseline instances only: the default centroid configuration at n10/n16.
    if folder.name.endswith('_00') and case['n']<=16:
        try:
            route,cert=exact_open_tsp(ot['centers']);info.update(exact=True,exact_path=route.tolist(),**cert)
        except ValueError as e:info['exact_benchmark_unavailable']=str(e)
    np.savez_compressed(folder/'TSP_benchmark_candidates.npz',paths=raw,lengths=costs)
    atomic_json(path,info);return info

def evaluate(case,ot_folder,experiment,index,tsp_seed,path_override=None,save=True):
    run_id=f'{case["scenario"]}_{experiment}_{index:02d}';out=HOME/'runs'/run_id
    if save and (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    start=time.perf_counter();cpu=time.process_time();out.mkdir(parents=True,exist_ok=True)
    raw,pop=population(case);x=pop.coords;pi=pop.inclusions;s=np.array(case['sample']);n=case['n'];N=case['N'];ids=pop.ids
    ot=np.load(Path(ot_folder)/'solution.npz');diag=json.loads((Path(ot_folder)/'diagnostics.json').read_text())
    assert diag['converged'] and np.array_equal(ot['coords'],x) and np.array_equal(ot['pi'],pi) and np.array_equal(ot['sample'],s)
    if path_override is None:
        path,tsp,_,_=solve_corrected(ot['centers'],ot['labels'],ids,ids[s],int(tsp_seed),15)
    else:path=np.array(path_override);tsp=dict(TSP_length=float(np.linalg.norm(np.diff(ot['centers'][path],axis=0),axis=1).sum()),TSP_seconds=0.)
    refined=refine(path,ot['labels'],x,pi,ids,ot['centers'],ids[s],mode='corrected');labels=refined['labels'];mu=refined['mu'];path=refined['path']
    counts=np.bincount(labels,minlength=n);hardmass=np.bincount(labels,weights=pi,minlength=n);softmass=pi@mu
    assert len(np.unique(labels))==n and counts.min()>0 and max(abs(hardmass-1))<=pi.max()+1e-10
    assert max(abs(softmass-1))<1e-9 and max(abs(mu.sum(axis=1)-1))<1e-12
    medoids,medoidcost=exact_medoids(x,pi,labels,n,ids);matching,matchcost,matchinfo=exact_match(x[medoids],x[s],ids[s],ids[s])
    matched=s[matching];translations=x[matched]-x[medoids];moved=x+translations[labels]
    scorer=ManuscriptScore(pop);f=scorer.original_density;g=scorer._density(moved);DI=float(scorer._score_single_density(g))
    q=scorer._scale((f-g)/(np.sqrt(2)*np.sqrt(f*f+g*g)),np.sin(np.pi/8));e=scorer._scale(1-(f+g)/(np.sqrt(2)*np.sqrt(f*f+g*g)),1-np.cos(np.pi/8))
    Dnet=float(q.mean());eta=float(e.mean());assert abs(DI-((1-eta)*Dnet+eta*np.sign(Dnet)))<5e-15
    hardcenters=np.array([np.average(x[labels==k],weights=pi[labels==k],axis=0) for k in range(n)])
    softcenters=((pi[:,None]*mu).T@x)/softmass[:,None]
    Qhard=float(np.sum(pi*np.sum((x-hardcenters[labels])**2,axis=1)))
    Qsoft=float(np.sum(pi[:,None]*mu*cdist(x,softcenters,metric='sqeuclidean')))
    canon=sorted((tuple(ids[labels==k].tolist()),int(ids[medoids[k]])) for k in range(n))
    refhash=hashlib.sha256(json.dumps(canon).encode()).hexdigest();medhash=hashlib.sha256(json.dumps(sorted(ids[medoids].tolist())).encode()).hexdigest()
    bmark=benchmark(case,Path(ot_folder),ot)
    row=dict(run_id=run_id,scenario=case['scenario'],population=case['population'],N=N,n=n,IP=case['IP'],experiment=experiment,index=index,
        status='completed',OT_id=Path(ot_folder).name,tsp_seed=int(tsp_seed),tsp_restarts=15,TSP_length=tsp['TSP_length'],
        TSP_benchmark_length=bmark['optimal_length'] if bmark['exact'] else bmark['heuristic_length'],TSP_benchmark_exact=bmark['exact'],
        DI=DI,Dnet=Dnet,eta=eta,hard_loss=Qhard,refined_soft_loss=Qsoft,medoid_loss=float(medoidcost.sum()),matching_squared_cost=matchcost,
        reference_hash=refhash,medoid_set_hash=medhash,hard_labels=json.dumps(labels.tolist()),medoid_ids=json.dumps(ids[medoids].tolist()),
        matched_sample_ids=json.dumps(ids[matched].tolist()),sample_ids=json.dumps(ids[s].tolist()),path=json.dumps(path.tolist()),
        constructed_clusters=int(len(np.unique(labels))),soft_mass_error=float(abs(softmass-1).max()),hard_mass_min=float(hardmass.min()),hard_mass_max=float(hardmass.max()),
        hard_mass_max_error=float(abs(hardmass-1).max()),max_pi=float(pi.max()),empty_clusters=int(np.sum(counts==0)),
        exact_half_ties=refined['exact_half_ties'],ordering_key_ties=refined['ordering_key_ties'],share_arithmetic=refined['share_arithmetic'],
        **{k:v for k,v in diag.items() if k not in ['status','scenario']},**matchinfo,
        evaluation_wall_seconds=time.perf_counter()-start,evaluation_CPU_seconds=time.process_time()-cpu)
    state=dict(coords=x,pi=pi,sample=s,initial_centers=x[s],OT_centers=ot['centers'],P=ot['P'],preliminary_labels=ot['labels'],path=path,omega=refined['omega'],
        mu=mu,labels=labels,hard_centers=hardcenters,soft_centers=softcenters,medoids=medoids,matched=matched,translations=translations,moved=moved,
        f=f,g=g,q=q,e=e,H=scorer.box.H,invroot=scorer.box.invroot,original_counts=np.rint(f*N*scorer.box.volume).astype(int),translated_counts=np.rint(g*N*scorer.box.volume).astype(int))
    if save:
        np.savez_compressed(out/'state.npz',**state);atomic_json(out/'summary.json',row)
        pd.DataFrame(dict(unit_id=ids,x_original=raw.x,y_original=raw.y,x=x[:,0],y=x[:,1],pi=pi,selected=np.isin(np.arange(N),s),cluster=labels,
            x_translated=moved[:,0],y_translated=moved[:,1],original_density=f,translated_density=g,q=q,eta_contribution=e)).to_csv(out/'units.csv',index=False,float_format='%.17g')
    return row

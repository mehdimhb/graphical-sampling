"""Bounded, resumable12x12/8-unit visual experiment; no old study is run."""
from method import *
import csv
from sklearn.metrics import adjusted_rand_score
ROOT=Path('/home/bardia/projects/graphical-sampling');OUT=Path(__file__).resolve().parent


def atomic_json(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def plan():
    candidates=[
      dict(name='compact asymmetric L',points=[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,3]]),
      dict(name='compact skew cross',points=[[1,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,3]]),
      dict(name='compact asymmetric wedge',points=[[0,0],[1,0],[2,0],[0,1],[1,1],[0,2],[1,2],[3,2]])]
    settings=dict(N=144,size=12,n=8,pi='1/18',solver_seed=17,candidates=candidates,
      shifts=[[3,2],[4,1],[2,3]],epsilon=.01,outer_tol=1e-10,marginal_tol=1e-12,
      tighter_outer_tol=1e-12,tighter_marginal_tol=1e-13,TSP_restarts=15,threads=1,
      max_outer=2000,max_seconds_per_fit=45,
      KDE='historical spherical tophat, scalar Scott',coordinates='population raw/11; no sample-based scaling',
      density_locations='f(original unit), g(translated unit)',medoid_metric='squared Euclidean',matching_metric='squared Euclidean',
      quota='exact EP overlap;18 whole units per cluster',
      method_SHA256=hashlib.sha256((OUT/'method.py').read_bytes()).hexdigest(),
      manuscript='DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex')
    path=OUT/'config.json'
    if path.exists():assert json.loads(path.read_text())==settings
    else:atomic_json(path,settings)
    return settings


def tsp_diagnostic(z):
    from itertools import permutations
    x=np.array(z['centers']);D=cdist(x,x)
    #8!=40320, with reversal-equivalent paths represented only once.
    paths=np.array([p for p in permutations(range(8)) if p[0]<p[-1]],dtype=np.int16)
    lengths=D[paths[:,:-1],paths[:,1:]].sum(axis=1)
    ordered=np.argsort(lengths,kind='stable');path=np.array(z['path'])
    actual=float(D[path[:-1],path[1:]].sum());best=float(lengths[ordered[0]])
    return dict(heuristic_length=actual,exact_best_length=best,relative_excess=actual/best-1,
      gap_first_to_second_undirected=float(lengths[ordered[1]]-lengths[ordered[0]]),
      undirected_routes_within_1e_10_of_best=int((lengths<=best+1e-10).sum()))


def checkpoint():
    summaries=[]
    for path in sorted((OUT/'runs').glob('*/summary.json')):
        z=json.loads(path.read_text());summaries.append(dict(run_id=path.parent.name,DI=z['DI'],Dnet=z['Dnet'],eta=z['eta'],
         sample=json.dumps(z['sample']),min_preliminary_gap=z['OT']['min_preliminary_gap'],
         near_preliminary_ties=z['OT']['gaps_below_1e_10'],iterations=z['OT']['iterations']))
    if summaries:
        tmp=OUT/'results.csv.tmp'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
        tmp.replace(OUT/'results.csv')
    atomic_json(OUT/'progress.json',dict(completed_full_fits=len(summaries),completed_ids=[r['run_id'] for r in summaries],
        complete=(OUT/'selection.json').exists(),original_study_reruns=0,threads=1))


def run_one(run_id,sample,outer_tol=1e-10,marginal_tol=1e-12):
    folder=OUT/'runs'/run_id;folder.mkdir(exist_ok=True)
    p=folder/'summary.json'
    if p.exists():
        z=json.loads(p.read_text());assert z['sample']==np.asarray(sample).tolist()
        assert z['OT']['outer_tol']==outer_tol and z['OT']['marginal_tol']==marginal_tol
        return z
    start=time.monotonic()
    print('START',run_id,flush=True)
    z=evaluate(12,sample,seed=17,outer_tol=outer_tol,marginal_tol=marginal_tol)
    z['tsp_diagnostic']=tsp_diagnostic(z)
    z['wall_seconds']=time.monotonic()-start
    arrays={k:np.asarray(z[k]) for k in ['sample','sample_ids','centers','coupling','preliminary_labels','path','omega','labels','medoid_ids','permutation','matched_ids','shifts','moved','f_counts','g_counts','f','g']}
    np.savez_compressed(folder/'state.npz',**arrays)
    atomic_json(p,z);checkpoint()
    print('DONE',run_id,'DI',z['DI'],'gap',z['OT']['min_preliminary_gap'],'resid',z['OT']['fixed_point_residual'],
     'medoid_ties',sum(len(t)>1 for t in z['medoid_ties']),'matching_unique',z['assignment']['unique'],
     'TSPgap',z['tsp_diagnostic']['gap_first_to_second_undirected'],'seconds',z['wall_seconds'],flush=True)
    return z


def compare(a,b):
    A=np.array(a['labels']);B=np.array(b['labels']);conf=np.zeros((8,8),int)
    for x,y in zip(A,B):conf[x,y]+=1
    rows,cols=linear_sum_assignment(-conf);new_to_old=np.empty(8,int);new_to_old[cols]=rows
    aligned=new_to_old[B];changed=np.flatnonzero(A!=aligned)
    X=np.array(a['centers']);Y=np.array(b['centers']);ri,ci=linear_sum_assignment(cdist(X,Y,metric='sqeuclidean'))
    return dict(changed_units=len(changed),changed_indices=changed.tolist(),new_to_old=new_to_old.tolist(),
      confusion=conf.tolist(),ARI=float(adjusted_rand_score(A,B)),DI_difference=b['DI']-a['DI'],
      centroid_max_aligned_difference=float(np.max(np.linalg.norm(X[ri]-Y[ci],axis=1))),
      minimum_preliminary_gap=min(a['OT']['min_preliminary_gap'],b['OT']['min_preliminary_gap']),
      minimum_TSP_distinct_route_gap=min(a['tsp_diagnostic']['gap_first_to_second_undirected'],b['tsp_diagnostic']['gap_first_to_second_undirected']))


def main():
    cfg=plan()
    if (OUT/'selection.json').exists():checkpoint();print('Already complete; preserving existing fits.');return
    comparisons=[]
    for i,entry in enumerate(cfg['candidates'],1):
        sample=np.array(entry['points'],int)
        a=run_one(f'sample{i}_original',sample)
        for delta in cfg['shifts']:
            translated=sample+delta
            assert ((translated>=0)&(translated<=11)).all()
            bid=f'sample{i}_shift{delta[0]}{delta[1]}'
            b=run_one(bid,translated);c=compare(a,b)
            row=dict(candidate=i,description=entry['name'],original_id=f'sample{i}_original',translated_id=bid,delta=delta,**c)
            row['classification']='unchanged partition' if c['changed_units']==0 else 'unvalidated difference'
            comparisons.append(row);atomic_json(OUT/'comparisons.json',comparisons)
            print('COMPARE',entry['name'],delta,c,flush=True)
            if c['changed_units']:
                ta=run_one(f'sample{i}_original_tight',sample,outer_tol=1e-12,marginal_tol=1e-13)
                tb=run_one(bid+'_tight',translated,outer_tol=1e-12,marginal_tol=1e-13)
                stableA=compare(a,ta);stableB=compare(b,tb);pair=compare(ta,tb)
                row.update(tight_original=stableA,tight_translated=stableB,tight_pair=pair)
                if stableA['changed_units']==0 and stableB['changed_units']==0 and c['centroid_max_aligned_difference']>1e-5:
                    row['classification']='stable distinct OT geometry and partition'
                elif c['centroid_max_aligned_difference']<=1e-5:
                    row['classification']='tie-sensitive ordering or hardening; no distinct OT geometry established'
                else:row['classification']='not stable under tighter convergence'
                atomic_json(OUT/'comparisons.json',comparisons)
    good=[r for r in comparisons if r['classification']=='stable distinct OT geometry and partition']
    # Choose the largest verified partition change, never the DI magnitude or sign.
    selected=max(good,key=lambda r:r['changed_units']) if good else comparisons[0]
    atomic_json(OUT/'selection.json',dict(status='verified distinct clustering' if good else 'no verified distinct clustering found',selected=selected))
    checkpoint();print('Bounded concentrated-sample experiment complete.',flush=True)

if __name__=='__main__':main()

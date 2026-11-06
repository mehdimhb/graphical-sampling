"""Small equal/unequal-FIP comparison; existing package/studies are read only."""
from base_method import *
import csv
from sklearn.metrics import adjusted_rand_score


def atomic_json(path, data):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(path)


def initialize():
    rng = np.random.default_rng(20261016)
    weights = rng.integers(24, 90, size=144)
    diff = 8192-int(weights.sum())
    weights += diff//144
    weights[rng.permutation(144)[:diff % 144]] += 1
    assert weights.sum() == 8192 and weights.min() > 0 and weights.max() < 1024
    cfg = dict(N=144, n=8, size=12, solver_seed=17, fip_seed=20261016,
        unequal_integer_weights=weights.tolist(), unequal_denominator=1024,
        fip_generation='default_rng(20261016).integers(24,90,144); common integer adjustment and seeded residual adjustment to sum 8192; pi=w/1024',
        equal_denominator=18, unequal_fixed_to_population=True,
        rows=[dict(id='dispersed', sample=[[1,1],[5,0],[9,2],[2,4],[7,5],[0,8],[4,10],[10,9]], shift=[1,1]),
              dict(id='compact', sample=[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,3]], shift=[9,8])],
        epsilon=.01, outer_tol=1e-10, marginal_tol=1e-12,
        tight_outer_tol=1e-12, tight_marginal_tol=1e-13, TSP_restarts=15,
        max_seconds_per_OT_fit=45, max_outer=2000, threads=1, workers=1,
        KDE='unweighted spherical tophat; scalar Scott h=N**(-1/6); normalized raw/11; no renormalization after internal translation',
        refinement='exact rational cumulative IP overlap; stable package within-cluster ordering',
        medoid='sum pi_l * squared Euclidean distance; exact integer weights; ties: highest pi then smallest unit ID',
        matching='optimal squared Euclidean, exact integer assignment; lexicographic tie convention',
        path_orientation='seeded OpenTSPSolver path as returned; no extra route search or forced matching',
        primary_runs=8, planned_total_fits=14,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            OUT/'base_method.py', ROOT/'graphical_sampling/clustering/fip_balanced_nmeans.py',
            ROOT/'DI_reference_sensitivity_JSSAM/numerical.py',
            ROOT/'DI_reference_sensitivity_JSSAM/accelerated_ot.py',
            ROOT/'DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex']})
    if (OUT/'config.json').exists(): assert json.loads((OUT/'config.json').read_text()) == cfg
    else: atomic_json(OUT/'config.json', cfg)
    with (OUT/'population_FIPs.csv').open('w', newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['unit_id','x','y','pi_equal','pi_unequal','unequal_integer_weight'])
        for i,w in enumerate(weights):writer.writerow([i,i%12,i//12,1/18,int(w)/1024,int(w)])
    return cfg


def evaluate(sample, weights, denominator, seed=17, outer_tol=1e-10, marginal_tol=1e-12, offset=(0,0)):
    raw=np.array([(i,j) for j in range(12) for i in range(12)], dtype=int)
    sample=np.asarray(sample,int); ids=sample[:,1]*12+sample[:,0]
    assert sample.shape == (8,2) and np.all((sample>=0)&(sample<=11)) and len(set(ids))==8
    pi=weights/denominator
    pop=Population(raw+np.asarray(offset),pi,n=8)
    x=pop.coords; assert np.array_equal(x,raw/11)
    centers,P,diag=solve_ot(x,pi,x[ids],outer_tol,marginal_tol)
    preliminary=np.argmax(P,axis=1)
    tsp_seed=int(np.random.SeedSequence(seed).generate_state(2)[1])
    path=OpenTSPSolver(random_state=tsp_seed).solve(centers,restarts=15)
    part=FIPBalancedNMeans(n=8,random_state=seed)
    part._generate_exact_clusters(path,preliminary,x,pi,np.arange(144))
    omega=part._before_indices
    # pi is exactly w/1024 for UP; EP uses its declared rational value 1/18.
    upper=np.cumsum(weights[omega],dtype=np.int64); lower=np.r_[0,upper[:-1]]
    cuts=np.arange(8,dtype=np.int64)*denominator
    overlap=np.maximum(np.minimum(upper[:,None],cuts[None,:]+denominator)
                       -np.maximum(lower[:,None],cuts[None,:]),0)
    mass=np.zeros((144,8),np.int64); mass[np.ix_(omega,path)]=overlap
    assert np.array_equal(mass.sum(1),weights) and np.all(mass.sum(0)==denominator)
    mu=mass/weights[:,None]; labels=np.argmax(mass,axis=1)
    maxmass=mass.max(1); hard_ties=np.flatnonzero((mass==maxmass[:,None]).sum(1)>1)
    medoids=[]; medoid_ties=[]
    for k in range(8):
        units=np.flatnonzero(labels==k); assert len(units)>0
        costs=(((raw[units,None,:]-raw[None,units,:])**2).sum(2)*weights[units][None,:]).sum(1)
        candidates=units[costs==costs.min()]
        chosen=min(candidates,key=lambda i:(-int(weights[i]),int(i)))
        medoids.append(int(chosen));medoid_ties.append(candidates.tolist())
    medoids=np.array(medoids,int)
    perm,assignment=optimal_match(raw[medoids],sample)
    matched=ids[perm];shifts=raw[matched]-raw[medoids];moved=raw+shifts[labels]
    f,fc,h,bf=density(raw,11);g,gc,hg,bg=density(moved,11);assert h==hg
    values=quantities(f,g)
    all_paths=np.array([p for p in itertools.permutations(range(8)) if p[0]<p[-1]],dtype=np.int16)
    D=cdist(centers,centers);lengths=D[all_paths[:,:-1],all_paths[:,1:]].sum(1)
    actual=float(D[path[:-1],path[1:]].sum());best=float(lengths.min())
    key_ties=0;minimum_key_gap=float('inf')
    for i,k in enumerate(path):
        units=np.flatnonzero(preliminary==k);key=np.zeros(len(units))
        for neighbor,sign in [(i-1,1),(i+1,-1)]:
            if 0<=neighbor<8:
                other=np.flatnonzero(preliminary==path[neighbor])
                if len(other):key+=sign*cdist(x[units],x[other]).min(1)
        gaps=np.diff(np.sort(key));key_ties+=int((gaps==0).sum())
        if np.any(gaps>0):minimum_key_gap=min(minimum_key_gap,float(gaps[gaps>0].min()))
    return dict(sample=sample.tolist(),sample_ids=ids.tolist(),weights=weights.tolist(),denominator=denominator,
        pi=pi.tolist(),seed=seed,tsp_seed=tsp_seed,offset=list(offset),**values,
        centers=centers.tolist(),coupling=P.tolist(),preliminary_labels=preliminary.tolist(),path=path.tolist(),
        omega=omega.tolist(),mass_numerators=mass.tolist(),mu=mu.tolist(),labels=labels.tolist(),
        clusters=[np.flatnonzero(labels==k).tolist() for k in range(8)],
        medoid_ids=medoids.tolist(),medoid_ties=medoid_ties,permutation=perm.tolist(),matched_ids=matched.tolist(),
        assignment=assignment,shifts=shifts.tolist(),moved=moved.tolist(),f=f.tolist(),g=g.tolist(),
        f_counts=fc.tolist(),g_counts=gc.tolist(),h=h,min_KDE_squared_boundary_margin=min(bf,bg),
        hard_membership_tie_ids=hard_ties.tolist(),ordering_key_exact_ties=key_ties,minimum_positive_key_gap=minimum_key_gap,
        hard_cluster_IP_totals=np.bincount(labels,weights=pi,minlength=8).tolist(),
        sample_clusters_occupied=int(len(np.unique(labels[ids]))),
        TSP=dict(length=actual,exact_best_length=best,relative_excess=actual/best-1,
                 undirected_paths_within_1e_10_of_best=int((lengths<=best+1e-10).sum())),OT=diag)


def checkpoint(cfg):
    records=[]
    for path in sorted((OUT/'runs').glob('*/summary.json')):
        z=json.loads(path.read_text())
        records.append(dict(run_id=path.parent.name,DI=z['DI'],Dnet=z['Dnet'],eta=z['eta'],
            sample_clusters_occupied=z['sample_clusters_occupied'],seed=z['seed'],tsp_seed=z['tsp_seed'],
            row_error=z['OT']['row_error'],column_error=z['OT']['column_error'],
            fixed_point_residual=z['OT']['fixed_point_residual'],minimum_preliminary_gap=z['OT']['min_preliminary_gap']))
    if records:
        tmp=OUT/'results.csv.tmp'
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        tmp.replace(OUT/'results.csv')
    count=len(records);atomic_json(OUT/'progress.json',dict(completed_fits=count,planned_fits=cfg['planned_total_fits'],
        completed_ids=[r['run_id'] for r in records],complete=count==cfg['planned_total_fits'],threads=1,workers=1))
    (OUT/'HANDOFF.md').write_text(f'''# Equal/unequal grid translation experiment

Completed {count}/{cfg['planned_total_fits']} independent fits. Eight are primary figure panels; six are numerical controls.
All completed fits are saved and skipped on resume. Existing studies/package/manuscripts remain read only.
CPU: one thread, one worker. No original 480-run study is rerun.

Resume from project root:
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python DI_translation_equal_unequal_JSSAM_20261016/run_experiment.py
```
After fits, run verify.py and make_outputs.py in this directory using the same interpreter.
''')


def run_one(cfg,run_id,sample,kind,tight=False,offset=(0,0)):
    path=OUT/'runs'/run_id;path.mkdir(parents=True,exist_ok=True)
    weights=np.ones(144,dtype=np.int64) if kind=='equal' else np.array(cfg['unequal_integer_weights'],np.int64)
    denominator=18 if kind=='equal' else 1024
    outer=cfg['tight_outer_tol'] if tight else cfg['outer_tol']
    marginal=cfg['tight_marginal_tol'] if tight else cfg['marginal_tol']
    p=path/'summary.json'
    if p.exists():
        z=json.loads(p.read_text());assert z['sample']==np.asarray(sample).tolist() and z['weights']==weights.tolist()
        assert z['offset']==list(offset) and z['OT']['outer_tol']==outer and z['OT']['marginal_tol']==marginal
        return z
    print('START',run_id,flush=True);start=time.monotonic()
    z=evaluate(sample,weights,denominator,outer_tol=outer,marginal_tol=marginal,offset=offset)
    z.update(run_id=run_id,kind=kind,tight=tight,wall_seconds=time.monotonic()-start)
    arrays={k:np.asarray(z[k]) for k in ['sample','sample_ids','weights','pi','centers','coupling','preliminary_labels','path','omega',
        'mass_numerators','mu','labels','medoid_ids','permutation','matched_ids','shifts','moved','f','g','f_counts','g_counts']}
    np.savez_compressed(path/'state.npz',**arrays);atomic_json(p,z);checkpoint(cfg)
    print('DONE',run_id,'DI',z['DI'],'gap',z['OT']['min_preliminary_gap'],'seconds',z['wall_seconds'],flush=True)
    return z


def compare(a,b):
    A=np.asarray(a['labels']);B=np.asarray(b['labels']);conf=np.zeros((8,8),int)
    np.add.at(conf,(A,B),1);r,c=linear_sum_assignment(-conf);mapping=np.empty(8,int);mapping[c]=r
    changed=np.flatnonzero(A!=mapping[B]);X=np.array(a['centers']);Y=np.array(b['centers']);ri,ci=linear_sum_assignment(cdist(X,Y))
    medoid_map=np.array(b['medoid_ids'])[np.argsort(mapping)]
    return dict(DI_before=a['DI'],DI_after=b['DI'],DI_difference=b['DI']-a['DI'],
        changed_memberships=int(len(changed)),changed_ids=changed.tolist(),new_to_old=mapping.tolist(),
        ARI=float(adjusted_rand_score(A,B)),same_partition=bool(np.array_equal(A[:,None]==A[None,:],B[:,None]==B[None,:])),
        centroid_max_aligned_difference=float(np.linalg.norm(X[ri]-Y[ci],axis=1).max()),
        same_aligned_medoid_ids=bool(np.array_equal(a['medoid_ids'],medoid_map)),
        same_original_density_counts=a['f_counts']==b['f_counts'],same_translated_density_counts=a['g_counts']==b['g_counts'])


def main():
    cfg=initialize();checkpoint(cfg);comparisons=[]
    for row in cfg['rows']:
        sample=np.array(row['sample']);shifted=sample+row['shift']
        for kind in ['equal','unequal']:
            aid=f"{row['id']}_{kind}_original";bid=f"{row['id']}_{kind}_shifted"
            a=run_one(cfg,aid,sample,kind);b=run_one(cfg,bid,shifted,kind)
            result=dict(row=row['id'],FIP=kind,shift=row['shift'],original_id=aid,shifted_id=bid,**compare(a,b))
            # Frozen-reference control: only the internally translated configuration shifts.
            f=np.asarray(a['f']);g,counts,_,_=density(np.asarray(a['moved'])+row['shift'],11)
            assert counts.tolist()==a['g_counts']
            result['fixed_reference_DI_difference']=quantities(f,g)['DI']-a['DI']
            if kind=='unequal':
                ta=run_one(cfg,aid+'_tight',sample,kind,tight=True)
                tb=run_one(cfg,bid+'_tight',shifted,kind,tight=True)
                result.update(tight_original=compare(a,ta),tight_shifted=compare(b,tb),tight_pair=compare(ta,tb))
            comparisons.append(result);atomic_json(OUT/'comparisons.json',comparisons)
    row=cfg['rows'][0];sample=np.array(row['sample'])
    run_one(cfg,'dispersed_unequal_replay',sample,'unequal')
    run_one(cfg,'dispersed_unequal_joint_coordinates',sample,'unequal',offset=(7,-4))
    checkpoint(cfg);print('All requested panels and controls complete.',flush=True)


if __name__=='__main__':main()

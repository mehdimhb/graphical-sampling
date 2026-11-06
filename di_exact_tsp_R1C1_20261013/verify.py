"""Independent dynamic-programming and serialized-state verification."""
from pathlib import Path
import json,hashlib,itertools,math
from decimal import Decimal,localcontext
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy
P=Path(__file__).resolve().parent
read=lambda name:pd.read_csv(P/name,float_precision='round_trip')
design=json.loads((P/'frozen_design.json').read_text())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h for name,h in design['source_hashes'].items())
data=read('exact_results.csv');extra=read('optimal_path_checks.csv');before=read('approximate_results.csv')
assert len(data)==20 and len(extra)==2 and len(before)==20 and (before.kind=='official').all()
original=pd.read_csv(design['source_population']);raw=original[['x','y']].to_numpy()
scale=np.ptp(raw,axis=0).max();normalized=(raw-raw.min(axis=0))/scale
original_pi=8*original.prob.to_numpy()/original.prob.to_numpy().sum()
original_H=144**(-1/3)*np.cov(normalized,rowvar=False,ddof=1)

def dynamic_program(points):
    # Held--Karp recurrence, free starting and ending nodes; no cycle-closing edge.
    z=np.asarray(points,dtype=np.longdouble);D=np.sqrt(((z[:,None,:]-z[None,:,:])**2).sum(axis=2));n=len(z)
    dp=np.full((1<<n,n),np.longdouble(np.inf));pred={};tol=np.longdouble('1e-15')
    for j in range(n):dp[1<<j,j]=0;pred[(1<<j,j)]=[]
    for mask in range(1,1<<n):
        for j in range(n):
            if not (mask>>j)&1 or mask==(1<<j):continue
            smaller=mask^(1<<j);candidates=[(dp[smaller,i]+D[i,j],i) for i in range(n) if (smaller>>i)&1]
            best=min(v for v,i in candidates);dp[mask,j]=best;pred[(mask,j)]=[i for v,i in candidates if abs(v-best)<=tol]
    mask=(1<<n)-1;minimum=dp[mask].min()
    def reconstruct(mask,j):
        if mask==(1<<j):return [(j,)]
        return [p+(j,) for i in pred[(mask,j)] for p in reconstruct(mask^(1<<j),i)]
    paths=[p for j in range(n) if abs(dp[mask,j]-minimum)<=tol for p in reconstruct(mask,j)]
    return float(minimum),paths

records=[];samplechecks=[]
for sample in ['S1','S2']:
    group=data[data['sample']==sample].sort_values('run');baseline=group.iloc[0];base=np.load(P/baseline.state_path)
    info=json.loads((P/'enumeration'/f'{sample}_optimal_paths.json').read_text());order=np.array(info['centroid_labels_in_spatial_rank_order'])
    table=pd.read_csv(P/'enumeration'/f'{sample}_all_unoriented_paths.csv',dtype=str)
    paths=[tuple(json.loads(t)) for t in table.spatial_rank_path];costs=[Decimal(v) for v in table.length_decimal]
    assert len(paths)==math.factorial(8)//2 and len(set(paths))==len(paths)
    assert all(sorted(p)==list(range(8)) and p<p[::-1] for p in paths)
    best=min(costs);found=[p for p,c in zip(paths,costs) if abs(c-best)<=Decimal('1e-60')]
    assert found==[tuple(p) for p in info['optimal_unoriented_paths']]
    nextgap=min(c for p,c in zip(paths,costs) if p not in found)-best
    assert nextgap>Decimal('.009') and abs(nextgap-Decimal(info['runner_up_unoriented_length_gap_decimal']))<Decimal('1e-26')
    length,dp_paths=dynamic_program(base['ot_centroids'])
    expected=[]
    for path in found:
        labels=tuple(order[np.array(path)]);expected.extend([labels,labels[::-1]])
    assert set(dp_paths)==set(expected) and abs(length-float(best))<1e-14
    # Recompute every canonical path independently at long-double precision.
    z=base['ot_centroids'][order].astype(np.longdouble);dist=np.sqrt(((z[:,None,:]-z[None,:,:])**2).sum(axis=2))
    independent=np.array([sum(dist[a,b] for a,b in zip(path[:-1],path[1:])) for path in paths])
    recorded=np.array([np.longdouble(str(value)) for value in costs])
    assert max(abs(independent-recorded))<1e-14
    assert abs(float(independent.min())-float(best))<1e-14
    independent_min={paths[i] for i in np.flatnonzero(abs(independent-independent.min())<1e-15)}
    assert independent_min==set(found)
    # A separate120-digit evaluation of the winning path from exact float inputs.
    with localcontext() as ctx:
        ctx.prec=120;points=[[Decimal.from_float(float(v)) for v in row] for row in base['ot_centroids'][order]]
        d=[[sum((a-b)**2 for a,b in zip(u,v)).sqrt() for v in points] for u in points]
        hi=sum((d[a][b] for a,b in sorted(tuple(sorted((u,v))) for u,v in zip(found[0][:-1],found[0][1:]))),Decimal(0))
        assert abs(hi-Decimal(info['high_precision_best_length_decimal']))<Decimal('1e-110')
    for old in before[before['sample']==sample].itertuples():
        state=np.load(P/'approximate_baseline'/old.run_id/'state.npz')
        assert np.array_equal(state['P'],base['P']) and np.array_equal(state['ot_centroids'],base['ot_centroids'])
    allgroup=pd.concat([group,extra[extra['sample']==sample]])
    for row in allgroup.itertuples():
        st=np.load(P/row.state_path);x=st['coords'];pi=st['pi'];lab=st['labels'];mu=st['mu'];med=st['medoids'];matched=st['matched']
        assert (st['observed_sample']+1).tolist()==design['samples'][int(sample[1:])-1]
        assert np.array_equal(st['raw'],raw) and np.array_equal(x,normalized)
        assert np.array_equal(pi,original_pi) and np.allclose(st['H'],original_H,rtol=0,atol=1e-16)
        assert np.array_equal(st['init'],st['observed_sample'])
        if row.variant=='deterministic_exact':
            for key in base.files:
                assert st[key].dtype==base[key].dtype and st[key].shape==base[key].shape and st[key].tobytes()==base[key].tobytes()
            assert row.DI==baseline.DI
        assert tuple(st['path']) in expected
        assert max(abs(st['P'].sum(axis=1)-pi))<1e-10
        assert max(abs(st['P'].sum(axis=0)-1))<1e-10
        assert max(abs(mu.sum(axis=1)-1))<1e-9 and max(abs(pi@mu-1))<1e-10
        assert np.array_equal(lab,mu.argmax(axis=1))
        assert np.all(np.sort(mu,axis=1)[:,-1]>np.sort(mu,axis=1)[:,-2])
        weights=pi[:,None]*mu;centers=(weights.T@x)/weights.sum(axis=0)[:,None]
        qsoft=float(np.sum(weights*cdist(x,centers,metric='sqeuclidean')))
        hard_centers=np.array([np.average(x[lab==k],weights=pi[lab==k],axis=0) for k in range(8)])
        qhard=float(np.sum(pi*np.sum((x-hard_centers[lab])**2,axis=1)))
        assert abs(qsoft-row.Q_soft)<1e-14 and abs(qhard-row.Q_hard)<1e-14
        C=cdist(x,st['ot_centroids'],metric='sqeuclidean')
        objective=float(np.sum(st['P']*C)/(np.median(C)+1e-12)+.01*np.sum(xlogy(st['P'],st['P'])-st['P']))
        assert abs(objective-row.J_OT_native)<1e-13
        for k in range(8):
            units=np.flatnonzero(lab==k);center=np.average(x[units],weights=pi[units],axis=0)
            obj=pi[units].sum()*np.sum((x[units]-center)**2,axis=1)
            ties=units[np.isclose(obj,obj.min(),rtol=1e-12,atol=1e-14)];ties=ties[pi[ties]==pi[ties].max()]
            assert med[k]==ties.min()
        cost=cdist(x[med],x[st['observed_sample']],metric='sqeuclidean');r,c=linear_sum_assignment(cost)
        assert np.isclose(np.sum((x[med]-x[matched])**2),cost[r,c].sum(),rtol=0,atol=1e-14)
        assert row.matching_second_best_gap>1e-12
        moved=x+(x[matched]-x[med])[lab];assert np.array_equal(moved,st['moved'])
        eig,V=np.linalg.eigh(st['H']);inv=(V*(1/np.sqrt(eig)))@V.T
        assert np.array_equal(st['H'],base['H']) and np.array_equal(pi,base['pi'])
        volume=4*np.sqrt(np.linalg.det(st['H']))
        f=(cdist(x@inv,x@inv,metric='chebyshev')<=1).sum(axis=1)/(144*volume)
        g=(cdist(moved@inv,moved@inv,metric='chebyshev')<=1).sum(axis=1)/(144*volume)
        assert np.array_equal(f,st['f']) and np.array_equal(g,st['g'])
        den=np.sqrt(2)*np.sqrt(f*f+g*g);D=np.clip((f-g)/den/np.sin(np.pi/8),-1,1).mean();eta=np.clip((1-(f+g)/den)/(1-np.cos(np.pi/8)),-1,1).mean()
        assert abs(D-row.Dnet)<1e-14 and abs(eta-row.eta)<1e-14
        assert abs((1-eta)*D+eta*np.sign(D)-row.DI)<1e-14
        canonical=sorted((tuple((np.flatnonzero(lab==k)+1).tolist()),int(med[k]+1)) for k in range(8))
        assert hashlib.sha256(json.dumps(canonical).encode()).hexdigest()==row.reference_hash
        records.append(dict(sample=sample,seed=row.run,variant=row.variant,full_numerical_checks='passed'))
    orient=read('optimal_orientation_comparisons.csv');orient=orient[orient['sample']==sample]
    assert orient.path_is_reversal.all() and orient.same_hard_partition.all() and orient.same_reference_hash.all() and orient.same_densities.all()
    assert (orient.DI_difference==0).all()
    samplechecks.append(dict(sample=sample,exact_algorithm='independent Held-Karp DP agrees with exhaustive enumeration',oriented_paths=40320,unoriented_paths=20160,optimal_oriented_paths=len(dp_paths),optimal_unoriented_paths=len(found),runner_up_gap=str(nextgap),repeated_DI_range=float(group.DI.max()-group.DI.min()),repeated_full_states_bit_identical=True,reverse_reference_and_DI_identical=True,reverse_soft_membership_roundoff=float(orient.soft_membership_max_abs_difference.max())))
result=dict(status='passed',source_hashes_preserved=len(design['source_hashes']),exact_complete_pipeline_runs_verified=len(records),prior_official_runs_compared=len(before),no_alternative_initializations=True,same_fixed_bandwidth_IPs_and_scoring=True,independent_dynamic_programming_verified=True,samples=samplechecks,records=records)
(P/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

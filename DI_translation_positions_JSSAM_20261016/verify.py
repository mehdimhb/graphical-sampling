"""Saved-state verification only. Does not run any reference optimization."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,math,itertools,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent
perms=np.array(list(itertools.permutations(range(8))),dtype=np.int16)
raw=np.array([(x,y) for y in range(12) for x in range(12)],int)
component_error=0.;maxrow=maxcol=maxfixed=maxstation=0.;minmargin=1.;maxTSPgap=0.
checks=[]
for path in sorted((P/'runs').glob('*/summary.json')):
    z=json.loads(path.read_text());labels=np.asarray(z['labels']);medoids=np.asarray(z['medoid_ids']);sample=np.asarray(z['sample'])
    state=path.parent/'state.npz'
    with np.load(state,allow_pickle=False) as arrays:
      for key in arrays.files:assert np.array_equal(arrays[key],np.asarray(z[key]))
    assert sample.shape==(8,2) and ((sample>=0)&(sample<=11)).all()
    assert len(set(map(tuple,sample)))==8
    assert sorted(i for c in z['clusters'] for i in c)==list(range(144))
    assert list(map(len,z['clusters']))==[18]*8
    assert np.array_equal(np.bincount(z['preliminary_labels'],minlength=8),np.full(8,18))
    assert z['preliminary_labels']==z['labels']
    for k,m in enumerate(medoids):
        units=np.flatnonzero(labels==k)
        costs={int(i):sum(sum(int(raw[i,t]-raw[j,t])**2 for t in range(2)) for j in units) for i in units}
        tied=sorted(i for i,v in costs.items() if v==min(costs.values()))
        assert tied==z['medoid_ties'][k] and int(m)==tied[0]
    C=((raw[medoids,None,:]-sample[None,:,:])**2).sum(axis=2)
    costs=C[np.arange(8)[None,:],perms].sum(axis=1)
    rank=np.argsort(costs,kind='stable');best=perms[rank[0]]
    assert best.tolist()==z['permutation']
    assert int(costs[rank[0]])==z['assignment']['best_cost']
    assert int(costs[rank[1]])==z['assignment']['second_cost']
    assert bool((costs==costs.min()).sum()==1)==z['assignment']['unique']
    moved=raw+sample[best][labels]-raw[medoids][labels]
    assert np.array_equal(moved,z['moved'])
    h=144**(-1/6)
    def count(points):
        return [sum(sum(int(a[k]-b[k])**2 for k in range(2))<(h*11)**2 for b in points) for a in points]
    fc=count(raw);gc=count(moved)
    assert fc==z['f_counts'] and gc==z['g_counts']
    F=np.array(fc)/(144*math.pi*h*h);G=np.array(gc)/(144*math.pi*h*h)
    assert np.allclose(F,z['f'],rtol=2e-14,atol=0) and np.allclose(G,z['g'],rtol=2e-14,atol=0)
    q=[];e=[]
    for f,g in zip(fc,gc):
        assert f>0 and g>0
        den=math.sqrt(2)*math.sqrt(f*f+g*g)
        q.append(max(-1,min(1,(f-g)/den/math.sin(math.pi/8))))
        e.append(max(-1,min(1,(1-(f+g)/den)/(1-math.cos(math.pi/8)))))
    Dnet=math.fsum(q)/144;eta=math.fsum(e)/144;sign=(Dnet>0)-(Dnet<0)
    DI=(1-eta)*Dnet+eta*sign
    error=max(abs(Dnet-z['Dnet']),abs(eta-z['eta']),abs(DI-z['DI']))
    assert error<2e-14;component_error=max(component_error,error)
    transport=np.array(z['coupling']);x=raw/11;centers=np.array(z['centers'])
    row=float(abs(transport.sum(axis=1)-1/18).max());col=float(abs(transport.sum(axis=0)-1).max())
    residual=float(np.linalg.norm((transport.T@x)/transport.sum(axis=0)[:,None]-centers))
    assert row<=z['OT']['marginal_tol'] and col<=z['OT']['marginal_tol'] and residual<=2*z['OT']['outer_tol']
    maxrow=max(maxrow,row);maxcol=max(maxcol,col);maxfixed=max(maxfixed,residual)
    C=((x[:,None,:]-centers[None,:,:])**2).sum(axis=2);C/=np.median(C)+1e-12
    assert (transport>0).all()
    dual=C+.01*np.log(transport);station=float(abs(dual-dual[:,0,None]-dual[None,0,:]+dual[0,0]).max())
    assert station<1e-11;maxstation=max(maxstation,station)
    margin=float(np.min(np.sort(transport,axis=1)[:,-1]-np.sort(transport,axis=1)[:,-2]));minmargin=min(minmargin,margin)
    actual=z['tsp_diagnostic']['heuristic_length'];best_length=z['tsp_diagnostic']['exact_best_length']
    assert abs(actual-best_length)<1e-12
    maxTSPgap=max(maxTSPgap,abs(actual-best_length))
    checks.append(dict(run_id=path.parent.name,valid=True,DI=DI))

placements=json.loads((P/'placements.json').read_text())
a=json.loads((P/'runs/original/summary.json').read_text())
for row in placements:
    b=json.loads((P/'runs'/row['id']/'summary.json').read_text())
    A=np.array(a['labels']);B=np.array(b['labels'])
    assert np.array_equal(A[:,None]==A[None,:],B[:,None]==B[None,:])
    assert np.array_equal(np.array(b['sample'])-np.array(a['sample']),np.tile(row['delta'],(8,1)))
    assert np.array_equal(np.array(b['moved'])-np.array(a['moved']),np.tile(row['delta'],(144,1)))
    assert a['g_counts']==b['g_counts']
    assert (a['Dnet'],a['eta'],a['DI'])==(b['Dnet'],b['eta'],b['DI'])
    assert row['comparison']['changed_units']==row['tight_stability']['changed_units']==0
protected=json.loads((P/'protected_before.json').read_text())
assert all(Path(f).is_file() and hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in protected.items())
summary=dict(verified_saved_states=len(checks),new_full_fits=6,reused_original_states=2,new_target_positions=3,
 changed_partition_pairs=0,maximum_DI_difference=0,minimum_preliminary_assignment_gap=minmargin,
 max_component_error=component_error,max_row_error=maxrow,max_column_error=maxcol,max_fixed_point_residual=maxfixed,
 max_stationarity_residual=maxstation,tight_controls_passed=True,protected_files_unchanged=len(protected))
(P/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

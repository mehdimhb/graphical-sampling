"""Independent saved-state checks only; no reference optimization."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
import json,math,itertools,hashlib,csv
from fractions import Fraction
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
raw=np.array([(x,y) for y in range(12) for x in range(12)],int)
perms=np.array(list(itertools.permutations(range(8))),dtype=np.int16)
cfg=json.loads((OUT/'config.json').read_text());runs={};checks=[]
max_component_error=maxrow=maxcol=maxfixed=maxstation=0.;min_gap=float('inf')
for p in sorted((OUT/'runs').glob('*/summary.json')):
    z=json.loads(p.read_text());runs[p.parent.name]=z
    with np.load(p.parent/'state.npz',allow_pickle=False) as state:
        for k in state.files:assert np.array_equal(state[k],np.asarray(z[k]))
    sample=np.array(z['sample']);ids=sample[:,1]*12+sample[:,0]
    assert np.array_equal(ids,z['sample_ids']) and len(set(ids))==8
    assert np.all((sample>=0)&(sample<=11))
    weights=np.asarray(z['weights'],int);den=z['denominator'];pi=weights/den
    assert sum(map(int,weights))==8*den and np.all((pi>0)&(pi<1))
    assert np.array_equal(pi,z['pi'])
    omega=np.asarray(z['omega']);path=np.asarray(z['path']);assert sorted(omega)==list(range(144)) and sorted(path)==list(range(8))
    # Exact rational formulation of manuscript overlap, independent of vectorized integer cuts.
    mass=np.zeros((144,8),int);prefix=Fraction(0)
    for unit in omega:
        upper=prefix+Fraction(int(weights[unit]),den)
        for position,k in enumerate(path):
            share=max(min(upper,Fraction(position+1))-max(prefix,Fraction(position)),Fraction(0))
            assert (share*den).denominator==1
            mass[unit,k]=int(share*den)
        prefix=upper
    assert prefix==8 and np.array_equal(mass,z['mass_numerators'])
    assert np.array_equal(mass.sum(1),weights) and np.all(mass.sum(0)==den)
    assert np.allclose(mass/weights[:,None],z['mu'],atol=0,rtol=0)
    labels=np.argmax(mass,axis=1);assert np.array_equal(labels,z['labels'])
    assert len(np.unique(labels))==8
    hardmass=np.bincount(labels,weights=pi,minlength=8)
    assert np.allclose(hardmass,z['hard_cluster_IP_totals'],rtol=0,atol=2e-15)
    assert max(abs(hardmass-1))<=pi.max()+1e-14
    for k,m in enumerate(z['medoid_ids']):
        units=np.flatnonzero(labels==k)
        costs={int(i):sum(int(weights[j])*sum(int(raw[i,t]-raw[j,t])**2 for t in range(2)) for j in units) for i in units}
        tied=sorted(i for i,v in costs.items() if v==min(costs.values()))
        assert tied==z['medoid_ties'][k]
        assert m==min(tied,key=lambda i:(-int(weights[i]),i))
    medoids=np.array(z['medoid_ids']);C=((raw[medoids,None,:]-sample[None,:,:])**2).sum(2)
    totals=C[np.arange(8)[None,:],perms].sum(1);order=np.argsort(totals,kind='stable')
    assert perms[order[0]].tolist()==z['permutation']
    assert int(totals[order[0]])==z['assignment']['best_cost']
    assert int(totals[order[1]])==z['assignment']['second_cost']
    assert bool((totals==totals.min()).sum()==1)==z['assignment']['unique']
    moved=raw+(sample[perms[order[0]]]-raw[medoids])[labels]
    assert np.array_equal(moved,z['moved'])
    h=144**(-1/6);assert h==z['h']
    def counts(points):
        return [sum(sum(int(a[k]-b[k])**2 for k in range(2))<(11*h)**2 for b in points) for a in points]
    fc=counts(raw);gc=counts(moved)
    assert fc==z['f_counts'] and gc==z['g_counts'] and min(fc)>0 and min(gc)>0
    F=np.array(fc)/(144*math.pi*h*h);G=np.array(gc)/(144*math.pi*h*h)
    assert np.allclose(F,z['f'],rtol=2e-14,atol=0) and np.allclose(G,z['g'],rtol=2e-14,atol=0)
    q=[];e=[]
    for f,g in zip(fc,gc):
        norm=math.sqrt(2)*math.sqrt(f*f+g*g)
        q.append(max(-1,min(1,(f-g)/norm/math.sin(math.pi/8))))
        e.append(max(-1,min(1,(1-(f+g)/norm)/(1-math.cos(math.pi/8)))))
    D=math.fsum(q)/144;eta=math.fsum(e)/144;DI=(1-eta)*D+eta*((D>0)-(D<0))
    error=max(abs(D-z['Dnet']),abs(eta-z['eta']),abs(DI-z['DI']))
    assert error<2e-14;max_component_error=max(max_component_error,error)
    P=np.array(z['coupling']);centers=np.array(z['centers']);x=raw/11
    row=float(abs(P.sum(1)-pi).max());col=float(abs(P.sum(0)-1).max())
    fixed=float(np.linalg.norm((P.T@x)/P.sum(0)[:,None]-centers))
    assert row<=z['OT']['marginal_tol'] and col<=z['OT']['marginal_tol'] and fixed<=2*z['OT']['outer_tol']
    C=((x[:,None,:]-centers[None,:,:])**2).sum(2);C/=np.median(C)+1e-12
    assert np.all(P>0)
    dual=C+.01*np.log(P);station=float(abs(dual-dual[:,0,None]-dual[None,0,:]+dual[0,0]).max())
    assert station<1e-11
    gap=float((np.sort(P,axis=1)[:,-1]-np.sort(P,axis=1)[:,-2]).min())
    maxrow=max(maxrow,row);maxcol=max(maxcol,col);maxfixed=max(maxfixed,fixed);maxstation=max(maxstation,station);min_gap=min(min_gap,gap)
    # Exhaustive reference-path check evaluates saved points; no heuristic is rerun.
    distances=np.sqrt(((centers[:,None,:]-centers[None,:,:])**2).sum(2))
    lengths=distances[perms[:,:-1],perms[:,1:]].sum(1)
    actual=float(distances[path[:-1],path[1:]].sum());best=float(lengths.min())
    assert abs(actual-z['TSP']['length'])<1e-13 and abs(best-z['TSP']['exact_best_length'])<1e-13
    checks.append(dict(run_id=p.parent.name,DI=DI,valid=True))

comparisons=json.loads((OUT/'comparisons.json').read_text())
assert len(comparisons)==4 and len(runs)==14
for c in comparisons:
    a=runs[c['original_id']];b=runs[c['shifted_id']];A=np.array(a['labels']);B=np.array(b['labels'])
    equal=bool(np.array_equal(A[:,None]==A[None,:],B[:,None]==B[None,:]))
    assert equal==c['same_partition']
    overlaps=np.array([[sum((A==i)&(B==j)) for j in range(8)] for i in range(8)])
    most=max(int(overlaps[np.arange(8),p].sum()) for p in perms)
    assert 144-most==c['changed_memberships']
    assert b['DI']-a['DI']==c['DI_difference']
    assert np.array_equal(np.array(b['sample'])-np.array(a['sample']),np.tile(c['shift'],(8,1)))
    if equal:
        assert c['changed_memberships']==0 and a['medoid_ids']==b['medoid_ids']
        assert np.array_equal(np.array(b['moved'])-np.array(a['moved']),np.tile(c['shift'],(144,1)))
        assert a['f_counts']==b['f_counts'] and a['g_counts']==b['g_counts']
        assert (a['Dnet'],a['eta'],a['DI'])==(b['Dnet'],b['eta'],b['DI'])
    if c['FIP']=='unequal':
        for run in [a,b]:
            tight=runs[run['run_id']+'_tight']
            for field in ['labels','medoid_ids','permutation','matched_ids','moved','f_counts','g_counts','DI','Dnet','eta']:
                assert run[field]==tight[field],(run['run_id'],field)
        assert a['pi']==b['pi']==runs['dispersed_unequal_original']['pi']
base=runs['dispersed_unequal_original']
for name in ['dispersed_unequal_replay','dispersed_unequal_joint_coordinates']:
    other=runs[name]
    for field in ['centers','coupling','preliminary_labels','path','omega','labels','medoid_ids','permutation','moved','f_counts','g_counts','DI','Dnet','eta']:
        assert base[field]==other[field],(name,field)
# Equal panels reproduce the earlier studies without overwriting their outputs.
for name,source in [
    ('dispersed_equal_original','DI_translation_visual_JSSAM_20261016/runs/sample1_original'),
    ('dispersed_equal_shifted','DI_translation_visual_JSSAM_20261016/runs/sample1_shift11'),
    ('compact_equal_original','DI_translation_concentrated_JSSAM_20261016/runs/sample1_original'),
    ('compact_equal_shifted','DI_translation_positions_JSSAM_20261016/runs/top_right')]:
    old=json.loads((OUT.parent/source/'summary.json').read_text());new=runs[name]
    for field in ['labels','medoid_ids','permutation','moved','f_counts','g_counts','DI','Dnet','eta']:assert old[field]==new[field],(name,field)
protected=json.loads((OUT/'protected_before.json').read_text())
assert all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
rows=list(csv.DictReader((OUT/'results.csv').open()));assert len(rows)==14
assert len({r['run_id'] for r in rows})==14
for row in rows:assert float(row['DI'])==runs[row['run_id']]['DI']
summary=dict(all_checks_passed=True,independently_verified_fits=len(checks),primary_panels=8,
    exact_membership_arithmetic=True,exhaustive_8_point_matching_and_path_checks=True,
    direct_spherical_counts_verified=True,zero_densities=0,minimum_preliminary_assignment_gap=min_gap,
    max_density_component_error=max_component_error,max_row_error=maxrow,max_column_error=maxcol,
    max_fixed_point_residual=maxfixed,max_dual_stationarity_residual=maxstation,
    tight_unequal_controls_passed=True,same_seed_replay_passed=True,joint_coordinate_control_passed=True,
    unchanged_old_equal_results=True,protected_files_unchanged=len(protected),
    changed_partition_pairs=sum(c['changed_memberships']>0 for c in comparisons),
    max_absolute_DI_shift_difference=max(abs(c['DI_difference']) for c in comparisons))
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

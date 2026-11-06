"""Independent CSV/NPZ numerical audit; does not import the experiment adapter."""
from pathlib import Path
import hashlib,json,itertools,sys,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-reference')
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy
P=Path(__file__).resolve().parent;ROOT=P.parent
sys.path.insert(0,str(ROOT))
from graphical_sampling.clustering import FIPBalancedNMeans
from numba import njit
@njit
def seed_numba(seed):np.random.seed(seed)
read=lambda s:pd.read_csv(P/s,float_precision='round_trip')
spec=json.loads((P/'preselection.json').read_text());rows=read('scored_references.csv')
assert len(rows)==40 and not json.loads((P/'failures.json').read_text())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha for name,sha in spec['source_hashes'].items())
raw=pd.read_csv(spec['population']);rawxy=raw[['x','y']].to_numpy();origin=rawxy.min(axis=0);scale=float(np.ptp(rawxy,axis=0).max());xy=(rawxy-origin)/scale
pi=8*raw.prob.to_numpy()/raw.prob.sum();H=144**(-1/3)*np.cov(xy,rowvar=False,ddof=1)
eig,V=np.linalg.eigh(H);inv=(V*(1/np.sqrt(eig)))@V.T;volume=4*np.sqrt(np.linalg.det(H))
perms=np.array(list(itertools.permutations(range(8))),int);records=[]
for row in rows.itertuples():
    st=np.load(P/'runs'/row.run_id/'state.npz');labels=st['labels'];mu=st['mu'];T=st['P'];med=st['medoids'];matched=st['matched'];s=st['observed_sample'];index=int(row.sample[1:])-1
    assert np.array_equal(st['raw'],rawxy) and np.array_equal(st['coords'],xy)
    assert np.allclose(st['pi'],pi,rtol=0,atol=1e-16)
    assert np.array_equal(s,spec['observed_samples_zero_based'][index])
    init=s if row.kind=='official' else np.array(spec['stress_centers_zero_based'][row.run])
    assert np.array_equal(st['init'],init)
    assert row.tsp_seed==(row.run if row.kind=='official' else 0)
    assert np.array_equal(np.argmax(mu,axis=1),labels)
    hardening_ties=int((np.sort(mu,axis=1)[:,-1]==np.sort(mu,axis=1)[:,-2]).sum())
    assert hardening_ties==0
    soft=pi@mu;hard=np.bincount(labels,weights=pi,minlength=8)
    assert max(abs(soft-1))<1e-10 and max(abs(mu.sum(axis=1)-1))<1e-9
    assert max(abs(hard-1))<=pi.max()+1e-9
    assert np.allclose(hard,json.loads(row.hard_IP_totals),rtol=0,atol=5e-15)
    assert max(abs(T.sum(axis=1)-pi))<1e-10 and max(abs(T.sum(axis=0)-1))<1e-10
    C=cdist(xy,st['ot_centroids'],metric='sqeuclidean');median=np.median(C)+1e-12
    assert np.array_equal(np.argmax(T,axis=1),st['preliminary_labels'])
    fixed=np.linalg.norm((T.T@xy)/T.sum(axis=0)[:,None]-st['ot_centroids']);assert fixed<=2e-7
    Z=C/median+.01*np.log(T)
    stationarity=float(abs(Z-Z[:,[0]]-Z[[0],:]+Z[0,0]).max());assert stationarity<2e-11
    J=float(np.sum(T*C)/median+.01*np.sum(xlogy(T,T)-T));assert abs(J-row.J_OT_native)<2e-13
    W=pi[:,None]*mu;softcent=(W.T@xy)/soft[:,None]
    Qsoft=float(np.sum(W*cdist(xy,softcent,metric='sqeuclidean')));assert abs(Qsoft-row.Q_soft)<2e-13
    medoid_obj=0.;ties=[]
    for k in range(8):
        units=np.flatnonzero(labels==k);w=pi[units];center=np.average(xy[units],weights=w,axis=0)
        # Weighted-centroid identity, independent of pairwise objective implementation.
        objective=w.sum()*np.sum((xy[units]-center)**2,axis=1)+np.sum(w*np.sum((xy[units]-center)**2,axis=1))
        cand=units[np.isclose(objective,objective.min(),rtol=1e-12,atol=1e-14)];ties.append(len(cand))
        cand=cand[pi[cand]==pi[cand].max()];chosen=cand.min()
        assert med[k]==chosen;medoid_obj+=objective[np.flatnonzero(units==chosen)[0]]
    assert abs(medoid_obj-row.Q_medoid)<2e-13
    costs=cdist(xy[med],xy[s],metric='sqeuclidean');allcosts=costs[np.arange(8)[None,:],perms].sum(axis=1)
    chosen_cost=float(np.sum((xy[med]-xy[matched])**2));assert abs(chosen_cost-allcosts.min())<1e-13
    assert sorted(matched.tolist())==s.tolist();assert np.sum(abs(allcosts-allcosts.min())<1e-12)==1
    secondgap=float(np.partition(allcosts,1)[1]-allcosts.min());assert abs(secondgap-row.matching_second_best_gap)<1e-13
    translations=xy[matched]-xy[med];moved=xy+translations[labels]
    assert np.array_equal(translations,st['translations']) and np.array_equal(moved,st['moved'])
    assert np.allclose(st['H'],H,rtol=0,atol=1e-16) and np.allclose(st['invroot'],inv,rtol=0,atol=1e-14)
    count0=(cdist(xy@inv,xy@inv,metric='chebyshev')<=1).sum(axis=1)
    count1=(cdist(moved@inv,moved@inv,metric='chebyshev')<=1).sum(axis=1)
    f=count0/(144*volume);g=count1/(144*volume)
    assert np.array_equal(f,st['f']) and np.array_equal(g,st['g'])
    norm=np.sqrt(2)*np.sqrt(f*f+g*g);q=np.clip((f-g)/norm/np.sin(np.pi/8),-1,1);e=np.clip((1-(f+g)/norm)/(1-np.cos(np.pi/8)),-1,1)
    D=float(q.mean());eta=float(e.mean());di=(1-eta)*D+eta*np.sign(D)
    assert abs(D-row.Dnet)<1e-14 and abs(eta-row.eta)<1e-14 and abs(di-row.DI)<1e-14
    assert abs((row.Dplus+row.Dminus)/144-row.Dnet)<1e-14
    canonical=sorted((tuple((np.flatnonzero(labels==k)+1).tolist()),int(med[k]+1)) for k in range(8));key=hashlib.sha256(json.dumps(canonical).encode()).hexdigest();assert key==row.reference_hash
    # Deterministic relabeling and matching permutation leaves the entire cloud equal.
    permutation=np.arange(8)[::-1];inverse=np.argsort(permutation)
    relabeled=inverse[labels];assert np.array_equal(xy+translations[permutation][relabeled],moved)
    distances=cdist(st['ot_centroids'],st['ot_centroids']);path=st['path'];pathcost=distances[path[:-1],path[1:]].sum();optimal=distances[perms[:,:-1],perms[:,1:]].sum(axis=1).min()
    assert abs(pathcost-row.TSP_length)<1e-13 and abs(optimal-row.TSP_optimal_length)<1e-13
    # Replay actual randomized ordering/refinement, independently of adapter.py.
    np.random.seed(row.tsp_seed);seed_numba(row.tsp_seed)
    replay=FIPBalancedNMeans(n=8,init_clust_method='ot',n_init=1,tol=1e-7,max_iter=200)
    replay_path=replay.tsp_solver.solve(st['ot_centroids']);assert np.array_equal(replay_path,path)
    replay_clusters=replay._generate_exact_clusters(replay_path,st['preliminary_labels'],st['coords'],st['pi'],np.arange(144))
    replay_mu=replay._generate_membership(replay_clusters,144,np.arange(144));assert np.array_equal(replay_mu,mu)
    table=read('runs/'+row.run_id+'/population_density.csv')
    assert np.array_equal(table[['x_original','y_original']].to_numpy(),rawxy)
    assert np.array_equal(table[['x_translated','y_translated']].to_numpy(),moved*scale+origin)
    records.append(dict(run_id=row.run_id,OT_marginal_error=float(max(abs(T.sum(axis=1)-pi))),OT_fixed_point_residual=float(fixed),OT_stationarity_error=stationarity,medoid_ties=ties,hardening_ties=hardening_ties,assignment_unique=True,DI_identity_error=float(di-row.DI),relabeling_invariant=True,actual_package_TSP_and_refinement_replayed=True))
pairs=read('reference_pairs.csv');tol=read('objective_tolerances.csv')
assert len(pairs)==30 and not pairs.opposite_Dnet_sign.any()
assert np.max(abs(pairs.DI_signed_difference-pairs.DI_difference_sign_term-pairs.DI_difference_weighted_Dnet_term))<1e-14
for row in tol.itertuples():
    chosen=pairs[pairs['sample']==row.sample]
    if row.scope=='official':chosen=chosen[chosen.official_pair]
    if row.scope=='stress':chosen=chosen[chosen.stress_pair]
    chosen=chosen[chosen.native_OT_relative_gap<=row.relative_tolerance]
    if row.criterion!='native_OT':chosen=chosen[chosen.Qsoft_relative_gap<=row.relative_tolerance]
    if row.criterion.endswith('common_OT'):chosen=chosen[chosen.common_OT_relative_gap<=row.relative_tolerance]
    assert len(chosen)==row.eligible_distinct_pairs
    if len(chosen):assert abs(chosen.DI_absolute_difference.max()-row.max_DI_difference)<1e-14
assert (P/'reference_comparison.pdf').exists() and (P/'reference_comparison.png').exists()
summary=dict(status='passed',runs=len(records),distinct_reference_pairs=len(pairs),sample_prespecification_verified=True,same_population_IPs_H_and_kernel_for_all_runs=True,source_hashes_unchanged=len(spec['source_hashes']),complete_pipeline_state_verified=True,assignment_permutations_checked_per_run=len(perms),all_matching_assignments_unique=True,hard_balance_assumed=False,label_permutations_verified_invariant=True,DI_identity_max_error=max(abs(x['DI_identity_error']) for x in records),max_OT_marginal_error=max(x['OT_marginal_error'] for x in records),max_OT_stationarity_error=max(x['OT_stationarity_error'] for x in records),records=records)
(P/'verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))

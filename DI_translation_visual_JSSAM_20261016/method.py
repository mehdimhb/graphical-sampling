"""Isolated manuscript squared-medoid/matching, historical-KDE grid checks.
No project-file writes, no old simulation/result modification.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
os.environ['MPLCONFIGDIR']='/tmp/jssam-comment5-visual-20261016/mpl'
os.environ['NUMBA_CACHE_DIR']='/tmp/jssam-comment5-visual-20261016/numba'
import sys,json,time,itertools,signal,hashlib
from pathlib import Path
ROOT=Path('/home/bardia/projects/graphical-sampling')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'DI_reference_sensitivity_JSSAM'))
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity
from graphical_sampling.population import Population
from graphical_sampling.clustering.fip_balanced_nmeans import FIPBalancedNMeans,OpenTSPSolver
from accelerated_ot import newton_refine
from numerical import stabilized_sinkhorn_batch


def quantities(f,g):
    den=np.sqrt(2)*np.sqrt(f*f+g*g)
    Dnet=float(np.mean(np.clip((f-g)/den/np.sin(np.pi/8),-1,1)))
    eta=float(np.mean(np.clip((1-(f+g)/den)/(1-np.cos(np.pi/8)),-1,1)))
    return dict(DI=(1-eta)*Dnet+eta*np.sign(Dnet),Dnet=Dnet,eta=eta)


def density(raw,scale):
    x=np.asarray(raw,dtype=float)/scale
    kde=KernelDensity(kernel='tophat',bandwidth='scott',rtol=1e-4).fit(x)
    f=np.exp(kde.score_samples(x));h=float(kde.bandwidth_)
    sq=((np.asarray(raw)[:,None,:]-np.asarray(raw)[None,:,:])**2).sum(axis=2)
    counts=(sq < (h*scale)**2).sum(axis=1)
    direct=counts/(len(x)*np.pi*h*h)
    assert np.allclose(f,direct,rtol=2e-14,atol=0)
    return f,counts,h,float(np.min(abs(sq-(h*scale)**2)))


def optimal_match(refs,sample):
    from numerical import exact_match,_hungarian_integer
    perm,objective,info=exact_match(refs,sample,np.arange(len(refs)),np.arange(len(sample)))
    cost=((refs[:,None,:]-sample[None,:,:])**2).sum(axis=2).astype(np.int64)
    best=int(cost[np.arange(len(refs)),perm].sum());alternatives=[]
    for row,col in enumerate(perm):
        C=cost.copy();C[row,col]=10**9
        candidate=_hungarian_integer(C.tolist())
        alternatives.append(int(C[np.arange(len(refs)),candidate].sum()))
    assert objective==best
    return perm,dict(best_cost=best,second_cost=min(alternatives),unique=min(alternatives)>best,
                    exact_integer_assignment=True,tie_policy=info['tie_policy'])


def solve_ot(x,pi,init,outer_tol=1e-10,marginal_tol=1e-12,max_outer=2000):
    centers=init.copy();f=np.zeros(len(x));g=np.zeros(len(init));total=0;history=[]
    start=time.monotonic()
    def coupling(centers):
        nonlocal f,g,total
        raw=cdist(x,centers,metric='sqeuclidean');median=float(np.median(raw)+1e-12);C=raw/median
        for batch in range(100):
            f,g=stabilized_sinkhorn_batch(np.ascontiguousarray(C),pi,f,g,.01,100,False)
            f,g,_=newton_refine(C,pi,f,g,.01,30,tol=min(3e-13,marginal_tol*.25))
            P=np.exp((f[:,None]+g[None,:]-C)/.01);total+=100
            row=float(abs(P.sum(axis=1)-pi).max());col=float(abs(P.sum(axis=0)-1).max())
            if max(row,col)<=marginal_tol:return P,median,row,col,C
        raise RuntimeError(f'Inner convergence failed: row={row},col={col}')
    initial_median=None
    for it in range(max_outer):
        P,median,row,col,C=coupling(centers)
        if initial_median is None:initial_median=median
        updated=(P.T@x)/P.sum(axis=0)[:,None]
        residual=float(np.linalg.norm(updated-centers));history.append(residual)
        centers=updated
        if time.monotonic()-start>45:raise TimeoutError('45-second per-fit bound')
        if residual<=outer_tol:
            P,median,row,col,C=coupling(centers)
            fixed=float(np.linalg.norm((P.T@x)/P.sum(axis=0)[:,None]-centers))
            if fixed>2*outer_tol:continue
            gaps=np.sort(P,axis=1)[:,-1]-np.sort(P,axis=1)[:,-2]
            positive=P>1e-280
            station=float(np.max(abs((C+.01*np.log(np.maximum(P,np.finfo(float).tiny))-f[:,None]-g[None,:])[positive])))
            return centers,P,dict(iterations=it+1,fixed_point_residual=fixed,row_error=row,column_error=col,
             stationarity_residual=station,min_preliminary_gap=float(gaps.min()),
             gaps_below_1e_10=int((gaps<1e-10).sum()),initial_cost_median=initial_median,
             final_cost_median=median,sinkhorn_updates=total,outer_tol=outer_tol,marginal_tol=marginal_tol)
    raise RuntimeError('Outer limit')


def evaluate(size,sample,seed=17,offset=(0,0),outer_tol=1e-10,marginal_tol=1e-12):
    # Rows increase y, then x. Sample input positions retain their identities after translation.
    raw=np.array([(x,y) for y in range(size) for x in range(size)],int)
    lookup={tuple(x):i for i,x in enumerate(raw)};sample=np.asarray(sample,int)
    ids=np.array([lookup[tuple(x)] for x in sample],int)
    assert len(set(ids))==len(ids)
    n=len(ids);N=len(raw);scale=size-1;pi=np.full(N,n/N)
    pop=Population(raw+offset,pi)
    x=pop.coords
    assert np.array_equal(x,raw/scale)
    centers,P,diag=solve_ot(x,pi,x[ids],outer_tol,marginal_tol)
    preliminary=np.argmax(P,axis=1)
    # Reuse the actual package TSP and within-cluster ordering, with 15 restarts.
    seed_tsp=int(np.random.SeedSequence(seed).generate_state(2)[1])
    path=OpenTSPSolver(random_state=seed_tsp).solve(centers,restarts=15)
    part=FIPBalancedNMeans(n=n,random_state=seed)
    part._generate_exact_clusters(path,preliminary,x,pi,np.arange(N))
    omega=part._before_indices
    # Exact EP evaluation of the manuscript's cumulative-mass overlap equation.
    h=np.arange(N,dtype=np.int64)[:,None];j=np.arange(n,dtype=np.int64)[None,:]
    overlap=np.maximum(np.minimum((h+1)*n,(j+1)*N)-np.maximum(h*n,j*N),0)
    numer=np.zeros((N,n),np.int64);numer[np.ix_(omega,path)]=overlap
    assert np.all(numer.sum(axis=1)==n) and np.all(numer.sum(axis=0)==N)
    labels=np.argmax(numer,axis=1)
    membership_ties=int(((numer==numer.max(axis=1,keepdims=True)).sum(axis=1)>1).sum())
    medoids=[];medoid_ties=[]
    for k in range(n):
        ix=np.flatnonzero(labels==k)
        if not len(ix):raise RuntimeError('Empty hard cluster')
        # All IPs equal, so the common weight does not alter the objective ordering.
        costs=(((raw[ix,None,:]-raw[None,ix,:])**2).sum(axis=2)).sum(axis=1)
        candidates=ix[costs==costs.min()]
        medoids.append(int(candidates.min()));medoid_ties.append(candidates.tolist())
    medoids=np.array(medoids,int)
    perm,assignment=optimal_match(raw[medoids],raw[ids])
    matched=ids[perm];shifts=raw[matched]-raw[medoids]
    moved=raw+shifts[labels]
    f,fc,h,bf=density(raw,scale);g,gc,hg,bg=density(moved,scale)
    assert h==hg==N**(-1/6)
    values=quantities(f,g)
    # Long-double recomputation from independent integer neighbor counts.
    F=fc.astype(np.longdouble);G=gc.astype(np.longdouble);theta=np.arccos(np.longdouble(-1))/8
    den=np.sqrt(np.longdouble(2))*np.sqrt(F*F+G*G)
    D=np.mean(np.clip((F-G)/den/np.sin(theta),-1,1));eta=np.mean(np.clip((1-(F+G)/den)/(1-np.cos(theta)),-1,1))
    check=float((1-eta)*D+eta*np.sign(D));assert abs(check-values['DI'])<2e-14
    return dict(size=size,n=n,seed=seed,sample=sample.tolist(),sample_ids=ids.tolist(),**values,
      centers=centers.tolist(),coupling=P.tolist(),preliminary_labels=preliminary.tolist(),path=path.tolist(),
      omega=omega.tolist(),labels=labels.tolist(),clusters=[np.flatnonzero(labels==k).tolist() for k in range(n)],
      medoid_ids=medoids.tolist(),medoid_ties=medoid_ties,permutation=perm.tolist(),matched_ids=matched.tolist(),
      assignment=assignment,shifts=shifts.tolist(),moved=moved.tolist(),f_counts=fc.tolist(),g_counts=gc.tolist(),
      f=f.tolist(),g=g.tolist(),h=h,min_KDE_squared_boundary_margin=min(bf,bg),
      hard_membership_ties=membership_ties,OT=diag)



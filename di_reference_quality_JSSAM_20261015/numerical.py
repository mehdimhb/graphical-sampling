"""Faithful manuscript subproblems with explicit, exact numerical conventions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-jssam-reference')
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent/'sources'))
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist
from fractions import Fraction
from graphical_sampling.population import Population

class FixedPopulation(Population):
    def __init__(self,coords,pi,n):
        assert len(pi)==len(coords) and np.all((pi>0)&(pi<=1))
        assert abs(math.fsum(pi)-n)<1e-10
        super().__init__(coords,np.asarray(pi,float))
        self._n=int(n)  # known design size; no renormalization/capping of pi

@njit(cache=True)
def _row(C,pi,f,g,eps):
    N,K=C.shape
    for i in range(N):
        top=-np.inf
        for j in range(K):
            a=(g[j]-C[i,j])/eps
            if a>top:top=a
        total=0.
        for j in range(K):total+=math.exp((g[j]-C[i,j])/eps-top)
        f[i]=eps*(math.log(pi[i])-top-math.log(total))

@njit(cache=True)
def _col(C,f,g,eps):
    N,K=C.shape
    for j in range(K):
        top=-np.inf
        for i in range(N):
            a=(f[i]-C[i,j])/eps
            if a>top:top=a
        total=0.
        for i in range(N):total+=math.exp((f[i]-C[i,j])/eps-top)
        g[j]=eps*(-top-math.log(total))

@njit(cache=True)
def sinkhorn_batch(C,pi,f,g,eps,iterations,reverse=False):
    for step in range(iterations):
        if reverse:
            _col(C,f,g,eps);_row(C,pi,f,g,eps)
        else:
            _row(C,pi,f,g,eps);_col(C,f,g,eps)
    return f,g

def integer_coordinates(x):
    ratios=[[float(v).as_integer_ratio() for v in row] for row in np.asarray(x)]
    den=max(d for row in ratios for a,d in row)
    return [[a*(den//d) for a,d in row] for row in ratios],den

def exact_medoids(x,pi,labels,n,ids):
    xy,den=integer_coordinates(x)
    weights=[Fraction.from_float(float(p)) for p in pi];wden=max(p.denominator for p in weights)
    wi=[p.numerator*(wden//p.denominator) for p in weights]
    med=[];loss=[]
    for k in range(n):
        ix=np.flatnonzero(labels==k).tolist()
        if not ix:raise ValueError(f'Empty hard cluster {k}; no fabricated representative')
        costs=[]
        for i in ix:
            value=sum(wi[j]*sum((a-b)**2 for a,b in zip(xy[i],xy[j])) for j in ix)
            costs.append((value,-wi[i],int(ids[i]),i))
        value,_,_,chosen=min(costs);med.append(chosen);loss.append(float(Fraction(value,wden*den*den)))
    return np.array(med,int),np.array(loss)

def _hungarian_integer(cost):
    n=len(cost);u=[0]*(n+1);v=[0]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
    for i in range(1,n+1):
        p[0]=i;j0=0;minv=[None]*(n+1);used=[False]*(n+1)
        while True:
            used[j0]=True;i0=p[j0];delta=None;j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cur=cost[i0-1][j-1]-u[i0]-v[j]
                    if minv[j] is None or cur<minv[j]:minv[j]=cur;way[j]=j0
                    if delta is None or minv[j]<delta:delta=minv[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                elif minv[j] is not None:minv[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    answer=np.empty(n,int)
    for j in range(1,n+1):answer[p[j]-1]=j-1
    return answer

def exact_match(refs,sample,cluster_ids,sample_ids):
    n=len(refs);xy,den=integer_coordinates(np.vstack((refs,sample)))
    primary=[[sum((a-b)**2 for a,b in zip(xy[i],xy[n+j])) for j in range(n)] for i in range(n)]
    ro=np.argsort(cluster_ids,kind='stable');co=np.argsort(sample_ids,kind='stable');base=n+1;dominance=base**n
    composite=[[primary[int(i)][int(j)]*dominance+rank*(base**(n-1-row)) for rank,j in enumerate(co)] for row,i in enumerate(ro)]
    local=_hungarian_integer(composite);match=np.empty(n,int);match[ro]=co[local]
    exact=sum(primary[i][int(match[i])] for i in range(n));value=Fraction(exact,den*den)
    return match,float(value),dict(exact_squared_assignment_cost=str(value),tie_policy='lexicographically smallest sample-ID sequence in persistent-cluster-ID order',exact_integer_assignment=True)

@njit(cache=True)
def _held_karp(D):
    n=len(D);count=1<<n;inf=np.iinfo(np.int64).max
    dp=np.full((count,n),inf,np.int64);parent=np.full((count,n),-1,np.int16)
    for j in range(n):dp[1<<j,j]=0
    for mask in range(1,count):
        for j in range(n):
            if not (mask&(1<<j)):continue
            prevmask=mask^(1<<j)
            if prevmask==0:continue
            best=inf;last=-1
            for k in range(n):
                if (prevmask&(1<<k)) and dp[prevmask,k]!=inf:
                    cand=dp[prevmask,k]+D[k,j]
                    if cand<best:best=cand;last=k
            dp[mask,j]=best;parent[mask,j]=last
    end=0
    for j in range(1,n):
        if dp[count-1,j]<dp[count-1,end]:end=j
    value=dp[count-1,end];path=np.empty(n,np.int32);mask=count-1
    for at in range(n-1,-1,-1):
        path[at]=end;old=end;end=parent[mask,old];mask^=1<<old
    return path,value

def exact_open_tsp(points):
    x=np.asarray(points);n=len(x)
    if n>16:raise ValueError('Exact DP limited to n<=16; no excessive search')
    D=cdist(x,x);ratios=[[float(v).as_integer_ratio() for v in row] for row in D];den=max(d for row in ratios for a,d in row)
    integers=[[a*(den//d) for a,d in row] for row in ratios]
    if max(max(row) for row in integers)*(n-1)>=np.iinfo(np.int64).max:raise ValueError('Exact edge integers exceed safe int64 DP; no rounding substitution')
    path,value=_held_karp(np.array(integers,np.int64))
    ranks=np.empty(n,int);order=np.lexsort((np.arange(n),x[:,1],x[:,0]));ranks[order]=np.arange(n)
    if tuple(ranks[path[::-1]])<tuple(ranks[path]):path=path[::-1].copy()
    integer_cost=sum(integers[a][b] for a,b in zip(path[:-1],path[1:]));assert integer_cost==int(value)
    return path,dict(certified_integer_edge_optimum=True,scope='global minimum for the supplied IEEE Euclidean edge-cost matrix; not a symbolic real-coordinate Euclidean certificate',
        exact_optimal_cost=str(Fraction(int(value),den)),optimal_length=float(Fraction(int(value),den)),edge_denominator=str(den),states=(1<<n)*n)

def stabilized_sinkhorn_batch(C,pi,f,g,eps,iterations,reverse=False):
    """Same Sinkhorn equations using absorbed diagonal scalings and BLAS GEMV.

    Periodic absorption preserves f/g while preventing overflow. Log updates
    recover any vanishing row/column due to kernel underflow; no probability,
    objective, epsilon, convergence target or kernel entry is clipped.
    """
    f=np.asarray(f).copy();g=np.asarray(g).copy();K=np.exp((f[:,None]+g[None,:]-C)/eps)
    u=np.ones(len(pi));v=np.ones(C.shape[1])
    for step in range(iterations):
        if reverse:
            den=K.T@u
            if np.any(den==0) or not np.all(np.isfinite(den)):
                f=f+eps*np.log(u);g=g+eps*np.log(v);f,g=sinkhorn_batch(C,pi,f,g,eps,1,True)
                K=np.exp((f[:,None]+g[None,:]-C)/eps);u.fill(1);v.fill(1);continue
            v=1/den;den=K@v
            if np.any(den==0) or not np.all(np.isfinite(den)):
                f=f+eps*np.log(u);g=g+eps*np.log(v);f,g=sinkhorn_batch(C,pi,f,g,eps,1,True)
                K=np.exp((f[:,None]+g[None,:]-C)/eps);u.fill(1);v.fill(1);continue
            u=pi/den
        else:
            den=K@v
            if np.any(den==0) or not np.all(np.isfinite(den)):
                f=f+eps*np.log(u);g=g+eps*np.log(v);f,g=sinkhorn_batch(C,pi,f,g,eps,1,False)
                K=np.exp((f[:,None]+g[None,:]-C)/eps);u.fill(1);v.fill(1);continue
            u=pi/den;den=K.T@u
            if np.any(den==0) or not np.all(np.isfinite(den)):
                f=f+eps*np.log(u);g=g+eps*np.log(v);f,g=sinkhorn_batch(C,pi,f,g,eps,1,False)
                K=np.exp((f[:,None]+g[None,:]-C)/eps);u.fill(1);v.fill(1);continue
            v=1/den
        if max(float(np.max(u)),float(np.max(v)))>1e30 or min(float(np.min(u)),float(np.min(v)))<1e-30:
            f=f+eps*np.log(u);g=g+eps*np.log(v);K=np.exp((f[:,None]+g[None,:]-C)/eps);u.fill(1);v.fill(1)
    return f+eps*np.log(u),g+eps*np.log(v)

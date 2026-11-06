import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'sources'))
import numpy as np
from numerical import sinkhorn_batch,exact_medoids,exact_match,exact_open_tsp,FixedPopulation
from graphical_sampling.clustering.fip_balanced_nmeans import _Clustering

def test_fast_updates_match_actual_package():
    rng=np.random.default_rng(381);C=rng.random((12,3));pi=np.full(12,.25);f=rng.normal(size=12)*.01;g=rng.normal(size=3)*.01
    solver=_Clustering(3,epsilon=.01,sinkhorn_iter=100,n_jobs=1)
    a,b=solver._sinkhorn_log(C,pi,np.ones(3),f.copy(),g.copy())
    u,v=sinkhorn_batch(C,pi,f.copy(),g.copy(),.01,100,False)
    P=np.exp((a[:,None]+b-C)/.01);Q=np.exp((u[:,None]+v-C)/.01)
    assert np.allclose(Q,P,atol=2e-14,rtol=2e-12)

def test_matching_and_medoids_preserve_exact_squared_objectives_and_ties():
    x=np.array([[0.,0.],[1.,0.],[.4,.8],[1.,1.]])
    m,obj=exact_medoids(x,np.full(4,.5),np.array([0,0,1,1]),2,np.arange(1,5))
    assert np.array_equal(m,[0,2])
    refs=np.array([[0.,0.],[0.,0.]])
    match,cost,info=exact_match(refs,np.array([[1.,0.],[-1.,0.]]),np.array([8,2]),np.array([9,3]))
    assert np.array_equal(match,[0,1])  # row with persistentID2 gets smaller sampleID3
    assert cost==2

def test_exact_open_TSP_matches_exhaustive_small_instance():
    import itertools,math
    from scipy.spatial.distance import cdist
    x=np.array([[0.,0.],[.2,.7],[1.,0.],[.8,.8],[.4,.3]])
    path,info=exact_open_tsp(x)
    D=cdist(x,x);cost=math.fsum(float(D[a,b]) for a,b in zip(path[:-1],path[1:]))
    optimum=min(math.fsum(float(D[a,b]) for a,b in zip(p[:-1],p[1:])) for p in itertools.permutations(range(5)))
    assert cost==optimum and info['certified_integer_edge_optimum']

def test_known_n_survives_floating_total_without_altering_pi():
    pi=np.full(144,16/144);pi*=16/pi.sum();assert int(pi.sum())==15
    pop=FixedPopulation(np.column_stack((np.arange(144),np.zeros(144))),pi,16)
    assert pop.n==16 and np.array_equal(pop.inclusions,pi)

def test_stabilized_scaling_matches_package_log_equations():
    from numerical import stabilized_sinkhorn_batch
    rng=np.random.default_rng(514);C=rng.uniform(0,2,(14,4));pi=np.full(14,4/14)
    f=rng.normal(size=14)*.01;g=rng.normal(size=4)*.01
    solver=_Clustering(4,epsilon=.01,sinkhorn_iter=100,n_jobs=1)
    a,b=solver._sinkhorn_log(C,pi,np.ones(4),f.copy(),g.copy())
    u,v=stabilized_sinkhorn_batch(C,pi,f.copy(),g.copy(),.01,100,False)
    assert np.allclose(np.exp((u[:,None]+v-C)/.01),np.exp((a[:,None]+b-C)/.01),atol=5e-14,rtol=5e-12)

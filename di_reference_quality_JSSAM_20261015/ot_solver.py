"""Sample-initialized manuscript OT; accelerated identical log-Sinkhorn equations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time,hashlib,math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import xlogy
from numerical import stabilized_sinkhorn_batch as sinkhorn_batch

def atomic_json(path,obj):
    p=Path(str(path)+'.tmp');p.write_text(json.dumps(obj,indent=2));p.replace(path)

def save_checkpoint(folder,**arrays):
    tmp=folder/'checkpoint.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(folder/'checkpoint.npz')

def diagnostic(x,pi,centers,P):
    raw=cdist(x,centers,metric='sqeuclidean');scale=float(np.median(raw)+1e-12);C=raw/scale
    row=float(abs(P.sum(axis=1)-pi).max());col=float(abs(P.sum(axis=0)-1).max())
    fixed=float(np.linalg.norm((P.T@x)/P.sum(axis=0)[:,None]-centers))
    entropy=float(np.sum(xlogy(P,P)-P));q=float(np.sum(P*raw));native=q/scale+.01*entropy
    # On nonunderflowed entries, C+eps logP must be the sum of row/column duals.
    Z=C+.01*np.log(np.maximum(P,np.finfo(float).tiny));mask=P>1e-280
    anchor_row=int(np.argmax(mask.sum(axis=1)));cols=np.flatnonzero(mask[anchor_row]);station=0.
    # Weighted residual from a least-squares additive fit avoids demanding a single common support rectangle.
    f=np.zeros(len(x));g=np.zeros(len(centers))
    for _ in range(25):
        f=np.sum(np.where(mask,Z-g[None,:],0),axis=1)/np.maximum(1,mask.sum(axis=1))
        g=np.sum(np.where(mask,Z-f[:,None],0),axis=0)/np.maximum(1,mask.sum(axis=0))
    station=float(np.max(abs((Z-f[:,None]-g[None,:])[mask]))) if np.any(mask) else float('inf')
    separations=cdist(centers,centers);np.fill_diagonal(separations,np.inf)
    sortedP=np.sort(P,axis=1);gaps=sortedP[:,-1]-sortedP[:,-2]
    return dict(row_marginal_error=row,column_marginal_error=col,fixed_point_residual=fixed,native_OT_objective=native,
        squared_transport_loss=q,entropy_term=entropy,cost_median=scale,approximate_additive_fit_residual=station,
        smallest_centroid_separation=float(separations.min()),near_coincident_centroid_pairs=int(np.sum(separations<2e-7)//2),
        smallest_preliminary_membership_gap=float(gaps.min()),preliminary_gaps_below_1e_10=int(np.sum(gaps<1e-10)))

def solve(x,pi,sample,folder,variant,seed,cpu_limit=360,outer_tol=1e-7,marginal_tol=1e-10,max_outer=2000):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    completed=folder/'solution.npz';summary=folder/'diagnostics.json'
    if completed.exists() and summary.exists() and json.loads(summary.read_text()).get('converged'):return np.load(completed),json.loads(summary.read_text())
    start=time.perf_counter();cpu_start=time.process_time();N=len(x);n=len(sample)
    beta=1. if variant==0 else [1.,.75,.5][(variant-1)%3];reverse=bool(variant%2)
    rng=np.random.default_rng(seed);amplitude=0. if variant==0 else [.01,.03,.1][(variant-1)%3]
    f=rng.normal(size=N)*amplitude;g=rng.normal(size=n)*amplitude;centers=x[sample].copy();iteration=0;total=0;inner_start=0;old_cpu=0.
    params=dict(variant=variant,seed=seed,beta=beta,reverse_sweep=reverse,initial_dual_sd=amplitude,outer_tol=outer_tol,marginal_tol=marginal_tol)
    token=hashlib.sha256(json.dumps(params,sort_keys=True).encode()+x.tobytes()+pi.tobytes()+np.asarray(sample).tobytes()).hexdigest()
    if (folder/'checkpoint.npz').exists():
        ck=np.load(folder/'checkpoint.npz');assert str(ck['token'])==token,'Resume parameters differ'
        centers=ck['centers'];f=ck['f'];g=ck['g'];iteration=int(ck['iteration']);total=int(ck['total']);inner_start=int(ck['inner_current']);old_cpu=float(ck['cpu_spent'])
    last_checkpoint=0.;last_row=last_col=float('inf');history=[];last_shift=float('inf')
    def budget():return old_cpu+time.process_time()-cpu_start
    def checkpoint(at,inner,stage):
        nonlocal last_checkpoint
        save_checkpoint(folder,centers=centers,f=f,g=g,iteration=at,total=total,inner_current=inner,cpu_spent=budget(),token=np.array(token))
        atomic_json(folder/'heartbeat.json',dict(stage=stage,outer_iteration=at,inner_updates=inner,total_updates=total,row_error=last_row,col_error=last_col,shift=last_shift,cpu_seconds=budget(),wall_seconds=time.perf_counter()-start,**params))
        last_checkpoint=time.monotonic()
    try:
        for it in range(iteration,max_outer):
            raw=cdist(x,centers,metric='sqeuclidean');C=np.ascontiguousarray(raw/(np.median(raw)+1e-12))
            for steps in range(inner_start,1000000,100):
                if budget()>cpu_limit:checkpoint(it,steps,'cpu_limit');raise TimeoutError(f'OT process CPU budget {cpu_limit}s reached')
                f,g=sinkhorn_batch(C,pi,f,g,.01,100,reverse);total+=100
                P=np.exp((f[:,None]+g[None,:]-C)/.01)
                last_row=float(abs(P.sum(axis=1)-pi).max());last_col=float(abs(P.sum(axis=0)-1).max())
                if max(last_row,last_col)<=marginal_tol:break
                if time.monotonic()-last_checkpoint>8:checkpoint(it,steps+100,'sinkhorn')
            else:checkpoint(it,1000000,'inner_limit');raise RuntimeError('1million inner-update limit')
            inner_start=0
            updated=(P.T@x)/P.sum(axis=0)[:,None];last_shift=float(np.linalg.norm(updated-centers))
            previous_labels=np.argmax(P,axis=1);centers=centers+beta*(updated-centers)
            history.append(dict(iteration=it+1,raw_centroid_residual=last_shift,row_error=last_row,column_error=last_col,total_updates=total,cpu_seconds=budget()))
            checkpoint(it+1,0,'centroid')
            if last_shift<=outer_tol:
                raw=cdist(x,centers,metric='sqeuclidean');C=np.ascontiguousarray(raw/(np.median(raw)+1e-12))
                for validation in range(10000):
                    if budget()>cpu_limit:checkpoint(it+1,0,'cpu_limit');raise TimeoutError(f'OT process CPU budget {cpu_limit}s during validation')
                    f,g=sinkhorn_batch(C,pi,f,g,.01,100,reverse);total+=100;P=np.exp((f[:,None]+g[None,:]-C)/.01)
                    last_row=float(abs(P.sum(axis=1)-pi).max());last_col=float(abs(P.sum(axis=0)-1).max())
                    if max(last_row,last_col)<=marginal_tol:break
                labels=np.argmax(P,axis=1);fixed=float(np.linalg.norm((P.T@x)/P.sum(axis=0)[:,None]-centers))
                if max(last_row,last_col)<=marginal_tol and fixed<=2*outer_tol and np.array_equal(labels,previous_labels):
                    info=diagnostic(x,pi,centers,P)
                    # Direct dual equation is independent of the additive-fit diagnostic's convergence speed.
                    positive=P>1e-280;Z=C+.01*np.log(np.maximum(P,np.finfo(float).tiny))
                    info['OT_stationarity_residual']=float(np.max(abs((Z-f[:,None]-g[None,:])[positive])))
                    info.update(converged=True,outer_iterations=it+1,total_sinkhorn_updates=total,outer_shift=last_shift,
                        cpu_seconds=budget(),wall_seconds=time.perf_counter()-start,cpu_limit=cpu_limit,**params)
                    assert info['OT_stationarity_residual']<1e-11
                    np.savez_compressed(completed,P=P,centers=centers,labels=labels,sample=sample,initial_centers=x[sample],coords=x,pi=pi,f=f,g=g)
                    atomic_json(summary,info)
                    if history:
                        import pandas as pd
                        pd.DataFrame(history).to_csv(folder/'outer_history.csv',index=False,float_format='%.17g')
                    return np.load(completed),info
        raise RuntimeError('2000 outer-iteration limit')
    except Exception as error:
        info=dict(converged=False,error=f'{type(error).__name__}: {error}',outer_iterations=it+1,total_sinkhorn_updates=total,
            last_row_error=last_row,last_column_error=last_col,last_centroid_residual=last_shift,cpu_seconds=budget(),wall_seconds=time.perf_counter()-start,cpu_limit=cpu_limit,**params)
        atomic_json(summary,info)
        if history:
            import pandas as pd
            pd.DataFrame(history).to_csv(folder/'outer_history.csv',index=False,float_format='%.17g')
        return None,info

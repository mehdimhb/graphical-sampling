"""Exact reused OT adapters, isolated here for reproducibility."""
import numpy as np
from scipy.spatial.distance import cdist
from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.clustering.fip_balanced_nmeans import _Clustering
MAX_SINKHORN_UPDATES=1000000

class ConvergedOT(_Clustering):
    """Reuse the package's updates and add explicit marginal convergence."""
    def __init__(self,K):
        super().__init__(K,epsilon=.01,max_iter=200,sinkhorn_iter=100,tol=1e-7,n_init=1,random_state=0,n_jobs=1)
        self.outer_calls=0;self.total_sinkhorn=0;self.max_updates=0
        self.worst_row_error=0.;self.worst_col_error=0.
    def _compute_distances(self,coords,centroids):
        self.outer_calls+=1;self.last_old_centroids=np.array(centroids,copy=True)
        return super()._compute_distances(coords,centroids)
    def _sinkhorn_log(self,C,pi,b,f,g):
        for batch in range(MAX_SINKHORN_UPDATES//100):
            f,g=super()._sinkhorn_log(C,pi,b,f,g)
            P=np.exp((f[:,None]+g[None,:]-C)/self.epsilon)
            row=float(abs(P.sum(axis=1)-pi).max());col=float(abs(P.sum(axis=0)-b).max())
            if max(row,col)<=1e-10:
                self.total_sinkhorn+=(batch+1)*100;self.max_updates=max(self.max_updates,(batch+1)*100)
                self.worst_row_error=max(self.worst_row_error,row);self.worst_col_error=max(self.worst_col_error,col)
                return f,g
        raise RuntimeError(f'OT marginals did not converge after {MAX_SINKHORN_UPDATES} updates: row={row},col={col}')

class ManuscriptPartition(FIPBalancedNMeans):
    def _get_labels_centroids(self,coords,probs,init_centroids=None):
        if init_centroids is None:raise ValueError('Observed sample must initialize OT')
        solver=ConvergedOT(self.K);solver.fit(coords,probs,np.asarray(init_centroids))
        shift=float(np.linalg.norm(solver.centroids-solver.last_old_centroids))
        if shift>=solver.tol:raise RuntimeError(f'Outer OT did not converge: shift={shift}, iterations={solver.outer_calls}')
        # Validate the final coupling against the returned centers.
        C=cdist(coords,solver.centroids,metric='sqeuclidean');C/=np.median(C)+1e-12
        f,g=solver._sinkhorn_log(C,probs,np.ones(self.K),np.zeros(len(coords)),np.zeros(self.K))
        P=np.exp((f[:,None]+g[None,:]-C)/solver.epsilon)
        updated=(P.T@coords)/P.sum(axis=0)[:,None]
        final_shift=float(np.linalg.norm(updated-solver.centroids))
        if final_shift>2e-7:raise RuntimeError(f'Final OT fixed-point residual={final_shift}')
        labels=np.argmax(P,axis=1)
        if not np.array_equal(labels,solver.labels):raise RuntimeError('Final OT preliminary hard labels are not stable')
        self.ot_centroids=solver.centroids.copy();self.transport=P
        self.preliminary_labels=labels.copy()
        self.ot_diagnostics=dict(outer_iterations=solver.outer_calls,outer_shift=shift,final_fixed_point_residual=final_shift,total_sinkhorn_updates=solver.total_sinkhorn,max_updates_per_solve=solver.max_updates,row_marginal_error=float(abs(P.sum(axis=1)-probs).max()),relative_row_marginal_error=float((abs(P.sum(axis=1)-probs)/probs).max()),column_marginal_error=float(abs(P.sum(axis=0)-1).max()),converged=True)
        return labels,solver.centroids

"""Manuscript-specific adapter; execute actual package routines without source edits."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-reference')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,time,itertools
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy
from graphical_sampling.population import Population
from graphical_sampling.index import DensityDisparity
from ot_adapter import ManuscriptPartition
from density_helper import BoxKDE,seed_numba
SPEC=json.loads((OUT/'preselection.json').read_text())
SOURCE=Path(SPEC['population']);EPS=.01

def load_population():
    raw=pd.read_csv(SOURCE);pop=Population(raw[['x','y']].to_numpy(),raw.prob.to_numpy(),n=8)
    return raw,pop

def writecsv(data,path):pd.DataFrame(data).to_csv(path,index=False,float_format='%.17g')

class ManuscriptBoxScore(DensityDisparity):
    def __init__(self,pop):
        self.box=BoxKDE(pop.coords,'box_covariance')
        super().__init__(pop,n_jobs=1,clustering_tol=1e-7,clustering_max_iter=200,kde_rtol=0,representative='nmedoids')
    def _density(self,coords):return self.box.density(coords)
    def _cluster_medoids(self,labels,centroids):
        med=[];objectives=[]
        for k in range(self.pop.n):
            ix=np.flatnonzero(labels==k)
            if not len(ix):raise RuntimeError('Empty hard cluster')
            cost=cdist(self.coords[ix],self.coords[ix],metric='sqeuclidean')@self.probs[ix]
            ties=ix[np.isclose(cost,cost.min(),rtol=1e-12,atol=1e-14)]
            ties=ties[self.probs[ties]==self.probs[ties].max()]
            chosen=int(ties[np.argmin(self.pop.ids[ties])]);med.append(chosen)
            objectives.append(float(cost[np.flatnonzero(ix==chosen)[0]]))
        self.medoid_objectives=np.array(objectives);med=np.array(med)
        return self.coords[med],med

def calculate_run(sample_number,kind,run):
    """Every invocation reconstructs OT, refinement, hardening, medoids, match and KDE."""
    started=time.perf_counter();raw,pop=load_population();s=np.array(SPEC['observed_samples_zero_based'][sample_number],int)
    seed=run if kind=='official' else 0
    init=s if kind=='official' else np.array(SPEC['stress_centers_zero_based'][run],int)
    run_id=f's{sample_number+1}_{kind}_{run:02d}';folder=OUT/'runs'/run_id;folder.mkdir(parents=True,exist_ok=False)
    np.random.seed(seed);seed_numba(seed)
    part=ManuscriptPartition(n=8,init_clust_method='ot',tol=1e-7,max_iter=200,n_init=1)
    part.fit(pop,init_centroids=pop.coords[init])
    pi=pop.inclusions;x=pop.coords;P=part.transport;mu=part.membership;lab=part.labels
    assert abs(mu.sum(axis=1)-1).max()<1e-9
    soft_mass=pi@mu;assert abs(soft_mass-1).max()<1e-10
    hard_mass=np.bincount(lab,weights=pi,minlength=8)
    # Larger-share border hardening can change an interior cluster by <=max pi.
    assert abs(hard_mass-1).max()<=pi.max()+1e-9
    scorer=ManuscriptBoxScore(pop);refs,med=scorer._cluster_medoids(lab,part.centroids)
    cost=cdist(refs,x[s],metric='sqeuclidean');rr,cc=linear_sum_assignment(cost)
    matched=np.empty(8,int);matched[rr]=s[cc]
    translations=x[matched]-refs;moved=x+translations[lab]
    g=scorer._density(moved);f=scorer.original_density
    di=float(scorer._score_single_density(g)) # Actual package, original formula unchanged.
    denom=np.sqrt(2)*np.sqrt(f*f+g*g)
    q=scorer._scale((f-g)/denom,np.sin(np.pi/8))
    e=scorer._scale(1-(f+g)/denom,1-np.cos(np.pi/8))
    D=float(q.mean());eta=float(e.mean())
    assert abs(di-((1-eta)*D+eta*np.sign(D)))<5e-15
    C=cdist(x,part.ot_centroids,metric='sqeuclidean');median=float(np.median(C)+1e-12)
    rawcost=float(np.sum(P*C));entropy=float(np.sum(xlogy(P,P)-P));J=rawcost/median+EPS*entropy
    W=pi[:,None]*mu;soft_centroids=(W.T@x)/soft_mass[:,None]
    hard_centroids=np.array([np.average(x[lab==k],weights=pi[lab==k],axis=0) for k in range(8)])
    Qsoft=float(np.sum(W*cdist(x,soft_centroids,metric='sqeuclidean')))
    Qhard=float(np.sum(pi*np.sum((x-hard_centroids[lab])**2,axis=1)))
    canonical=sorted((tuple(pop.ids[lab==k].tolist()),int(pop.ids[med[k]])) for k in range(8))
    refhash=hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
    assignmentcost=float(cost[rr,cc].sum());alternatives=[]
    for row,col in zip(rr,cc):
        altered=cost.copy();altered[row,col]=np.inf;r2,c2=linear_sum_assignment(altered)
        alternatives.append(float(altered[r2,c2].sum()))
    assignmentgap=min(alternatives)-assignmentcost
    path=part.path_order;dist=cdist(part.ot_centroids,part.ot_centroids)
    pathlength=float(dist[path[:-1],path[1:]].sum())
    permutations=np.array(list(itertools.permutations(range(8))),dtype=int)
    tspopt=float(dist[permutations[:,:-1],permutations[:,1:]].sum(axis=1).min())
    scale=float(np.ptp(raw[['x','y']].to_numpy(),axis=0).max());origin=raw[['x','y']].to_numpy().min(axis=0)
    rawmoved=moved*scale+origin
    summary=dict(run_id=run_id,sample=f'S{sample_number+1}',kind=kind,run=run,tsp_seed=seed,sample_ids=json.dumps(pop.ids[s].tolist()),initial_centroid_ids=json.dumps(pop.ids[init].tolist()),reference_hash=refhash,medoid_ids=json.dumps(pop.ids[med].tolist()),sorted_medoid_ids=json.dumps(sorted(pop.ids[med].tolist())),hard_IP_totals=json.dumps(hard_mass.tolist()),hard_IP_min=float(hard_mass.min()),hard_IP_max=float(hard_mass.max()),hard_IP_max_deviation=float(abs(hard_mass-1).max()),soft_IP_error=float(abs(soft_mass-1).max()),soft_membership_row_error=float(abs(mu.sum(axis=1)-1).max()),J_OT_native=J,transport_raw_squared_cost=rawcost,transport_entropy=entropy,OT_cost_median=median,Q_soft=Qsoft,Q_hard=Qhard,Q_medoid=float(scorer.medoid_objectives.sum()),matching_squared_cost=assignmentcost,matching_second_best_gap=assignmentgap,TSP_length=pathlength,TSP_optimal_length=tspopt,TSP_relative_regret=(pathlength-tspopt)/tspopt,Dnet=D,eta=eta,DI=di,Dplus=float(np.maximum(q,0).sum()),Dminus=float(np.minimum(q,0).sum()),**part.ot_diagnostics,seconds=time.perf_counter()-started)
    np.savez_compressed(folder/'state.npz',raw=raw[['x','y']].to_numpy(),coords=x,pi=pi,observed_sample=s,init=init,P=P,ot_centroids=part.ot_centroids,preliminary_labels=part.preliminary_labels,mu=mu,labels=lab,soft_centroids=soft_centroids,hard_centroids=hard_centroids,medoids=med,matched=matched,translations=translations,moved=moved,f=f,g=g,q=q,e=e,H=scorer.box.H,invroot=scorer.box.invroot,path=path,omega=part._before_indices)
    writecsv(dict(unit_id=pop.ids,x_original=raw.x,y_original=raw.y,x_normalized=x[:,0],y_normalized=x[:,1],pi=pi,sampled=np.isin(np.arange(pop.N),s),hard_cluster=lab,cluster_medoid_id=pop.ids[med[lab]],x_translated=rawmoved[:,0],y_translated=rawmoved[:,1],x_translated_normalized=moved[:,0],y_translated_normalized=moved[:,1],original_density=f,translated_density=g,signed_contribution=q,eta_contribution=e,Dnet_contribution=q/pop.N,eta_mean_contribution=e/pop.N),folder/'population_density.csv')
    writecsv(dict(cluster_id=np.arange(8),medoid_unit_id=pop.ids[med],matched_sample_unit_id=pop.ids[matched],medoid_x=raw.x.to_numpy()[med],medoid_y=raw.y.to_numpy()[med],sample_x=raw.x.to_numpy()[matched],sample_y=raw.y.to_numpy()[matched],translation_x_normalized=translations[:,0],translation_y_normalized=translations[:,1],hard_IP_total=hard_mass,soft_IP_total=soft_mass,weighted_squared_medoid_objective=scorer.medoid_objectives),folder/'reference_matching.csv')
    writecsv(dict(unit_id=np.repeat(pop.ids,8),cluster_id=np.tile(np.arange(8),pop.N),preliminary_transport=P.ravel(),final_fractional_membership=mu.ravel()),folder/'memberships.csv')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

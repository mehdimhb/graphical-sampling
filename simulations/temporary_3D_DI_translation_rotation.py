"""Four-panel 3D illustration of DI invariance to rigid translation/rotation."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.index import DensityDisparity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulations/results/multivariate_3D_DI/3D_DI_translation_rotation_four_cases.png"
rng = np.random.default_rng(20260903)
N, n = 600, 3

centers = np.array([[-.45,-.15,-.20],[.30,.35,.20],[.35,-.35,.15]])
raw = np.vstack([rng.normal(c,[.62,.58,.60],size=(200,3)) for c in centers])
coords = raw - raw.min(axis=0)
coords /= np.ptp(coords,axis=0).max()
pik = np.full(N,n/N)
pop = SimpleNamespace(coords=coords,inclusions=pik,indices=np.arange(N),ids=np.arange(N),variable=np.zeros(N),N=N,n=n)

class SampleInitializedOT(FIPBalancedNMeans):
    def _get_labels_centroids(self,x,p,init_centroids=None):
        supplied=np.asarray(init_centroids,float)
        assert supplied.shape==(self.K,x.shape[1])
        return self._get_labels_centroids_ot_kmeans(x,p,supplied.copy())

# Establish one IP-balanced OT reference configuration.
initial = coords[[30,270,510]]
fit = SampleInitializedOT(n=3,init_clust_method="ot")
fit.fit(pop,init_centroids=initial)
base_scorer = DensityDisparity(pop,representative="nmedoids",n_jobs=1)
medoids,_ = base_scorer._cluster_medoids(fit.labels,fit.centroids)

def evaluate(x, labels, refs, sample):
    local_pop=SimpleNamespace(coords=x,inclusions=pik,indices=np.arange(N),ids=np.arange(N),variable=np.zeros(N),N=N,n=n)
    scorer=DensityDisparity(local_pop,representative="nmedoids",n_jobs=1)
    r,c=linear_sum_assignment(cdist(refs,sample)); assigned=np.empty_like(refs); assigned[r]=sample[c]
    moved=x+(assigned-refs)[labels]
    return scorer._score_single_density(scorer._density(moved)),assigned

def rz(deg):
    a=np.deg2rad(deg); return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

def rodrigues(axis,deg):
    u=np.asarray(axis,float); u/=np.linalg.norm(u); x,y,z=u; a=np.deg2rad(deg)
    K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)*np.cos(a)+(1-np.cos(a))*np.outer(u,u)+np.sin(a)*K

v1=np.array([.13,-.08,.07]); v2=np.array([-.10,.12,-.06])
R1=rz(35); R2=rodrigues([1,1,1],-40)
configs=[
    ("(a) Translation 1",coords,medoids,medoids+v1,r"$v_1=(0.13,-0.08,0.07)$"),
    ("(b) Translation 2",coords,medoids,medoids+v2,r"$v_2=(-0.10,0.12,-0.06)$"),
    ("(c) Rotated translation 1",coords@R1.T,medoids@R1.T,(medoids+v1)@R1.T,r"$35^\circ$ about $c_3$"),
    ("(d) Rotated translation 2",coords@R2.T,medoids@R2.T,(medoids+v2)@R2.T,r"$-40^\circ$ about $(1,1,1)$"),
]

results=[]
fig=plt.figure(figsize=(12,10))
axes=[fig.add_subplot(2,2,i+1,projection="3d") for i in range(4)]
colors=["#168BD2","#FF7A00","#24A148"]
for ax,(title,x,refs,sample,note) in zip(axes,configs):
    di,assigned=evaluate(x,fit.labels,refs,sample); results.append(di)
    for k,col in enumerate(colors):
        mask=fit.labels==k
        ax.scatter(*x[mask].T,s=9,color=col,alpha=.36,depthshade=False,rasterized=True)
    ax.scatter(*sample.T,marker="o",s=90,c="#E53935",edgecolors="black",linewidths=.9,depthshade=False,zorder=6)
    ax.scatter(*refs.T,marker="x",s=140,c="black",linewidths=2.5,depthshade=False,zorder=7)
    for k in range(n):
        seg=np.vstack([refs[k],assigned[k]])
        ax.plot(*seg.T,color=".25",ls="--",lw=1.2,alpha=.8)
    ax.set_title(f"{title}\n{note}\nDI = {di:.3f}",fontsize=11,pad=6)
    ax.set_xlabel(r"$c_1$",labelpad=2); ax.set_ylabel(r"$c_2$",labelpad=2); ax.set_zlabel(r"$c_3$",labelpad=2)
    ax.set_proj_type("ortho"); ax.view_init(elev=18,azim=-58); ax.tick_params(labelsize=7,pad=0)

legend=[
    Line2D([0],[0],marker="o",color="none",markerfacecolor="#E53935",markeredgecolor="black",markersize=8,label="Observed sample (shifted medoids)"),
    Line2D([0],[0],marker="x",color="black",linestyle="none",markeredgewidth=2.2,markersize=9,label="IPnM medoid (reference unit)"),
]
fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,frameon=False)
fig.suptitle("DI under Rigid Translations and Rotations",fontsize=15,y=.985)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.88,wspace=.02,hspace=.12)
OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=300,bbox_inches="tight")
print("DI values:",*[f"{v:.12g}" for v in results])
print("Saved:",OUT)

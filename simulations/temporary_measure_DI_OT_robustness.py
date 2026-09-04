"""Temporary 500-repetition OT-only DI robustness check for Measure Figures 7/8."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from package_sampling.utils import inclusion_probabilities
from graphical_sampling.population import Population
from graphical_sampling.clustering import FIPBalancedNMeans
from graphical_sampling.index import DensityDisparity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulations/results/temporary_measure_DI_OT_sample_initialized_M500"
OUT.mkdir(parents=True, exist_ok=True)
M, SEED = 500, 20260904
OLD_FILE = ROOT / "simulations/Store/master_summary_DI_nmedoids_updated.csv"
METHODS = {"MAX":"Maxe", "SCP":"Scps", "LPM":"Lopi", "WAV":"Wave"}
POPS = {
    "meuse": ("real/meuse.csv", [5,10,20], ["MAX","SCP","LPM"]),
}

ro.r("suppressPackageStartupMessages(library(BalancedSampling)); suppressPackageStartupMessages(library(WaveSampling)); suppressPackageStartupMessages(library(sampling))")

def load_pop(name, rel, n, inc):
    d=pd.read_csv(ROOT/"simulations/populations"/rel); xy=d[["x","y"]].to_numpy(float); N=len(d)
    if name=="meuse": y=d.cadmium.to_numpy(float); w=d.copper.to_numpy(float)
    elif name=="swiss": y=np.clip(d.AREA_A.to_numpy(float),5,100); w=np.clip(d.AREA.to_numpy(float),5,100)
    else: y=d["z.90"].to_numpy(float); w=d.prob.to_numpy(float)
    pik=inclusion_probabilities((np.ones(N) if inc=="EP" else w).copy(),n); pik=np.asarray(pik,float); pik*=n/pik.sum()
    return Population(coords=xy,inclusions=pik,variable=y,n=n), d

def samples(method,pop,m,seed):
    N,n=len(pop.coords),pop.n; ans=np.empty((m,n),int)
    ro.r["set.seed"](seed)
    with localconverter(ro.default_converter+numpy2ri.converter):
        ro.globalenv["coords_ot_check"]=pop.coords; ro.globalenv["pik_ot_check"]=pop.inclusions
        for i in range(m):
            if method=="Maxe": out=ro.r("sampling::UPmaxentropy(pik_ot_check)")
            elif method=="Scps": out=ro.r("BalancedSampling::scps(pik_ot_check, coords_ot_check)")
            elif method=="Lopi": out=ro.r("BalancedSampling::lpm1(pik_ot_check, coords_ot_check)")
            elif method=="Wave": out=ro.r("WaveSampling::wave(coords_ot_check, pik_ot_check)")
            a=np.asarray(out).ravel()
            idx=np.flatnonzero(a>.5) if a.size==N and np.all(np.isin(a,[0,1])) else a.astype(int)-(1 if a.size and a.min()>=1 else 0)
            if len(idx)!=n: raise RuntimeError((method,N,n,len(idx)))
            ans[i]=idx
    return ans

class SampleInitializedOT(FIPBalancedNMeans):
    """Temporary adapter: honor the paper's sample-as-initial-centers rule."""
    def _get_labels_centroids(self, coords, probs, init_centroids=None):
        if init_centroids is None:
            raise ValueError("A realized sample must initialize paper-defined OT IPnM.")
        supplied = np.asarray(init_centroids, dtype=float)
        expected_shape = (self.K, coords.shape[1])
        if supplied.shape != expected_shape:
            raise ValueError(f"Expected sample centers shaped {expected_shape}, got {supplied.shape}.")
        # This bypasses _get_labels_centroids's internal grid and passes the
        # realized sample coordinates directly into _Clustering.fit.
        return self._get_labels_centroids_ot_kmeans(coords, probs, supplied.copy())

def one_score(s,pop,scorer):
    coords=pop.coords; raw=coords[s]
    f=SampleInitializedOT(n=pop.n,init_clust_method="ot",split_size=.001)
    f.fit(pop,init_centroids=raw)
    med,_=scorer._cluster_medoids(f.labels,f.centroids)
    r,c=linear_sum_assignment(cdist(med,raw)); assigned=np.empty_like(med); assigned[r]=raw[c]
    moved=coords+(assigned-med)[f.labels]
    return scorer._score_single_density(scorer._density(moved))

old=pd.read_csv(OLD_FILE)
old["Method"]=old.Method.replace({"LPV":"LPM"})
checkpoint=OUT/"DI_OT_M500_vs_paper.csv"
rows=pd.read_csv(checkpoint).query("Method != 'WAV'").to_dict("records") if checkpoint.exists() else []
done={(r["Population"],r["Probability"],int(r["n"]),r["Method"]) for r in rows}
setting=0
for name,(rel,ns,methods) in POPS.items():
  for inc in ["EP","UP"]:
   for n in ns:
    setting+=1; pop,_=load_pop(name,rel,n,inc); scorer=DensityDisparity(pop,representative="nmedoids",n_jobs=1)
    for j,meth in enumerate(methods):
      if (name,inc,n,meth) in done: continue
      print(name,inc,n,meth,flush=True)
      ss=samples(METHODS[meth],pop,M,SEED+10000*setting+100*j)
      vals=np.asarray(Parallel(n_jobs=-1)(delayed(one_score)(s,pop,scorer) for s in ss))
      prob={"EP":"Equal","UP":"Unequal"}[inc]
      q=old[(old.Population==name)&(old.Probability==prob)&(old.Method==meth)&(old.n==n)]
      if len(q)!=1: raise RuntimeError(f"old row count {len(q)}: {name} {inc} {n} {meth}")
      o=q.iloc[0]
      rows.append(dict(Population=name,Probability=inc,Method=meth,n=n,M=M,OT_mean_DI=vals.mean(),OT_SD=vals.std(ddof=1),paper_mean_DI=o.Dm,paper_SD=o.Ds,mean_difference=vals.mean()-o.Dm,SD_difference=vals.std(ddof=1)-o.Ds))
      pd.DataFrame(rows).to_csv(checkpoint,index=False)

res=pd.DataFrame(rows)
res.to_csv(checkpoint,index=False)

def panel(data,pops,path,title):
 fig,axs=plt.subplots(1,len(pops),figsize=(4.2*len(pops),4),sharey=True); axs=np.atleast_1d(axs)
 for ax,p in zip(axs,pops):
  z=data[data.Population==p]
  for meth in z.Method.unique():
   for inc,ls in [("EP","-"),("UP","--")]:
    a=z[(z.Method==meth)&(z.Probability==inc)].sort_values("n")
    ax.errorbar(a.n,a.paper_mean_DI,yerr=a.paper_SD,marker="o",ls=ls,alpha=.38,label=f"{meth} {inc} old")
    ax.errorbar(a.n,a.OT_mean_DI,yerr=a.OT_SD,marker="x",ls=ls,label=f"{meth} {inc} OT")
  ax.axhline(0,color="black",lw=.7); ax.set_title(p); ax.set_xlabel("n")
 axs[0].set_ylabel("Mean DI ± SD"); handles,labels_=axs[-1].get_legend_handles_labels()
 fig.legend(handles,labels_,loc="center left",bbox_to_anchor=(1,.5),fontsize=7); fig.suptitle(title); fig.tight_layout()
 fig.savefig(path,dpi=250,bbox_inches="tight"); plt.close(fig)

panel(res[res.Population.eq("Random")],["Random"],OUT/"Random_DI_old_vs_OT_M500.png","Random DI robustness: paper versus OT")
panel(res[res.Population.eq("meuse")],["meuse"],OUT/"Meuse_DI_old_vs_OT_M500.png","Meuse DI robustness: paper versus OT")
print(res.to_string(index=False)); print("SAVED",OUT)

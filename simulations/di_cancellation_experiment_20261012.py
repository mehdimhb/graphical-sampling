"""Audit existing medoid DI; no package algorithms are overridden.
Run from repository root: MPLCONFIGDIR=/tmp/mpl-di .venv/bin/python simulations/di_cancellation_experiment_20261012.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys, json, hashlib, platform, importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit
from graphical_sampling.population import Population
from graphical_sampling.index import DensityDisparity

OUT = ROOT / 'simulations/results/di_cancellation_20261012'
SOURCE = ROOT / 'simulations/populations/simulated/clust_N144_perm.csv'
SEED = 12345
@njit
def seed_numba(seed):
    np.random.seed(seed)

def components(scorer, td):
    # Exact operations used by DensityDisparity._score_single_density.
    inv = 1.0 / (scorer._sqrt_2 * np.sqrt(scorer._orig_density_sq + td ** 2))
    d = np.mean(scorer._scale((scorer.original_density - td) * inv, scorer._sin_pi_8)).item()
    eta = np.mean(scorer._scale(1.0 - (scorer.original_density + td) * inv, scorer._one_minus_cos_pi_8)).item()
    return d, eta

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    pop = Population(raw[['x','y']].to_numpy(), np.ones(len(raw)), n=4)
    assert pop.N == 144
    scorer = DensityDisparity(pop, representative='nmedoids', n_jobs=1)
    cache = {}
    def evaluate(s):
        s = tuple(sorted(map(int, s)))
        if s in cache: return cache[s]
        # Seed both RNGs because FIPBalancedNMeans uses a randomized Numba TSP.
        rs = int(np.random.SeedSequence([SEED, *s]).generate_state(1)[0])
        np.random.seed(rs); seed_numba(rs)
        score, td = scorer._process_one_sample(np.array(s))
        d, eta = components(scorer, td)
        reconstructed = (1-eta)*d + eta*np.sign(d)
        assert abs(score-reconstructed) < 5e-15
        row = dict(Dnet=d, eta=eta, DI=score, formula_error=score-reconstructed, pipeline_seed=rs)
        for j, i in enumerate(s, 1):
            row.update({f'index_{j}':i, f'id_{j}':int(pop.ids[i]), f'x_{j}':pop.coords[i,0], f'y_{j}':pop.coords[i,1], f'raw_x_{j}':raw.x.iloc[i], f'raw_y_{j}':raw.y.iloc[i]})
        cache[s] = row
        return row
    rng = np.random.default_rng(SEED)
    samples = np.array([np.sort(rng.choice(pop.N, 4, replace=False)) for _ in range(5000)])
    base = []
    for i,s in enumerate(samples):
        base.append(dict(draw=i+1, **evaluate(s)))
        if (i+1)%500 == 0: print('random draws', i+1, flush=True)
    df = pd.DataFrame(base)
    df.to_csv(OUT/'random_samples.csv', index=False, float_format='%.17g')
    # Exhaust every one-unit substitution for six high-eta near-zero anchors,
    # taking three anchors from each sign; this is targeted, not random inference.
    eligible = df[df.eta > .3].copy()
    anchors = pd.concat([eligible[eligible.Dnet > 0].assign(absD=lambda z:abs(z.Dnet)).nsmallest(3,'absD'), eligible[eligible.Dnet < 0].assign(absD=lambda z:abs(z.Dnet)).nsmallest(3,'absD')])
    pairs=[]; local=[]
    for _,a in anchors.iterrows():
        s=tuple(int(a[f'index_{j}']) for j in range(1,5))
        for removed in s:
            for added in range(pop.N):
                if added in s: continue
                t=tuple(sorted((set(s)-{removed})|{added}))
                b=evaluate(t)
                distance=float(np.linalg.norm(pop.coords[removed]-pop.coords[added]))
                local.append(dict(anchor_draw=int(a.draw), removed_id=int(pop.ids[removed]), added_id=int(pop.ids[added]), replacement_distance=distance, **b))
                if a.Dnet*b['Dnet'] < 0 and b['eta'] > .3:
                    pairs.append(dict(anchor_draw=int(a.draw), removed_id=int(pop.ids[removed]), added_id=int(pop.ids[added]), replacement_distance=distance, Dnet_A=a.Dnet, eta_A=a.eta, DI_A=a.DI, Dnet_B=b['Dnet'], eta_B=b['eta'], DI_B=b['DI'], max_abs_Dnet=max(abs(a.Dnet),abs(b['Dnet'])), DI_gap=abs(a.DI-b['DI']), sample_A=json.dumps(s), sample_B=json.dumps(t)))
        print('local anchor',int(a.draw),'done',flush=True)
    pd.DataFrame(local).to_csv(OUT/'local_search.csv',index=False,float_format='%.17g')
    pairdf=pd.DataFrame(pairs)
    if len(pairdf): pairdf=pairdf.sort_values(['max_abs_Dnet','replacement_distance'])
    pairdf.to_csv(OUT/'opposite_sign_pairs.csv',index=False,float_format='%.17g')
    relevant=eligible.assign(absD=lambda z:abs(z.Dnet)).nsmallest(20,'absD')
    relevant.to_csv(OUT/'near_cancellation.csv',index=False,float_format='%.17g')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax in axs:
        ax.scatter(df.Dnet,df.eta,s=7,alpha=.3)
        ax.axvspan(-.001,.001,color='red',alpha=.2,label='|Dnet| < 0.001')
        ax.axhline(.3,color='red',ls='--',lw=.8); ax.axvline(0,color='black',lw=.6)
        ax.set(xlabel='Dnet',ylabel='η'); ax.legend(fontsize=8)
    axs[1].set_xlim(-.01,.01); axs[1].set_title('Near cancellation (zoom)')
    fig.tight_layout(); fig.savefig(OUT/'eta_vs_Dnet.png',dpi=220); plt.close(fig)
    illustrated=[]
    if len(pairdf):
        # Also select the geographically closest sign flip with both |D| < .01.
        choices=[pairdf.iloc[0]]
        close=pairdf[pairdf.max_abs_Dnet < .01]
        if len(close): choices.append(close.nsmallest(1,'replacement_distance').iloc[0])
        for k,p in enumerate(choices,1):
            fig,axs=plt.subplots(1,2,figsize=(10,4))
            for ax,key in zip(axs,['A','B']):
                s=json.loads(p[f'sample_{key}']); xy=pop.coords[s]
                ax.scatter(*pop.coords.T,s=10,c='lightgray'); ax.scatter(*xy.T,s=75,c='crimson')
                for i in s: ax.annotate(str(pop.ids[i]),pop.coords[i],xytext=(3,3),textcoords='offset points')
                ax.set_aspect('equal'); ax.set_title(f"{key}: Dnet={p[f'Dnet_{key}']:.9g}\nη={p[f'eta_{key}']:.9g}, DI={p[f'DI_{key}']:.9g}")
            fig.suptitle(f"One unit replaced; normalized distance={p.replacement_distance:.9g}")
            fig.tight_layout(); fig.savefig(OUT/f'spatial_pair_{k}.png',dpi=220); plt.close(fig)
            illustrated.append(p.to_dict())
    bins=[0,.001,.005,.01,.05,.1,1.01]
    joint=df.assign(absD=abs(df.Dnet),bin=pd.cut(abs(df.Dnet),bins,right=False)).groupby('bin',observed=False).agg(count=('DI','size'),eta_max=('eta','max'),eta_median=('eta','median'),high_eta=('eta',lambda x:int((x>.3).sum())))
    joint.to_csv(OUT/'joint_distribution.csv',float_format='%.17g')
    manifest=dict(seed=SEED,N=pop.N,n=4,M=5000,source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),python=platform.python_version(),versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','numba','graphical-sampling']},unique_random_samples=len(set(map(tuple,samples))),high_eta_count=len(eligible),threshold_count=int(((abs(df.Dnet)<.001)&(df.eta>.3)).sum()),min_abs_Dnet_high_eta=float(abs(eligible.Dnet).min()) if len(eligible) else None,max_formula_error=float(abs(df.formula_error).max()),local_evaluations=len(local),unique_pipeline_evaluations=len(cache),illustrated_pairs=illustrated)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(OUT/'population_and_samples.npz',coords=pop.coords,raw_coords=raw[['x','y']].to_numpy(),ids=pop.ids,inclusions=pop.inclusions,samples=samples)
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()

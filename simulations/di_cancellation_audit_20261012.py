"""Independent artifact and illustrative-pair verification, using package calls."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from di_cancellation_experiment_20261012 import *
from sklearn.neighbors import KernelDensity

def main():
    df=pd.read_csv(OUT/'random_samples.csv',float_precision='round_trip')
    manifest=json.loads((OUT/'manifest.json').read_text())
    raw=pd.read_csv(SOURCE)
    pop=Population(raw[['x','y']].to_numpy(),np.ones(144),n=4)
    scorer=DensityDisparity(pop,representative='nmedoids',n_jobs=1)
    rng=np.random.default_rng(12345)
    expected=np.array([np.sort(rng.choice(144,4,replace=False)) for _ in range(5000)])
    np.testing.assert_array_equal(df[[f'index_{j}' for j in range(1,5)]],expected)
    assert (np.diff(expected,axis=1)>0).all()
    np.testing.assert_allclose(df.DI,(1-df.eta)*df.Dnet+df.eta*np.sign(df.Dnet),atol=5e-15,rtol=0)
    assert ((df.eta>=-1e-14)&(df.eta<=1)).all()
    checks=[]
    for k,p in enumerate(manifest['illustrated_pairs'],1):
        for key in ['A','B']:
            s=np.array(json.loads(p[f'sample_{key}']))
            saved_seed=int(np.random.SeedSequence([SEED,*map(int,s)]).generate_state(1)[0])
            for run_seed in [saved_seed,*range(10)]:
                np.random.seed(run_seed); seed_numba(run_seed)
                score,td=scorer._process_one_sample(s)
                d,eta=components(scorer,td)
                if run_seed==saved_seed:
                    assert score==p[f'DI_{key}']
                    assert d==p[f'Dnet_{key}']
                    np.savez_compressed(OUT/f'pair_{k}_{key}_densities.npz',original=scorer.original_density,translated=td,sample_indices=s)
                    # Independent trigonometric equivalent; actual KDE values only.
                    theta=np.arctan2(td,scorer.original_density)
                    da=np.mean(np.clip(np.sin(np.pi/4-theta)/np.sin(np.pi/8),-1,1))
                    ea=np.mean(np.clip((1-np.cos(theta-np.pi/4))/(1-np.cos(np.pi/8)),-1,1))
                    assert abs(d-da)<5e-15 and abs(eta-ea)<5e-15
                checks.append(dict(pair=k,sample=key,seed=run_seed,Dnet=d,eta=eta,DI=score))
    checkdf=pd.DataFrame(checks)
    checkdf.to_csv(OUT/'rng_sensitivity.csv',index=False,float_format='%.17g')
    bw=KernelDensity(kernel='tophat',bandwidth='scott',rtol=1e-4).fit(pop.coords).bandwidth_
    audit=dict(artifact_checks='passed',random_draws=5000,identity_max_error=float(abs(df.DI-((1-df.eta)*df.Dnet+df.eta*np.sign(df.Dnet))).max()),scott_bandwidth=float(bw),expected_scott=float(144**(-1/6)),sensitivity_summary=checkdf.groupby(['pair','sample']).agg(Dnet_min=('Dnet','min'),Dnet_max=('Dnet','max'),DI_min=('DI','min'),DI_max=('DI','max')).reset_index().to_dict('records'))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()

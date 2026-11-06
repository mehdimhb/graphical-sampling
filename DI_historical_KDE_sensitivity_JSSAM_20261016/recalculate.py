"""Density-only rescoring of verified saved references; safely resumable.

No import of the GFS package, original runner, pipeline or optimization code.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
SOURCE=ROOT/'DI_reference_sensitivity_JSSAM'
SCENARIOS=('C16','M10','S100')
RTOL=1e-4


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');tmp=Path(handle.name)
    os.replace(tmp,path)


def atomic_csv(path,frame):
    tmp=Path(str(path)+'.tmp');frame.to_csv(tmp,index=False,float_format='%.17g');os.replace(tmp,path)


def quantities(f,g):
    theta=np.pi/8
    den=np.sqrt(2)*np.sqrt(f*f+g*g)
    assert np.isfinite(den).all() and (den>0).all()
    q=np.clip((f-g)/den/np.sin(theta),-1.,1.)
    e=np.clip((1-(f+g)/den)/(1-np.cos(theta)),-1.,1.)
    Dnet=float(q.mean());eta=float(e.mean())
    DI=float((1-eta)*Dnet+eta*np.sign(Dnet))
    return DI,Dnet,eta,q,e


def fit_density(points,rtol):
    kde=KernelDensity(kernel='tophat',bandwidth='scott',rtol=rtol,atol=0).fit(points)
    values=np.exp(kde.score_samples(points))
    assert np.isfinite(values).all() and (values>0).all()
    return values,float(kde.bandwidth_)


def checkpoint(preflight):
    records=[]
    for entry in preflight['selected']:
        path=OUT/'evaluations'/entry['run_id']/'summary.json'
        if path.exists():records.append(json.loads(path.read_text()))
    df=pd.DataFrame(records)
    primary=df[df.experiment.isin(list('ABC'))].sort_values(['scenario','experiment','index']) if len(df) else df
    bases=df[df.experiment=='baseline'].sort_values('scenario') if len(df) else df
    atomic_csv(OUT/'results.csv',primary);atomic_csv(OUT/'baselines.csv',bases)
    groups={f'{sid}_{exp}':int(((primary.scenario==sid)&(primary.experiment==exp)).sum()) if len(primary) else 0
            for sid in SCENARIOS for exp in 'ABC'}
    atomic_json(OUT/'progress.json',dict(stage='density_recalculation' if len(primary)<360 else 'density_recalculation_complete',
        primary_completed=len(primary),primary_planned=360,percent=100*len(primary)/360,
        baselines_completed=len(bases),group_counts=groups,reference_optimizations_run=0,
        clustering_runs=0,TSP_searches=0,excluded_scenario='C32'))
    return len(primary)


def main():
    preflight=json.loads((OUT/'preflight.json').read_text());assert preflight['sufficient']
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    cases={c['scenario']:c for c in protocol['scenarios'] if c['scenario'] in SCENARIOS}
    config=dict(scenarios=list(SCENARIOS),primary_evaluations=360,baseline_controls=3,
        kernel='spherical uniform / radial tophat',bandwidth='N**(-1/(d+4))',
        kernel_metric='euclidean',rtol=RTOL,atol=0.,default_rtol_cross_check=0.,
        coordinate_normalization='(raw - raw.min(axis=0)) / np.ptp(raw,axis=0).max()',
        no_renormalization_after_translation=True,KDE_weights='equal, not inclusion-probability weights',
        theta0='pi/8',clipping='[-1,1] for each q_i and e_i as in existing definitions',
        DI='(1-eta)*Dnet + eta*np.sign(Dnet)',zero_tolerance='none',
        normalization_source_sha256=sha(ROOT/'DI_historical_implementation_audit_20261016/evidence/e71d6935b_population.py'),
        historical_KDE_source_sha256=sha(ROOT/'DI_historical_implementation_audit_20261016/evidence/e71d6935b_density_disparity.py'),
        preflight_SHA256=sha(OUT/'preflight.json'),
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','pandas')})
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cp=OUT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:atomic_json(cp,config)
    source_rows=pd.read_csv(SOURCE/'results.csv').set_index('run_id')
    quality=pd.read_csv(SOURCE/'analysis_results.csv').set_index('run_id')
    original={};completed=0;start=time.monotonic()
    for entry in preflight['selected']:
        rid=entry['run_id'];sid=entry['scenario'];folder=OUT/'evaluations'/rid
        saved=folder/'summary.json'
        assert sid in SCENARIOS and sha(ROOT/entry['state'])==entry['state_SHA256']
        assert sha(ROOT/entry['summary'])==entry['summary_SHA256']
        if saved.exists():
            old=json.loads(saved.read_text())
            assert old['config_SHA256']==config_hash and old['state_SHA256']==entry['state_SHA256']
            with np.load(folder/'densities.npz',allow_pickle=False) as cached:
                checks=quantities(cached['f'],cached['g'])[:3]
                assert all(abs(value-old[key])<2e-15 for value,key in zip(checks,('DI','Dnet','eta')))
            continue
        assert sid in SCENARIOS and sha(ROOT/entry['state'])==entry['state_SHA256']
        assert sha(ROOT/entry['summary'])==entry['summary_SHA256']
        row=json.loads((ROOT/entry['summary']).read_text())
        folder.mkdir(parents=True,exist_ok=True)
        with np.load(ROOT/entry['state'],allow_pickle=False) as z:
            x=z['coords'];moved=z['moved'];N,d=x.shape
            assert np.array_equal(moved,x+z['translations'][z['labels']])
            assert np.array_equal(z['translations'],x[z['matched']]-x[z['medoids']])
            if sid not in original:
                f,h=fit_density(x,RTOL);f_exact,h0=fit_density(x,0.)
                assert h==h0==N**(-1/(d+4))
                original[sid]=(x.copy(),f,h,f_exact)
            reference_x,f,h,f_exact=original[sid]
            assert np.array_equal(reference_x,x)
            g,h_g=fit_density(moved,RTOL);g_exact,h_g0=fit_density(moved,0.)
            assert h_g==h_g0==h
            DI,Dnet,eta,q,e=quantities(f,g)
            exact_DI,exact_Dnet,exact_eta,_,_=quantities(f_exact,g_exact)
            old_DI,old_Dnet,old_eta,_,_=quantities(z['f'],z['g'])
            assert abs(old_DI-row['DI'])<2e-14 and abs(old_Dnet-row['Dnet'])<2e-14 and abs(old_eta-row['eta'])<2e-14
            if row['experiment'] in 'ABC':
                assert abs(row['DI']-source_rows.loc[rid,'DI'])<2e-15
                assert abs(row['DI']-quality.loc[rid,'DI'])<2e-15
            result={k:row[k] for k in ('run_id','scenario','population','N','n','IP','experiment','index',
                'OT_id','tsp_seed','tsp_restarts','TSP_length','reference_hash','medoid_set_hash')}
            result.update(KDE='historical_spherical_tophat_scalar_Scott',h=h,h_translated=h_g,
                DI=DI,Dnet=Dnet,eta=eta,old_box_DI=row['DI'],old_box_Dnet=row['Dnet'],old_box_eta=row['eta'],
                DI_historical_minus_box=DI-row['DI'],Dnet_historical_minus_box=Dnet-row['Dnet'],
                eta_historical_minus_box=eta-row['eta'],
                sign_changed_from_box=bool(np.sign(DI)!=np.sign(row['DI'])),
                strict_sign_reversal_from_box=bool(DI*row['DI']<0),
                zero_transition_from_box=bool((DI==0)!=(row['DI']==0)),
                f_min=float(f.min()),g_min=float(g.min()),f_zero_count=int((f==0).sum()),g_zero_count=int((g==0).sum()),
                f_g_identical=bool(np.array_equal(f,g)),abs_Dnet_below_1e_12=bool(abs(Dnet)<1e-12),
                default_rtol0_DI=exact_DI,default_rtol0_Dnet=exact_Dnet,default_rtol0_eta=exact_eta,
                rtol_DI_difference=DI-exact_DI,
                max_rtol_density_difference=float(max(abs(f-f_exact).max(),abs(g-g_exact).max())),
                default_rtol0_sign_disagreement=bool(np.sign(DI)!=np.sign(exact_DI)),
                selection_seed=cases[sid]['selection_seed'],
                state_SHA256=entry['state_SHA256'],summary_SHA256=entry['summary_SHA256'],
                moved_array_SHA256=hashlib.sha256(moved.tobytes()).hexdigest(),
                coords_array_SHA256=hashlib.sha256(x.tobytes()).hexdigest(),config_SHA256=config_hash)
            if rid in quality.index:
                result.update({key:float(quality.loc[rid,key]) for key in ('native_OT_excess','common_OT_excess','TSP_excess',
                    'aligned_hard_changes','medoid_set_changes','centroid_geometric_max_distance',
                    'permutation_aligned_coupling_max_difference')})
            temp=folder/'densities.tmp.npz'
            np.savez_compressed(temp,f=f,g=g,q=q,e=e,h=h,H=h*h*np.eye(d),
                                f_default_rtol0=f_exact,g_default_rtol0=g_exact)
            os.replace(temp,folder/'densities.npz')
            atomic_json(saved,result)
        count=checkpoint(preflight);completed+=1
        if completed%5==0 or row['experiment']=='baseline':
            (OUT/'HANDOFF.md').write_text(f'''# Historical KDE sensitivity handoff

Density-only rescoring in progress: {count}/360 primary evaluations. Three scenarios only: C16, M10, S100. Preflight verified all saved references, normalization and translations; no optimization is needed or run. Source hashes and selected-state fingerprints are frozen. New per-evaluation density arrays and summary.json are durable; results.csv/progress.json refresh after each completion.

Resume from project root:
```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python DI_historical_KDE_sensitivity_JSSAM_20261016/recalculate.py
```
Completed evaluation IDs are skipped and validated against frozen input/config hashes. Original study, package and manuscripts are read only. Never invoke pipeline.evaluate or an original runner. Remaining stages: independent density/statistics verification, comparison report and separate proposed Appendix artifacts. C32 is excluded.
''')
            print(f'{count}/360 primary, {completed} new saved evaluations incl baselines, {time.monotonic()-start:.1f}s',flush=True)
    checkpoint(preflight)
    print('Density-only calculation complete; 360 primary plus three baseline controls.',flush=True)


if __name__=='__main__':main()

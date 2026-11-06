"""Independent saved-density/count and individual-row summary verification.

Never imports recalculate.py, summarize.py, GFS or any reference optimizer.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from scipy.spatial.distance import cdist

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
SOURCE=ROOT/'DI_reference_sensitivity_JSSAM'


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def independent_score(f,g):
    # Extended precision and independent reduction, same declared mathematical definition.
    f=np.asarray(f,np.longdouble);g=np.asarray(g,np.longdouble)
    theta=np.longdouble(str(np.pi))/8
    norm=np.sqrt(np.longdouble(2))*np.sqrt(f*f+g*g)
    q=np.minimum(np.longdouble(1),np.maximum(-np.longdouble(1),(f-g)/(norm*np.sin(theta))))
    e=np.minimum(np.longdouble(1),np.maximum(-np.longdouble(1),(1-(f+g)/norm)/(1-np.cos(theta))))
    dn=np.mean(q,dtype=np.longdouble);eta=np.mean(e,dtype=np.longdouble)
    return float((1-eta)*dn+eta*np.sign(dn)),float(dn),float(eta)


def main():
    preflight=json.loads((OUT/'preflight.json').read_text())
    primary=read('results.csv');baselines=read('baselines.csv')
    assert len(primary)==360 and len(baselines)==3
    rows={r['run_id']:r for r in primary+baselines}
    assert len(rows)==363 and {r['scenario'] for r in rows.values()}=={'C16','M10','S100'}
    assert {r['experiment'] for r in primary}==set('ABC')
    # Execute only the original pure normalization function, not any package imports.
    historical=ROOT/'DI_historical_implementation_audit_20261016/evidence/e71d6935b_population.py'
    tree=ast.parse(historical.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Population')
    fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_normalize_coords')
    fn.decorator_list=[];env={'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(historical),'exec'),env)
    expected_x={}
    for sid in ('C16','M10','S100'):
        with np.load(SOURCE/'frozen_inputs'/f'{sid}.npz',allow_pickle=False) as z:
            expected_x[sid]=env['_normalize_coords'](z['raw_coords'])
    audits=[];original_cache={};max_density_error=0.;max_score_error=0.;max_old_score_error=0.
    for entry in preflight['selected']:
        rid=entry['run_id'];row=rows[rid]
        state_path=ROOT/entry['state'];assert hashlib.sha256(state_path.read_bytes()).hexdigest()==entry['state_SHA256']
        assert row['state_SHA256']==entry['state_SHA256'] and row['reference_hash']==entry['reference_hash']
        assert int(row['tsp_seed'])==entry['tsp_seed']
        with np.load(state_path,allow_pickle=False) as source, np.load(OUT/'evaluations'/rid/'densities.npz',allow_pickle=False) as saved:
            x=source['coords'];moved=source['moved'];N,d=x.shape;h=float(row['h'])
            assert np.array_equal(x,expected_x[row['scenario']])
            assert h==float(row['h_translated'])==N**(-1/(d+4))
            assert np.array_equal(moved,x+(x[source['matched']]-x[source['medoids']])[source['labels']])
            volume=math.pi**(d/2)/math.gamma(1+d/2)
            counts=[];boundary=[];direct=[]
            for which,points in enumerate((x,moved)):
                key=(row['scenario'],which)
                if which==0 and key in original_cache:
                    count,margin,density=original_cache[key]
                else:
                    distances=cdist(points,points,'euclidean')
                    count=np.sum(distances<h,axis=1)
                    margin=float(np.min(np.abs(distances-h)))
                    density=count/(N*volume*h**d)
                    if which==0:original_cache[key]=(count,margin,density)
                counts.append(count);boundary.append(margin);direct.append(density)
            for actual,expected in zip((saved['f'],saved['g']),direct):
                error=float(np.max(abs(actual-expected)));max_density_error=max(max_density_error,error)
                np.testing.assert_allclose(actual,expected,rtol=5e-14,atol=1e-14)
            assert all(count.min()>=1 for count in counts)
            di,dn,eta=independent_score(*direct)
            error=max(abs(di-float(row['DI'])),abs(dn-float(row['Dnet'])),abs(eta-float(row['eta'])))
            max_score_error=max(max_score_error,error);assert error<3e-13
            assert np.sign(di)==np.sign(float(row['DI']))
            # Original box arrays independently match their saved direct counts and summaries.
            old_dens=[]
            for points,key in ((x,'original_counts'),(moved,'translated_counts')):
                white=points@source['invroot']
                count=np.sum(cdist(white,white,'chebyshev')<=1.,axis=1)
                np.testing.assert_array_equal(count,source[key])
                old_dens.append(count/(N*2**d*np.sqrt(np.linalg.det(source['H']))))
            old_DI,old_dn,old_eta=independent_score(*old_dens)
            old_error=max(abs(old_DI-float(row['old_box_DI'])),abs(old_dn-float(row['old_box_Dnet'])),abs(old_eta-float(row['old_box_eta'])))
            max_old_score_error=max(max_old_score_error,old_error);assert old_error<3e-13
            assert np.array_equal(saved['f'],saved['f_default_rtol0'])
            assert np.array_equal(saved['g'],saved['g_default_rtol0'])
            assert float(row['max_rtol_density_difference'])==0 and float(row['rtol_DI_difference'])==0
            audits.append(dict(run_id=rid,maximum_density_error=max(float(abs(a-b).max()) for a,b in zip((saved['f'],saved['g']),direct)),
                longdouble_score_error=error,old_box_longdouble_score_error=old_error,
                minimum_radius_boundary_margin=min(boundary),f_min_count=int(counts[0].min()),g_min_count=int(counts[1].min()),
                sign_verified=True,rtol0_density_arrays_identical=True))
    bases={r['scenario']:r for r in baselines}
    summary=read('summary.csv');assert len(summary)==9
    checked_statistics=0
    for s in summary:
        group=[r for r in primary if r['scenario']==s['scenario'] and r['experiment']==s['experiment']]
        assert len(group)==40 and {int(r['index']) for r in group}==set(range(1,41))
        b=bases[s['scenario']];values=[float(r['DI']) for r in group];old=[float(r['old_box_DI']) for r in group]
        delta=[v-float(b['DI']) for v in values];old_delta=[v-float(b['old_box_DI']) for v in old]
        expected=dict(DI_mean=math.fsum(values)/40,DI_sd=statistics.stdev(values),DI_min=min(values),DI_max=max(values),
            mean_delta_from_baseline=math.fsum(delta)/40,MAD_from_baseline=math.fsum(map(abs,delta))/40,
            max_abs_delta_from_baseline=max(map(abs,delta)),old_box_DI_sd=statistics.stdev(old),old_box_DI_min=min(old),old_box_DI_max=max(old),
            old_box_MAD_from_baseline=math.fsum(map(abs,old_delta))/40,old_box_max_abs_delta_from_baseline=max(map(abs,old_delta)),
            sign_differences_from_baseline=sum(np.sign(v)!=np.sign(float(b['DI'])) for v in values),
            strict_sign_reversals_from_baseline=sum(v*float(b['DI'])<0 for v in values),
            old_box_sign_differences_from_baseline=sum(np.sign(v)!=np.sign(float(b['old_box_DI'])) for v in old),
            sign_changes_between_KDEs=sum(np.sign(v)!=np.sign(o) for v,o in zip(values,old)),
            strict_reversals_between_KDEs=sum(v*o<0 for v,o in zip(values,old)),
            zero_transitions_between_KDEs=sum((v==0)!=(o==0) for v,o in zip(values,old)),
            mean_DI_change_between_KDEs=math.fsum(v-o for v,o in zip(values,old))/40,
            max_abs_DI_change_between_KDEs=max(abs(v-o) for v,o in zip(values,old)),
            reference_count=len({r['reference_hash'] for r in group}),translated_population_count=len({r['moved_array_SHA256'] for r in group}),
            Dnet_min=min(float(r['Dnet']) for r in group),Dnet_max=max(float(r['Dnet']) for r in group),
            eta_min=min(float(r['eta']) for r in group),eta_max=max(float(r['eta']) for r in group))
        for key,value in expected.items():assert abs(float(s[key])-value)<3e-14,(s['scenario'],s['experiment'],key)
        checked_statistics+=len(expected)
    pairwise=read('pairwise.csv');assert len(pairwise)==7380
    for p in pairwise:
        a,b=rows[p['run_a']],rows[p['run_b']]
        assert abs(float(p['absolute_DI_difference'])-abs(float(a['DI'])-float(b['DI'])))<2e-15
        for tol in (.001,.01,.02):
            valid=max(float(p[k+'_'+side]) for k in ('native_OT_excess','common_OT_excess','TSP_excess') for side in ('a','b'))<=tol
            assert (p['both_nearbest_'+str(tol)]=='True')==valid
    for row in read('near_best_summary.csv'):
        tol=float(row['tolerance'])
        eligible=[p for p in pairwise if p['scenario']==row['scenario'] and p['experiment']==row['experiment']
                  and p['distinct_translation']=='True' and p['both_nearbest_'+str(tol)]=='True']
        assert len(eligible)==int(row['distinct_translation_pairs'])
        if eligible:assert abs(max(float(p['absolute_DI_difference']) for p in eligible)-float(row['max_abs_DI_difference']))<2e-15
        else:assert row['max_abs_DI_difference']==''
    with (OUT/'verification.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audits[0]));writer.writeheader();writer.writerows(audits)
    protected=json.loads((OUT/'source_hashes_before.json').read_text())
    changed=[name for name,sha in protected.items() if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha]
    assert not changed,changed
    result=dict(all_passed=True,primary_evaluations_verified=360,baseline_controls_verified=3,
        group_statistics_verified=checked_statistics,pairwise_comparisons_verified=len(pairwise),
        protected_files_verified=len(protected),changed_original_files=changed,
        maximum_direct_density_error=max_density_error,maximum_longdouble_score_error=max_score_error,
        maximum_old_box_score_error=max_old_score_error,
        minimum_radius_boundary_margin=min(r['minimum_radius_boundary_margin'] for r in audits),
        rtol0_crosscheck='density arrays bitwise identical for all363 states',
        zero_densities=0,nonfinite_scores=0,normalization_matches_historical_pure_function=True,
        OT_clustering_TSP_reruns=0,excluded_scenario_evaluations=0)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

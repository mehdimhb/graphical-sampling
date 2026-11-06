import numpy as np
from experiment import balance_metrics

def test_known_two_clusters():
    d=balance_metrics(np.array([.7,.4,.5,.4]),np.array([0,0,1,1]),2)
    assert np.allclose(d['mass'],[1.1,.9])
    assert np.array_equal(d['size'],[2,2])
    assert np.isclose(d['mean_abs_error'],.1)
    assert np.isclose(d['sd_mass'],.1)
    assert d['pct_gt_025']==0 and d['bound_pass']

def test_empty_clusters_are_retained():
    d=balance_metrics(np.array([1.,1.]),np.array([0,0]),2)
    assert np.array_equal(d['size'],[2,0])
    assert d['empty_clusters']==[1]
    assert d['max_abs_error']==1 and d['bound_pass']

def test_thresholds_use_mathematical_comparison():
    # Exactly .25 must not be counted as strictly greater.
    d=balance_metrics(np.array([.75,.25,.5,.5]),np.array([0,1,1,1]),2)
    assert d['pct_gt_010']==100 and d['pct_gt_025']==0

def test_worker_control_reduces_running_count_without_pause_oscillation():
    from experiment import worker_control
    # Background load raises total CPU above the target; existing workers must pause.
    target,full=worker_control(8,8,59.,54.,30.,False)
    assert target==7 and not full
    # A hard-ceiling pause must not make low measured CPU immediately restore workers.
    target,full=worker_control(7,8,12.,41.,30.,True)
    assert target==7 and not full
    # At the desired measured CPU, hold six runnable workers despite a low rolling mean.
    target,full=worker_control(6,8,52.,42.,30.,False)
    assert target==6 and not full
    target,full=worker_control(8,8,64.,54.,30.,False)
    assert target==7 and full

def test_reused_OT_keeps_current_case_identity(tmp_path,monkeypatch):
    import experiment,json,shutil
    from pathlib import Path
    home=Path(experiment.HOME)
    case=next(c for c in json.loads((home/'design.json').read_text())['cases'] if c['case_id']=='Clustered_n16_EP_r1')
    shutil.copytree(home/'populations',tmp_path/'populations')
    shutil.copytree(home/case['cache'],tmp_path/case['cache'])
    monkeypatch.setattr(experiment,'HOME',tmp_path)
    result=experiment.run_case(case)
    assert result['status']=='completed'
    assert result['case_id']==case['case_id']
    assert 'J_OT_native' not in result

def test_exact_half_ties_can_create_an_empty_hard_cluster_and_are_reported():
    # A unit-mass middle interval lies between two exact half-unit border shares.
    # Both borders choose lower persistent IDs outside the middle cluster.
    from adapter import refine
    pi=np.array([.5,1.,1.,.5]);coords=np.array([[0.,0.],[.1,0.],[.9,0.],[1.,0.]])
    result=refine(np.array([0,1,2]),np.array([0,0,2,2]),coords,pi,np.arange(1,5),
                  np.array([[0.,0.],[.5,0.],[1.,0.]]),np.array([1,4,3]),mode='corrected')
    metrics=balance_metrics(pi,result['labels'],3)
    assert np.allclose(pi@result['mu'],1)
    assert metrics['empty_clusters']==[1]
    assert np.array_equal(metrics['size'],[2,0,2])
    assert np.array_equal(metrics['mass'],[1.5,0.,1.5])
    assert metrics['bound_pass'] and metrics['max_abs_error']==1

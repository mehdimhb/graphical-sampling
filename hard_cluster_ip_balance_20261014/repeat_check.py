"""Repeat complete small UP pipelines without overwriting the delivered experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import shutil,json
import numpy as np
import experiment
HOME=Path(__file__).resolve().parent

def run():
    original=HOME/'runs';target=HOME/'repeatability_runs';target.mkdir(exist_ok=True)
    if not (target/'populations').exists():shutil.copytree(HOME/'populations',target/'populations')
    cases=json.loads((HOME/'design.json').read_text())['cases'];checks=[]
    # One small UP example from each population where the first run converged quickly.
    for name in ['Gridded','Random','Clustered','Meuse','Swiss']:
        case=next(c for c in cases if c['population']==name and c['IP']=='UP' and c['replicate']==1)
        f=original/case['case_id']/'state.npz'
        if not f.exists():continue
        summary=json.loads((f.parent/'summary.json').read_text())
        if summary['cpu_seconds']>40:continue
        experiment.HOME=target
        result=experiment.run_case(case);assert result['status']=='completed',result
        first=np.load(f);second=np.load(target/'runs'/case['case_id']/'state.npz')
        identical=all(np.array_equal(first[key],second[key]) for key in first.files)
        assert identical,case['case_id']
        checks.append(dict(case_id=case['case_id'],all_arrays_bit_identical=True,array_count=len(first.files),repeat_cpu_seconds=result['cpu_seconds']))
    experiment.HOME=HOME
    assert checks
    (HOME/'repeatability.json').write_text(json.dumps(dict(fixed_seed_full_pipeline_checks=checks,all_passed=True),indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':run()

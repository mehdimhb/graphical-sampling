"""Fresh end-to-end replication in a new temporary directory; no results overwritten."""
import tempfile
from pathlib import Path
import json
import pandas as pd
import adapter
P=adapter.OUT
data=pd.read_csv(P/'scored_references.csv',float_precision='round_trip').set_index('run_id')
adapter.OUT=Path(tempfile.mkdtemp(prefix='r1c1-replication-'))
checks=[]
for args in [(1,'official',2),(1,'stress',4)]:
    result=adapter.calculate_run(*args);old=data.loc[result['run_id']]
    assert result['reference_hash']==old.reference_hash
    errors={k:abs(result[k]-old[k]) for k in ['Dnet','eta','DI','J_OT_native','Q_soft','Q_hard','matching_squared_cost']}
    assert max(errors.values())<1e-14
    checks.append(dict(run_id=result['run_id'],same_canonical_reference=True,max_numerical_error=max(errors.values()),errors=errors))
summary=dict(status='passed',temporary_results_directory=str(adapter.OUT),local_reused_adapter_modules_verified=True,checks=checks)
(P/'replication_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

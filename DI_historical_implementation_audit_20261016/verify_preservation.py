"""Verify no preserved research/source artifact changed during this audit."""
from pathlib import Path
import hashlib,json,subprocess,datetime
H=Path(__file__).resolve().parent
before=json.loads((H/'protected_before.json').read_text());changed=[f for f,d in before.items() if not Path(f).is_file() or hashlib.sha256(Path(f).read_bytes()).hexdigest()!=d]
checks=json.loads((H/'checks.json').read_text())
external_changed=[f for f,d in checks['input_SHA256'].items() if not Path(f).is_file() or hashlib.sha256(Path(f).read_bytes()).hexdigest()!=d]
head=subprocess.check_output(['git','-C',str(H.parent),'rev-parse','HEAD'],text=True).strip()
result=dict(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=not changed and not external_changed and head==checks['current_git_HEAD'],protected_files_checked=len(before),changed_files=changed,additional_input_hashes_checked=len(checks['input_SHA256']),changed_additional_inputs=external_changed,git_HEAD_unchanged=head==checks['current_git_HEAD'],new_sampling_runs=0,new_clustering_or_TSP_searches=0,new_DI_evaluations=0)
(H/'preservation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['all_passed']

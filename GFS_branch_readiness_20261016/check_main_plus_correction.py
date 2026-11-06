"""Test exact main plus correction patch in an isolated non-Git export."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
report_path=HERE/'main_plus_correction.json'
if report_path.exists():
    raise FileExistsError('Preserving completed candidate evidence')

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)

main=git('rev-parse','origin/main').decode().strip()
correction=git('rev-parse','902ed8ef4').decode().strip()
assert not git('diff','--name-only',main,correction+'^','--',
    'graphical_sampling/population.py','graphical_sampling/clustering/fip_balanced_nmeans.py',
    'graphical_sampling/index/density_disparity.py').strip()
snapshot=Path(tempfile.mkdtemp(prefix='gfs-main-di-only-'))/'snapshot'
snapshot.mkdir()
archive=subprocess.Popen(['git','archive',main,'graphical_sampling','pyproject.toml'],
                         cwd=ROOT,stdout=subprocess.PIPE)
with tarfile.open(fileobj=archive.stdout,mode='r|') as tar:
    tar.extractall(snapshot,filter='data')
assert archive.wait()==0
patch=git('diff','--binary',correction+'^',correction)
subprocess.run(['git','apply','--check'],cwd=snapshot,input=patch,check=True)
subprocess.run(['git','apply'],cwd=snapshot,input=patch,check=True)
env=os.environ.copy()
env.update({k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')})
env['PYTHONPATH']=str(snapshot)
env['MPLCONFIGDIR']=str(snapshot.parent/'matplotlib')
python=str(ROOT/'.venv/bin/python')
imported=subprocess.check_output([python,'-c','import graphical_sampling; print(graphical_sampling.__file__)'],
                                cwd=snapshot,env=env,text=True).strip()
assert Path(imported).is_relative_to(snapshot)
result=dict(main_SHA=main,correction_commit=correction,snapshot=str(snapshot),imported_package=imported,
            patch_applies_cleanly=True,changed_paths=git('diff','--name-only',correction+'^',correction).decode().splitlines(),
            patch_SHA256=hashlib.sha256(patch).hexdigest(),test_status='running',
            test_command=[python,'-m','pytest','tests','-q'],
            main_modified=False,branches_modified=False,
            note='Temporary main export plus exactly the correction patch; no Git branch, ref, index or main update performed.')
report_path.write_text(json.dumps(result,indent=2)+'\n')
with (HERE/'main_plus_correction_tests.log').open('w') as output:
    tests=subprocess.run(result['test_command'],cwd=snapshot,env=env,stdout=output,stderr=subprocess.STDOUT)
result.update(test_status='complete',test_exit_code=tests.returncode)
report_path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('snapshot','patch_applies_cleanly','test_exit_code')},indent=2))
print('\n'.join((HERE/'main_plus_correction_tests.log').read_text().splitlines()[-5:]))

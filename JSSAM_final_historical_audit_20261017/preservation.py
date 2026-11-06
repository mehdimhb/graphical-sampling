"""Parallel hashing of originals; this audit directory is the only write target."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
WORKERS=max(1,(len(os.sched_getaffinity(0))+1)//2)


def hashed(path):
    return str(path),hashlib.sha256(path.read_bytes()).hexdigest()


def before():
    file=OUT/'protected_before.json'
    if file.exists():
        raise RuntimeError('Refusing to replace the preflight preservation baseline')
    paths={ROOT/entry.decode() for entry in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0') if entry}
    previous=ROOT/'DI_kernel_comparison_JSSAM_20261017/protected_before.json'
    paths.update(Path(p) for p in json.loads(previous.read_text()))
    for folder in ['DI_kernel_comparison_JSSAM_20261017',
                   'DI_historical_implementation_audit_20261016',
                   'DI_historical_KDE_sensitivity_JSSAM_20261016',
                   'DI_translation_equal_unequal_JSSAM_20261016',
                   'simulations/results','simulations/populations']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.add(ROOT/'DI_reference_sensitivity_JSSAM/sources/Measuring_Spatial_Spread_current_2026-10-12.tex')
    paths={p for p in paths if p.is_file() and not p.is_relative_to(OUT)}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        result=dict(pool.map(hashed,sorted(paths)))
    file.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Protected {len(result)} files using {WORKERS} hashing workers')


def after():
    expected=json.loads((OUT/'protected_before.json').read_text())
    paths=[Path(p) for p in expected]
    assert all(p.is_file() for p in paths)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        actual=dict(pool.map(hashed,paths))
    changed=[p for p in expected if expected[p]!=actual[p]]
    result=dict(all_originals_unchanged=not changed,protected_files=len(expected),changed=changed,
                hashing_workers=WORKERS,available_cpus=len(os.sched_getaffinity(0)))
    (OUT/'preservation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert not changed


if __name__=='__main__':
    before() if len(sys.argv)>1 and sys.argv[1]=='before' else after()

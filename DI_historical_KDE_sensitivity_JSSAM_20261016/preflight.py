"""Inspect and hash saved inputs; no reference fitting or density calculation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCE = ROOT/'DI_reference_sensitivity_JSSAM'
CASES = {'C16':(144,16), 'M10':(155,10), 'S100':(959,100)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = OUT/'preflight.json'
    if destination.exists():
        raise FileExistsError('Preserving prior preflight evidence')
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    cases = {c['scenario']:c for c in protocol['scenarios'] if c['scenario'] in CASES}
    rows = pd.read_csv(SOURCE/'results.csv')
    primary = rows[rows.scenario.isin(CASES) & rows.experiment.isin(list('ABC'))]
    assert len(primary)==360 and primary.run_id.nunique()==360
    selected=[]; record=[]
    for sid,(N,n) in CASES.items():
        case=cases[sid]
        assert (case['N'],case['n'])==(N,n)
        with np.load(SOURCE/'frozen_inputs'/f'{sid}.npz',allow_pickle=False) as frozen:
            raw=frozen['raw_coords'];pi=frozen['pi'];sample=frozen['sample']
            x=(raw-raw.min(axis=0))/np.ptp(raw,axis=0).max()
            assert np.array_equal(sample,case['sample'])
            assert int(frozen['selection_seed'])==case['selection_seed']
            table=pd.read_csv(SOURCE/'populations'/f'{case["stem"]}.csv')
            assert np.array_equal(table[['x','y']].to_numpy(float),raw)
        group=primary[primary.scenario==sid]
        for exp in 'ABC':
            g=group[group.experiment==exp]
            assert len(g)==40 and set(g['index'])==set(range(1,41))
        ids=[f'{sid}_baseline_00']+sorted(group.run_id)
        a_first=None
        for rid in ids:
            folder=SOURCE/'runs'/rid
            summary=json.loads((folder/'summary.json').read_text())
            assert summary['run_id']==rid and summary['scenario']==sid and summary['status']=='completed'
            with np.load(folder/'state.npz',allow_pickle=False) as z:
                needed=['coords','sample','pi','labels','medoids','matched','translations','moved',
                        'P','OT_centers','preliminary_labels','path','f','g','q','e','H','invroot']
                assert set(needed)<=set(z.files)
                assert np.array_equal(z['coords'],x) and np.array_equal(z['pi'],pi)
                assert np.array_equal(z['sample'],sample)
                assert np.array_equal(z['initial_centers'],x[sample])
                assert np.array_equal(z['translations'],x[z['matched']]-x[z['medoids']])
                assert np.array_equal(z['moved'],x+z['translations'][z['labels']])
                assert np.isfinite(z['moved']).all()
                assert np.array_equal(z['labels'],json.loads(summary['hard_labels']))
                assert np.array_equal(z['path'],json.loads(summary['path']))
                assert np.array_equal(z['medoids']+1,json.loads(summary['medoid_ids']))
                assert np.array_equal(z['matched']+1,json.loads(summary['matched_sample_ids']))
                assert len(set(z['matched']))==n and set(z['matched'])==set(sample)
                labels=z['labels'];medoids=z['medoids'];ids1=np.arange(1,N+1)
                canonical=sorted((tuple(ids1[labels==k].tolist()),int(ids1[medoids[k]])) for k in range(n))
                assert hashlib.sha256(json.dumps(canonical).encode()).hexdigest()==summary['reference_hash']
                if summary['experiment']=='A':
                    signature=[z[k].copy() for k in ('P','OT_centers','preliminary_labels')]
                    if a_first is None:a_first=signature
                    else:assert all(np.array_equal(a,b) for a,b in zip(a_first,signature))
                if summary['experiment']=='C':
                    other=SOURCE/'runs'/f'{sid}_B_{summary["index"]:02d}'/'state.npz'
                    with np.load(other,allow_pickle=False) as b:
                        assert all(np.array_equal(z[k],b[k]) for k in ('P','OT_centers','preliminary_labels'))
            selected.append(dict(run_id=rid,scenario=sid,experiment=summary['experiment'],
                summary=str((folder/'summary.json').relative_to(ROOT)),
                state=str((folder/'state.npz').relative_to(ROOT)),
                summary_SHA256=digest(folder/'summary.json'),state_SHA256=digest(folder/'state.npz'),
                reference_hash=summary['reference_hash'],tsp_seed=summary['tsp_seed']))
        record.append(dict(scenario=sid,N=N,n=n,primary_runs=120,baseline_runs=1,
                           normalization='(raw_coords - per-column minimum) / maximum per-column span',
                           translated_coordinates_saved=True,translations_reconstructed_bitwise=True,
                           frozen_sample_and_probability_vectors_verified=True,
                           A_coupling_and_preliminary_labels_fixed=True,B_C_OT_reuse_verified=True))
    # Freeze the entire original study and protected research/package files before recalculation.
    old=json.loads((ROOT/'GFS_package_corrections_20261016/protected_before.json').read_text())
    protected={Path(name) for name in old if Path(name).is_file()}
    protected.update(p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.update((ROOT/'graphical_sampling').rglob('*.py'))
    manifest={str(p):digest(p) for p in sorted(protected)}
    (OUT/'source_hashes_before.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(sufficient=True,scenarios=list(CASES),excluded_scenario='C32',
                primary_evaluations=360,baseline_controls=3,
                reference_optimizations_required=0,checks=record,selected=selected,
                protected_source_files=len(manifest),
                note='No OT, clustering, path search, medoid selection or matching is needed; saved moved arrays are verified bitwise from the unchanged saved reference and matching.')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='selected'},indent=2))


if __name__=='__main__':main()

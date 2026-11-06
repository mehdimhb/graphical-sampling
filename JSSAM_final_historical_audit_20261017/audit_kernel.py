"""Read historical source and saved outputs only; never evaluates a KDE or DI."""
from pathlib import Path
import ast,csv,hashlib,json,subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
EV=OUT/'evidence/kernel'
EV.mkdir(parents=True,exist_ok=True)


def main():
    revisions=['d047278fc','a6102e6cc','e71d6935b','82a7eca7d','v1.0.0','HEAD']
    rows=[]
    for rev in revisions:
        commit=subprocess.check_output(['git','rev-parse',rev],cwd=ROOT,text=True).strip()
        path='graphical_sampling/index/density_disparity.py'
        raw=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)
        source=raw.decode()
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DensityDisparity')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_density')
        text='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
        assert 'kernel="tophat"' in text and 'bandwidth="scott"' in text
        assert 'score_samples(coords)' in text and 'self.kde_rtol' in text
        target=EV/f'{rev.replace("/","_")}_density_source.txt'
        target.write_text('\n'.join(f'{i+1:04d}: {line}' for i,line in enumerate(source.splitlines()))+'\n')
        rows.append(dict(revision=rev,commit=commit,path=path,source_sha256=hashlib.sha256(raw).hexdigest(),
                         method_start=node.lineno,method_end=node.end_lineno,kernel='tophat',
                         bandwidth='scott',rtol='self.kde_rtol (constructor default 1e-4)',
                         density_evaluation='self positions',evidence_file=str(target.relative_to(OUT))))
    # The executed medoid scorer is an override of the base pipeline, not its
    # KDE. Verify this directly from the historical notebook AST.
    commit=subprocess.check_output(['git','rev-parse','e71d6935b'],cwd=ROOT,text=True).strip()
    nbpath='simulations/jupyters/measure_calculation_nmedoids_clean.ipynb'
    raw=subprocess.check_output(['git','show',f'{commit}:{nbpath}'],cwd=ROOT)
    nb=json.loads(raw)
    cell=''.join(nb['cells'][7]['source'])
    cls=next(n for n in ast.parse(cell).body if isinstance(n,ast.ClassDef) and n.name=='DensityDisparityFlexible')
    methods=[n.name for n in cls.body if isinstance(n,ast.FunctionDef)]
    assert '_density' not in methods and '_score_single_density' not in methods
    build=''.join(nb['cells'][12]['source'])
    assert '"D": DensityDisparityFlexible(' in build
    imports=''.join(nb['cells'][3]['source'])
    assert 'from graphical_sampling.index import DensityDisparity as _BaseDensityDisparity' in imports
    (EV/'historical_execution_path.json').write_text(json.dumps(dict(commit=commit,
        notebook=nbpath,scorer_cell=7,scorer_execution_count=nb['cells'][7].get('execution_count'),
        override_methods=methods,KDE_override_present=False,DI_formula_override_present=False,
        imported_base='graphical_sampling.index.DensityDisparity',build_scorers_cell=12,
        retained_output_cells=[19,20],runtime_import_version_frozen=False,
        conclusion='Spherical top-hat is confirmed in the recovered selected source and best-supported for the traced medoid Monte Carlo outputs; historical runtime provenance remains qualified.'),indent=2)+'\n')
    with (EV/'kernel_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Verified {len(rows)} historical/current source versions; zero KDE/DI evaluations.')


if __name__=='__main__':main()

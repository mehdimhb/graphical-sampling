import subprocess,json,re,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REFS=['b612f9e7c','1f376e9fc','d57f9341e','247d9422e','902ed8ef4','d047278fc','a6102e6cc','e71d6935bfe824e72a7b722067a24775ed160610','b255918ea','82a7eca7d','7c61dbeafa6808e9eafa0b1fc678a25c077109cc','af36c9e3ff36f957243188ade01a1e01400f3709']
def git(*a): return subprocess.check_output(['git',*a],text=True)
manifest=[]; scan=[]
for ref in REFS:
    full=git('rev-parse',ref).strip()
    paths=git('ls-tree','-r','--name-only',ref).splitlines()
    selected=[p for p in paths if p=='graphical_sampling/index/density_disparity.py' or (ref not in ['b612f9e7c','1f376e9fc','d57f9341e','247d9422e','902ed8ef4'] and p.endswith('.ipynb') and (p.startswith('simulations/jupyters/') or p.startswith('simulations/Store/jupyter/')))]
    for path in selected:
        raw=git('show',f'{ref}:{path}')
        if path.endswith('.py'):
            cells=[{'source':raw.splitlines(True),'execution_count':None,'outputs':[]}]
        else:
            try: cells=json.loads(raw).get('cells',[])
            except json.JSONDecodeError:
                scan.append({'ref':full,'path':path,'error':'Git blob is not JSON; may be LFS pointer','raw':raw[:300]})
                continue
        hits=[]
        for i,c in enumerate(cells):
            src=''.join(c.get('source',[])); lines=src.splitlines()
            if re.search(r'linear_sum_assignment|sqeuclidean',src):
                windows=set()
                for n,line in enumerate(lines):
                    if re.search(r'linear_sum_assignment|sqeuclidean|cdist|def _cluster_medoids',line): windows.update(range(max(0,n-8),min(len(lines),n+12)))
                scan.append({'ref':full,'path':path,'cell':i if path.endswith('.ipynb') else '', 'execution_count':c.get('execution_count'),'output_count':len(c.get('outputs',[])),'matches':' | '.join(f'{n+1}:{lines[n]}' for n in sorted(windows))})
                hits.append(i)
        main=(path=='graphical_sampling/index/density_disparity.py' or bool(re.search(r'/(01_measure_calculation_spreadness_indices|03_variance_estimation_NHT|04_paper_plots_and_tables|measure_calculation_nmedoids_clean|paper_plots)\.ipynb$',path)))
        if main or hits:
            name=ref[:9]+'__'+Path(path).stem+'.txt'
            out=[]
            for i,c in enumerate(cells):
                out.append(f'===== Git {full}:{path} CELL {i} type={c.get("cell_type","module")} execution_count={c.get("execution_count")} outputs={len(c.get("outputs",[]))} =====')
                for n,l in enumerate(''.join(c.get('source',[])).splitlines(),1): out.append(f'C{i}:L{n:04d} {l}')
                for j,o in enumerate(c.get('outputs',[])):
                    out.append(f'OUTPUT {j} {o.get("output_type")}')
                    t=o.get('text',o.get('data',{}).get('text/plain',[]))
                    if isinstance(t,list): t=''.join(t)
                    out.extend(str(t).splitlines())
            (ROOT/name).write_text('\n'.join(out)+'\n')
            manifest.append({'ref':full,'path':path,'blob_sha256':hashlib.sha256(raw.encode()).hexdigest(),'extraction':name,'matching_cells':hits})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'all_notebook_matching_scan.json').write_text(json.dumps(scan,indent=2)+'\n')
print(f'Extracted {len(manifest)} primary sources; matching-source hits {len(scan)}')

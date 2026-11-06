"""Change only the TSP solver in the saved, manuscript-adapted package pipeline."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-exact-tsp')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,json
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'sources'))
import numpy as np
import pandas as pd
import adapter as base
from exact_tsp import ExactOpenTSPSolver,path_cost,decimal_distances

OriginalPartition=base.ManuscriptPartition
PATH_INDEX=0;LAST=None
class ExactPartition(OriginalPartition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tsp_solver=ExactOpenTSPSolver(PATH_INDEX)
        global LAST
        LAST=self

def run_one(sample,seed,path_index=0):
    global PATH_INDEX
    PATH_INDEX=path_index
    base.ManuscriptPartition=ExactPartition
    base.OUT=OUT/'exact_runs' if path_index==0 else OUT/'optimal_path_checks'/f'path_{path_index:03d}'
    result=base.calculate_run(sample,'official',seed)
    solver=LAST.tsp_solver;folder=base.OUT/'runs'/result['run_id']
    info=solver.info
    (folder/'exact_solver.json').write_text(json.dumps(dict(info,selected_oriented_optimal_index=path_index,selected_spatial_rank_path=solver.selected_spatial_rank_path),indent=2))
    result.update(variant='deterministic_exact' if path_index==0 else 'alternative_oriented_optimum',oriented_optimal_index=path_index,state_path=str((folder/'state.npz').relative_to(OUT)),optimal_unoriented_classes=len(info['optimal_unoriented_paths']),optimal_oriented_path_count=info['optimal_oriented_paths_count'],exact_length_decimal=info['best_length_decimal'],exact_gap_decimal='0',chosen_path_spatial_ranks=json.dumps(solver.selected_spatial_rank_path),chosen_path_original_labels=json.dumps(LAST.path_order.tolist()),next_unoriented_path_gap_decimal=info['runner_up_unoriented_length_gap_decimal'])
    if seed==0 and path_index==0:
        dest=OUT/'enumeration';dest.mkdir(exist_ok=True)
        pd.DataFrame([dict(spatial_rank_path=json.dumps(p),length_decimal=str(cost),optimal=p in [tuple(t) for t in info['optimal_unoriented_paths']]) for p,cost in solver.records]).to_csv(dest/f'S{sample+1}_all_unoriented_paths.csv',index=False)
        (dest/f'S{sample+1}_optimal_paths.json').write_text(json.dumps(info,indent=2))
    return result

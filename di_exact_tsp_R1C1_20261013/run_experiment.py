from runner import *
from joblib import Parallel,delayed
if __name__=='__main__':
    assert not (OUT/'exact_results.csv').exists(),'Preserve this run; reproduce in a fresh directory.'
    rows=Parallel(n_jobs=8,verbose=5)(delayed(run_one)(sample,seed) for sample in range(2) for seed in range(10))
    pd.DataFrame(rows).to_csv(OUT/'exact_results.csv',index=False,float_format='%.17g')
    extra=[]
    for sample in range(2):
        info=json.loads((OUT/'enumeration'/f'S{sample+1}_optimal_paths.json').read_text())
        for path_index in range(1,info['optimal_oriented_paths_count']):extra.append(run_one(sample,0,path_index))
    pd.DataFrame(extra).to_csv(OUT/'optimal_path_checks.csv',index=False,float_format='%.17g')
    print('Exact repeated runs',len(rows),'additional oriented optimal-path checks',len(extra),flush=True)

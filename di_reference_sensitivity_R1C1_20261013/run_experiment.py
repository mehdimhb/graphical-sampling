from adapter import *
from joblib import Parallel,delayed
def safe_run(task):
    try:return dict(ok=True,result=calculate_run(*task))
    except Exception as e:return dict(ok=False,task=task,error=repr(e))
if __name__=='__main__':
    assert not (OUT/'results.csv').exists(),'Use a fresh experiment directory for a new run.'
    tasks=[(s,kind,k) for s in range(2) for kind in ['official','stress'] for k in range(10)]
    results=Parallel(n_jobs=8,verbose=5)(delayed(safe_run)(t) for t in tasks)
    writecsv([r['result'] for r in results if r['ok']],OUT/'results.csv')
    failures=[r for r in results if not r['ok']];(OUT/'failures.json').write_text(json.dumps(failures,indent=2))
    raw,pop=load_population();writecsv(dict(unit_id=pop.ids,x_original=raw.x,y_original=raw.y,x_normalized=pop.coords[:,0],y_normalized=pop.coords[:,1],stored_weight=raw.prob,pi=pop.inclusions),OUT/'population.csv')
    print('Completed',len(results)-len(failures),'of',len(tasks),'; failures',failures,flush=True)

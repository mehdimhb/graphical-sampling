"""Three valid target placements of the unchanged compact sample."""
from helpers import *

def main():
    original=json.loads((OUT/'runs/original/summary.json').read_text())
    tight=json.loads((OUT/'runs/original_tight/summary.json').read_text())
    locations=[dict(id='top_right',label='Top-right corner',delta=[9,8]),
               dict(id='middle',label='Middle',delta=[5,4]),
               dict(id='top_left',label='Top-left corner',delta=[0,8])]
    config=dict(sample=original['sample'],placements=locations,solver_seed=17,
     outer_tol=1e-10,marginal_tol=1e-12,tight_outer_tol=1e-12,tight_marginal_tol=1e-13,
     epsilon=.01,TSP_restarts=15,threads=1,N=144,n=8,pi='1/18',
     method='sample-initialized manuscript OT; squared medoids and matching; historical spherical Scott KDE',
     original_source='DI_translation_concentrated_JSSAM_20261016/runs/sample1_original',
     original_source_SHA256=hashlib.sha256((OUT/'runs/original/summary.json').read_bytes()).hexdigest(),
     method_SHA256=hashlib.sha256((OUT/'method.py').read_bytes()).hexdigest())
    p=OUT/'config.json'
    if p.exists():assert json.loads(p.read_text())==config
    else:atomic_json(p,config)
    if (OUT/'selection.json').exists():checkpoint();print('All target placements already complete.');return
    results=[]
    for loc in locations:
        sample=np.array(original['sample'])+loc['delta']
        assert ((sample>=0)&(sample<=11)).all()
        value=run_one(loc['id'],sample)
        control=run_one(loc['id']+'_tight',sample,outer_tol=1e-12,marginal_tol=1e-13)
        row=dict(**loc,comparison=compare(original,value),tight_stability=compare(value,control),
           tight_comparison=compare(tight,control),DI=value['DI'],Dnet=value['Dnet'],eta=value['eta'])
        results.append(row);atomic_json(OUT/'placements.json',results)
    atomic_json(OUT/'selection.json',dict(complete=True,placements=[r['id'] for r in results]))
    checkpoint()
    rows=[dict(position='Original (bottom-left)',shift='(0, 0)',DI=original['DI'],DI_difference=0,changed_memberships=0)]
    for loc in results:
        rows.append(dict(position=loc['label'],shift=str(tuple(loc['delta'])),DI=loc['DI'],
         DI_difference=loc['comparison']['DI_difference'],changed_memberships=loc['comparison']['changed_units']))
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()

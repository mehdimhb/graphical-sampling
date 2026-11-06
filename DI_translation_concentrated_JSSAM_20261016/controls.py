"""Additional bounded checks for the displayed pair; never overwrites a run."""
from run_experiment import *

def main():
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    a=json.loads((OUT/'runs'/selected['original_id']/'summary.json').read_text())
    b=json.loads((OUT/'runs'/selected['translated_id']/'summary.json').read_text())
    ta=run_one('display_original_tight',a['sample'],outer_tol=1e-12,marginal_tol=1e-13)
    tb=run_one('display_translated_tight',b['sample'],outer_tol=1e-12,marginal_tol=1e-13)
    replay=run_one('display_original_replay',a['sample'])
    # Full solver rerun under a joint raw-coordinate shift, using identical index identities.
    path=OUT/'runs/display_joint'
    if not (path/'summary.json').exists():
        z=evaluate(12,a['sample'],seed=17,offset=(8,-4))
        z['tsp_diagnostic']=tsp_diagnostic(z);path.mkdir(exist_ok=True)
        np.savez_compressed(path/'state.npz',**{k:np.asarray(z[k]) for k in ['sample','sample_ids','centers','coupling','preliminary_labels','path','omega','labels','medoid_ids','permutation','matched_ids','shifts','moved','f_counts','g_counts','f','g']})
        atomic_json(path/'summary.json',z)
    joint=json.loads((path/'summary.json').read_text())
    # Repeatability: compare scientific fields; timing/counters may differ by design.
    scientific=['sample_ids','centers','coupling','preliminary_labels','path','omega','labels','medoid_ids','permutation','matched_ids','shifts','moved','f_counts','g_counts','f','g','DI','Dnet','eta']
    assert all(a[k]==replay[k] for k in scientific)
    assert all(a[k]==joint[k] for k in scientific)
    result=dict(tighter_original=compare(a,ta),tighter_translated=compare(b,tb),
      tighter_pair=compare(ta,tb),same_seed_replay_identical=True,joint_coordinate_shift_identical=True)
    # Frozen reference control for the displayed pair.
    raw=np.array([(x,y) for y in range(12) for x in range(12)],int)
    labels=np.array(a['labels']);d=np.array(selected['delta']);moved=np.array(a['moved'])+d
    f,_,_,_=density(raw,11);g,count,_,_=density(moved,11)
    assert count.tolist()==a['g_counts']
    fixed=quantities(f,g);result['fixed_reference_control']=dict(**fixed,DI_difference=fixed['DI']-a['DI'])
    atomic_json(OUT/'controls.json',result);checkpoint();print(json.dumps(result,indent=2))

if __name__=='__main__':main()

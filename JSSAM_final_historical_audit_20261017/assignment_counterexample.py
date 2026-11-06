"""Two-pair deterministic assignment check; no sampling, clustering or DI study."""
from pathlib import Path
from decimal import Decimal,localcontext
from itertools import permutations
import json,csv
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

OUT=Path(__file__).resolve().parent


def main():
    medoids=np.array([[0.,0.],[2.,0.]])
    sample=np.array([[0.,0.],[1.,1.]])
    delta=np.array([3.,0.])
    shifted=sample+delta
    perms=list(permutations(range(2)))
    rows=[]
    result=dict(medoids=medoids.tolist(),original_sample=sample.tolist(),
                shift=delta.tolist(),shifted_sample=shifted.tolist(),
                interpretation='rows are fixed medoids; permutation entries select sample columns')
    for metric in ['euclidean','sqeuclidean']:
        before=cdist(medoids,sample,metric=metric)
        after=cdist(medoids,shifted,metric=metric)
        initial=[float(sum(before[i,p[i]] for i in range(2))) for p in perms]
        final=[float(sum(after[i,p[i]] for i in range(2))) for p in perms]
        rb,cb=linear_sum_assignment(before);ra,ca=linear_sum_assignment(after)
        assert np.array_equal(rb,[0,1]) and np.array_equal(ra,[0,1])
        assert list(cb)==list(perms[int(np.argmin(initial))])
        assert list(ca)==list(perms[int(np.argmin(final))])
        result[metric]=dict(cost_before=before.tolist(),cost_after=after.tolist(),
                           assignment_before=cb.tolist(),assignment_after=ca.tolist(),
                           identity_cost_before=initial[0],swap_cost_before=initial[1],
                           identity_cost_after=final[0],swap_cost_after=final[1])
        rows.append(dict(metric=metric,identity_before=initial[0],swap_before=initial[1],
                         identity_after=final[0],swap_after=final[1],
                         winner_before='identity' if initial[0]<initial[1] else 'swap',
                         winner_after='identity' if final[0]<final[1] else 'swap'))
    assert result['euclidean']['assignment_before']==[0,1]
    assert result['euclidean']['assignment_after']==[1,0]
    assert result['sqeuclidean']['assignment_before']==result['sqeuclidean']['assignment_after']==[0,1]
    common_term=2*delta@(sample.sum(axis=0)-medoids.sum(axis=0))+len(sample)*(delta@delta)
    assert common_term==12
    assert rows[1]['identity_after']-rows[1]['identity_before']==common_term
    assert rows[1]['swap_after']-rows[1]['swap_before']==common_term
    before_translations=sample-medoids
    after_translations=shifted[[1,0]]-medoids
    changes=after_translations-before_translations
    assert not np.array_equal(changes[0],changes[1])
    result['euclidean_cluster_translation_changes']=changes.tolist()
    result['squared_common_additive_term']=int(common_term)
    # The same example fits the unit square after dividing all coordinates and
    # the shift by five; assignment changes are not an out-of-domain artifact.
    result['unit_square_equivalent']=dict(medoids=(medoids/5).tolist(),
        sample=(sample/5).tolist(),shift=(delta/5).tolist(),shifted=(shifted/5).tolist())
    with localcontext() as ctx:
        ctx.prec=70
        identity=Decimal(3)+Decimal(5).sqrt()
        swap=Decimal(1)+Decimal(17).sqrt()
        assert swap<identity
        result['high_precision_after_gap_identity_minus_swap']=str(identity-swap)
    result['all_checks_passed']=True
    (OUT/'assignment_counterexample.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'assignment_costs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

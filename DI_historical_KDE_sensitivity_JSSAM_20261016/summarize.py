"""Descriptive paired comparison and unchanged optimization-quality screens."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'DI_reference_sensitivity_JSSAM'
SID=('C16','M10','S100')
BANDS=(.001,.01,.02)


def main():
    data=pd.read_csv(OUT/'results.csv',float_precision='round_trip')
    bases=pd.read_csv(OUT/'baselines.csv',float_precision='round_trip').set_index('scenario')
    assert len(data)==360 and data.run_id.nunique()==360 and set(data.scenario)==set(SID)
    summaries=[];qualities=[];pairs=[]
    for sid in SID:
        base=bases.loc[sid];group=data[data.scenario==sid]
        ids=[f'{sid}_baseline_00']+list(group.run_id)
        originals={rid:json.loads((SOURCE/'runs'/rid/'summary.json').read_text()) for rid in ids}
        base_raw=originals[base.run_id]
        native_best=min(r['native_OT_objective'] for r in originals.values())
        common={rid:r['squared_transport_loss']/base_raw['cost_median']+.01*r['entropy_term'] for rid,r in originals.items()}
        common_best=min(common.values())
        tsp_best={}
        for rid,r in originals.items():
            oid=r['OT_id'];best=min(r['TSP_length'],r['TSP_benchmark_length'])
            tsp_best[oid]=min(tsp_best.get(oid,float('inf')),best)
        def quality(rid):
            r=originals[rid]
            return dict(native_OT_excess=max(0,(r['native_OT_objective']-native_best)/max(abs(native_best),1e-14)),
                common_OT_excess=max(0,(common[rid]-common_best)/max(abs(common_best),1e-14)),
                TSP_excess=max(0,(r['TSP_length']-tsp_best[r['OT_id']])/tsp_best[r['OT_id']]))
        base=base.to_dict();base.update(quality(base['run_id']))
        for exp in 'ABC':
            g=group[group.experiment==exp].copy();assert len(g)==40
            g['sign_diff_from_baseline']=np.sign(g.DI)!=np.sign(base['DI'])
            g['strict_reversal_from_baseline']=g.DI*base['DI']<0
            g['old_sign_diff_from_baseline']=np.sign(g.old_box_DI)!=np.sign(base['old_box_DI'])
            delta=g.DI-base['DI'];old_delta=g.old_box_DI-base['old_box_DI']
            summary=dict(scenario=sid,population=base['population'],N=int(base['N']),n=int(base['n']),
                experiment=exp,runs=40,baseline_DI=base['DI'],old_box_baseline_DI=base['old_box_DI'],
                DI_mean=g.DI.mean(),DI_sd=g.DI.std(ddof=1),DI_min=g.DI.min(),DI_max=g.DI.max(),
                mean_delta_from_baseline=delta.mean(),MAD_from_baseline=abs(delta).mean(),
                max_abs_delta_from_baseline=abs(delta).max(),
                old_box_DI_sd=g.old_box_DI.std(ddof=1),old_box_DI_min=g.old_box_DI.min(),old_box_DI_max=g.old_box_DI.max(),
                old_box_MAD_from_baseline=abs(old_delta).mean(),old_box_max_abs_delta_from_baseline=abs(old_delta).max(),
                sign_differences_from_baseline=int(g.sign_diff_from_baseline.sum()),
                strict_sign_reversals_from_baseline=int(g.strict_reversal_from_baseline.sum()),
                old_box_sign_differences_from_baseline=int(g.old_sign_diff_from_baseline.sum()),
                sign_changes_between_KDEs=int(g.sign_changed_from_box.sum()),
                strict_reversals_between_KDEs=int(g.strict_sign_reversal_from_box.sum()),
                zero_transitions_between_KDEs=int(g.zero_transition_from_box.sum()),
                mean_DI_change_between_KDEs=g.DI_historical_minus_box.mean(),
                max_abs_DI_change_between_KDEs=abs(g.DI_historical_minus_box).max(),
                reference_count=g.reference_hash.nunique(),translated_population_count=g.moved_array_SHA256.nunique(),
                Dnet_min=g.Dnet.min(),Dnet_max=g.Dnet.max(),eta_min=g.eta.min(),eta_max=g.eta.max())
            summaries.append(summary)
            records=[base]
            for r in g.to_dict('records'):
                independent=quality(r['run_id'])
                assert all(abs(r[k]-v)<1e-12 for k,v in independent.items())
                r.update(independent);records.append(r)
            for tol in BANDS:
                eligible=[r for r in records if max(r[k] for k in ('native_OT_excess','common_OT_excess','TSP_excess'))<=tol]
                distinct_pairs=[(a,b) for a,b in itertools.combinations(eligible,2)
                                if a['moved_array_SHA256']!=b['moved_array_SHA256']]
                best=max(distinct_pairs,key=lambda pair:abs(pair[0]['DI']-pair[1]['DI'])) if distinct_pairs else None
                qualities.append(dict(scenario=sid,experiment=exp,tolerance=tol,
                    eligible_evaluations_including_baseline=len(eligible),
                    eligible_primary_evaluations=sum(r['experiment']!='baseline' for r in eligible),
                    distinct_translation_pairs=len(distinct_pairs),
                    max_abs_DI_difference=max((abs(a['DI']-b['DI']) for a,b in distinct_pairs),default=np.nan),
                    old_box_difference_for_same_max_pair=abs(best[0]['old_box_DI']-best[1]['old_box_DI']) if best else np.nan,
                    strict_pair_sign_reversals=sum(a['DI']*b['DI']<0 for a,b in distinct_pairs),
                    run_a=best[0]['run_id'] if best else '',run_b=best[1]['run_id'] if best else '',
                    DI_a=best[0]['DI'] if best else np.nan,DI_b=best[1]['DI'] if best else np.nan))
            for a,b in itertools.combinations(records,2):
                pair=dict(scenario=sid,experiment=exp,run_a=a['run_id'],run_b=b['run_id'],
                    distinct_translation=a['moved_array_SHA256']!=b['moved_array_SHA256'],
                    distinct_partition_medoid_reference=a['reference_hash']!=b['reference_hash'],
                    DI_a=a['DI'],DI_b=b['DI'],absolute_DI_difference=abs(a['DI']-b['DI']),
                    old_box_absolute_DI_difference=abs(a['old_box_DI']-b['old_box_DI']),
                    sign_difference=np.sign(a['DI'])!=np.sign(b['DI']),strict_sign_reversal=a['DI']*b['DI']<0)
                pair.update({key+'_a':a[key] for key in ('native_OT_excess','common_OT_excess','TSP_excess')})
                pair.update({key+'_b':b[key] for key in ('native_OT_excess','common_OT_excess','TSP_excess')})
                for tol in BANDS:
                    pair['both_nearbest_'+str(tol)]=max(a[k] for k in ('native_OT_excess','common_OT_excess','TSP_excess'))<=tol and max(b[k] for k in ('native_OT_excess','common_OT_excess','TSP_excess'))<=tol
                pairs.append(pair)
    summary=pd.DataFrame(summaries)
    assert len(pairs)==7380
    summary.to_csv(OUT/'summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(qualities).to_csv(OUT/'near_best_summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(pairs).to_csv(OUT/'pairwise.csv',index=False,float_format='%.17g')
    print(bases[['old_box_DI','DI','Dnet','eta','h']].to_string())
    print(summary[['scenario','experiment','DI_sd','old_box_DI_sd','DI_min','DI_max',
        'max_abs_delta_from_baseline','sign_differences_from_baseline','sign_changes_between_KDEs']].to_string(index=False))
    print(pd.DataFrame(qualities)[['scenario','experiment','tolerance','eligible_evaluations_including_baseline',
        'max_abs_DI_difference','run_a','run_b']].to_string(index=False))


if __name__=='__main__':main()

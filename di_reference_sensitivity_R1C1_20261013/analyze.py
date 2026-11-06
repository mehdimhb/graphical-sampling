"""Loss-based pair eligibility is independent of the DI differences being compared."""
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
P=Path(__file__).resolve().parent
read=lambda s:pd.read_csv(P/s,float_precision='round_trip')
def state(run):return np.load(P/'runs'/run/'state.npz')
def write(data,name):pd.DataFrame(data).to_csv(P/name,index=False,float_format='%.17g')
def gap(a,b):return abs(a-b)/max(abs(a),abs(b),1e-15)
if __name__=='__main__':
    data=read('results.csv');pairs=[];distinct=[];equivalence=[];groupstats=[]
    for sample,group in data.groupby('sample',sort=False):
        baseline=group[group.run_id.eq(sample.lower()+'_official_00')].iloc[0]
        median=baseline.OT_cost_median
        for i in group.index:data.loc[i,'J_OT_common_scale']=data.loc[i,'transport_raw_squared_cost']/median+.01*data.loc[i,'transport_entropy']
    write(data,'scored_references.csv')
    for sample,group in data.groupby('sample',sort=False):
        unique=group.drop_duplicates('reference_hash',keep='first');distinct.append(unique)
        for key,g in group.groupby('reference_hash',sort=False):
            assert g.DI.max()-g.DI.min()<1e-12
            equivalence.append(dict(sample=sample,reference_hash=key,representative=g.iloc[0].run_id,runs=json.dumps(g.run_id.tolist()),count=len(g),DI_range=float(g.DI.max()-g.DI.min()),native_OT_loss_range=float(g.J_OT_native.max()-g.J_OT_native.min()),Qsoft_range=float(g.Q_soft.max()-g.Q_soft.min())))
        official_hashes=set(group[group.kind=='official'].reference_hash);stress_hashes=set(group[group.kind=='stress'].reference_hash)
        for kind in ['official','stress']:
            g=group[group.kind==kind]
            groupstats.append(dict(sample=sample,kind=kind,runs=len(g),distinct_references=g.reference_hash.nunique(),DI_min=float(g.DI.min()),DI_max=float(g.DI.max()),DI_range=float(g.DI.max()-g.DI.min()),TSP_regret_min=float(g.TSP_relative_regret.min()),TSP_regret_max=float(g.TSP_relative_regret.max())))
        for (_,a),(_,b) in itertools.combinations(unique.iterrows(),2):
            sa,sb=state(a.run_id),state(b.run_id);x=sa['coords'];pi=sa['pi']
            costs=cdist(x[sa['medoids']],x[sb['medoids']],metric='sqeuclidean');ra,rb=linear_sum_assignment(costs)
            overlap=np.zeros((8,8));np.add.at(overlap,(sa['labels'],sb['labels']),pi)
            la,lb=linear_sum_assignment(-overlap);disagree=1-overlap[la,lb].sum()/pi.sum()
            # Center out any common shift before measuring translated deformation.
            wa=sa['moved']@sa['invroot'];wb=sb['moved']@sb['invroot'];d=wa-wb;center=(d.min(axis=0)+d.max(axis=0))/2
            deformation=float(abs(d-center).max());dist=cdist(wa,wa,metric='chebyshev');mask=~np.eye(len(x),dtype=bool)
            margin=float(abs(dist[mask]-1).min());old=dist<=1;new=cdist(wb,wb,metric='chebyshev')<=1
            signdiff=float(b.eta*np.sign(b.Dnet)-a.eta*np.sign(a.Dnet));residual=float((1-b.eta)*b.Dnet-(1-a.eta)*a.Dnet)
            pair=dict(sample=sample,run_a=a.run_id,run_b=b.run_id,official_pair=a.reference_hash in official_hashes and b.reference_hash in official_hashes,stress_pair=a.reference_hash in stress_hashes and b.reference_hash in stress_hashes,native_OT_absolute_gap=abs(a.J_OT_native-b.J_OT_native),native_OT_relative_gap=gap(a.J_OT_native,b.J_OT_native),common_OT_absolute_gap=abs(a.J_OT_common_scale-b.J_OT_common_scale),common_OT_relative_gap=gap(a.J_OT_common_scale,b.J_OT_common_scale),Qsoft_absolute_gap=abs(a.Q_soft-b.Q_soft),Qsoft_relative_gap=abs(a.Q_soft-b.Q_soft)/min(a.Q_soft,b.Q_soft),raw_transport_relative_gap=gap(a.transport_raw_squared_cost,b.transport_raw_squared_cost),Qhard_relative_gap=abs(a.Q_hard-b.Q_hard)/min(a.Q_hard,b.Q_hard),medoid_set_RMS=float(np.sqrt(costs[ra,rb].mean())),medoid_set_max_distance=float(np.sqrt(costs[ra,rb].max())),partition_IP_disagreement=float(disagree),Dnet_a=a.Dnet,Dnet_b=b.Dnet,eta_a=a.eta,eta_b=b.eta,DI_a=a.DI,DI_b=b.DI,DI_signed_difference=float(b.DI-a.DI),DI_absolute_difference=abs(a.DI-b.DI),opposite_Dnet_sign=bool(a.Dnet*b.Dnet<0),near_zero_sign_reversal=bool(a.Dnet*b.Dnet<0 and max(abs(a.Dnet),abs(b.Dnet))<.01),DI_difference_sign_term=signdiff,DI_difference_weighted_Dnet_term=residual,translated_centered_deformation_whitened=deformation,kernel_margin_a=margin,box_margin_sufficient_for_equal_score=2*deformation<margin,changed_kernel_pairs_directed=int((old!=new).sum()),changed_translated_density_units=int((sa['g']!=sb['g']).sum()))
            assert abs(pair['DI_signed_difference']-signdiff-residual)<5e-15
            if pair['box_margin_sufficient_for_equal_score']:assert np.array_equal(sa['g'],sb['g'])
            pairs.append(pair)
    distinct=pd.concat(distinct,ignore_index=True);pairs=pd.DataFrame(pairs)
    write(distinct,'distinct_references.csv');write(pairs,'reference_pairs.csv');write(equivalence,'equivalent_runs.csv');write(groupstats,'run_group_summary.csv')
    tolerances=[]
    for sample,g in pairs.groupby('sample'):
        for scope in ['official','stress','all']:
            scoped=g[g.official_pair] if scope=='official' else g[g.stress_pair] if scope=='stress' else g
            for tol in [1e-10,.001,.01,.05]:
                for criterion in ['native_OT','native_OT_and_Qsoft','native_OT_Qsoft_and_common_OT']:
                    eligible=scoped[scoped.native_OT_relative_gap<=tol]
                    if criterion!='native_OT':eligible=eligible[eligible.Qsoft_relative_gap<=tol]
                    if criterion.endswith('common_OT'):eligible=eligible[eligible.common_OT_relative_gap<=tol]
                    maximum=float(eligible.DI_absolute_difference.max()) if len(eligible) else None
                    worst=eligible.loc[eligible.DI_absolute_difference.idxmax()] if len(eligible) else None
                    tolerances.append(dict(sample=sample,scope=scope,criterion=criterion,relative_tolerance=tol,eligible_distinct_pairs=len(eligible),max_DI_difference=maximum,material_pairs_over_point05=int((eligible.DI_absolute_difference>.05).sum()),material_pairs_over_point1=int((eligible.DI_absolute_difference>.1).sum()),worst_run_a=worst.run_a if worst is not None else '',worst_run_b=worst.run_b if worst is not None else ''))
    write(tolerances,'objective_tolerances.csv')
    # Compact table includes the two actual official references plus the same,
    # pre-generated lower-loss stress reference for each fixed sample.
    chosen=[]
    for sample in ['S1','S2']:
        g=distinct[distinct['sample']==sample];official=g[g.kind=='official']
        chosen.extend(official.run_id.tolist())
        # Lower native OT loss and lower Qsoft than both official references.
        lower=g[(g.J_OT_native<official.J_OT_native.min())&(g.Q_soft<official.Q_soft.min())]
        if len(lower):chosen.append(lower.sort_values('J_OT_native').iloc[0].run_id)
    compact=data[data.run_id.isin(chosen)].copy()
    columns=['sample','run_id','kind','J_OT_native','J_OT_common_scale','Q_soft','Q_hard','hard_IP_min','hard_IP_max','Dnet','eta','DI','medoid_ids','hard_IP_totals','TSP_relative_regret']
    write(compact[columns],'comparison_table.csv')
    summary=dict(reference_count=int(len(distinct)),samples=2,runs=40,failures=json.loads((P/'failures.json').read_text()),opposite_sign_distinct_pairs=int(pairs.opposite_Dnet_sign.sum()),near_zero_sign_reversal_distinct_pairs=int(pairs.near_zero_sign_reversal.sum()),selected_runs=chosen,conclusion_scope='This fixed population, two prespecified samples, fixed UP vector and box bandwidth; 10 official and10 stress runs per sample.')
    (P/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    print(compact[columns].to_string(index=False));print(pd.DataFrame(tolerances).query('scope=="all"').to_string(index=False));print(json.dumps(summary,indent=2))

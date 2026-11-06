from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent
design=json.loads((P/'frozen_design.json').read_text())
OLD=Path(next(name for name in design['source_hashes'] if Path(name).name=='results.csv')).parent
read=lambda name:pd.read_csv(P/name,float_precision='round_trip')
write=lambda data,name:pd.DataFrame(data).to_csv(P/name,index=False,float_format='%.17g')
def partition_disagreement(a,b,pi):
    overlap=np.zeros((8,8));np.add.at(overlap,(a,b),pi)
    r,c=linear_sum_assignment(-overlap);mapping=np.empty(8,int);mapping[c]=r
    return float(1-overlap[r,c].sum()/pi.sum()),int((a!=mapping[b]).sum())

if __name__=='__main__':
    exact=read('exact_results.csv');checks=read('optimal_path_checks.csv')
    old=read('sources/results.csv');old=old[old.kind=='official'].copy()
    write(old,'approximate_results.csv');rows=[];repeats=[];reversal=[];comparisons=[]
    for sample in ['S1','S2']:
        e=exact[exact['sample']==sample].sort_values('run');base=e.iloc[0];st=np.load(P/base.state_path)
        for row in e.itertuples():
            s=np.load(P/row.state_path);same={key:st[key].dtype==s[key].dtype and st[key].shape==s[key].shape and st[key].tobytes()==s[key].tobytes() for key in st.files}
            assert all(same.values());assert row.DI==base.DI
            repeats.append(dict(sample=sample,seed=row.run,DI=row.DI,reference_hash=row.reference_hash,all_intermediate_arrays_bit_identical=all(same.values())))
        for row in checks[checks['sample']==sample].itertuples():
            s=np.load(P/row.state_path);disagree,count=partition_disagreement(st['labels'],s['labels'],st['pi'])
            reversal.append(dict(sample=sample,oriented_path_index=row.oriented_optimal_index,path_is_reversal=bool(np.array_equal(st['path'][::-1],s['path'])),same_preliminary_transport=np.array_equal(st['P'],s['P']),same_hard_partition=np.array_equal(st['labels'],s['labels']),partition_IP_disagreement=disagree,changed_hard_units=count,same_reference_hash=row.reference_hash==base.reference_hash,same_medoids=np.array_equal(st['medoids'],s['medoids']),same_matching=np.array_equal(st['matched'],s['matched']),same_translated_coordinates=np.array_equal(st['moved'],s['moved']),soft_membership_max_abs_difference=float(abs(st['mu']-s['mu']).max()),same_densities=np.array_equal(st['g'],s['g']),Dnet_difference=row.Dnet-base.Dnet,eta_difference=row.eta-base.eta,DI_difference=row.DI-base.DI))
        o=old[old['sample']==sample]
        for row in o.itertuples():
            source=OLD/'runs'/row.run_id;dest=P/'approximate_baseline'/row.run_id
            if not dest.exists():shutil.copytree(source,dest)
            s=np.load(dest/'state.npz');disagree,count=partition_disagreement(st['labels'],s['labels'],st['pi'])
            comparisons.append(dict(sample=sample,approximate_run=row.run_id,seed=row.run,approximate_path=json.dumps(s['path'].tolist()),exact_path=json.dumps(st['path'].tolist()),same_OT_transport=np.array_equal(st['P'],s['P']),same_OT_centroids=np.array_equal(st['ot_centroids'],s['ot_centroids']),same_preliminary_labels=np.array_equal(st['preliminary_labels'],s['preliminary_labels']),approx_path_is_exact_or_reverse=bool(np.array_equal(s['path'],st['path']) or np.array_equal(s['path'],st['path'][::-1])),approximate_TSP_length=row.TSP_length,exact_TSP_length=base.TSP_length,TSP_absolute_gap=row.TSP_length-base.TSP_length,TSP_relative_gap=(row.TSP_length-base.TSP_length)/base.TSP_length,same_final_reference=row.reference_hash==base.reference_hash,partition_IP_disagreement=disagree,changed_hard_units=count,same_medoid_set=sorted(st['medoids'])==sorted(s['medoids']),soft_membership_max_abs_difference=float(abs(st['mu']-s['mu']).max()),approximate_Dnet=row.Dnet,exact_Dnet=base.Dnet,approximate_eta=row.eta,exact_eta=base.eta,approximate_DI=row.DI,exact_DI=base.DI,DI_difference=row.DI-base.DI,approximate_Qsoft=row.Q_soft,exact_Qsoft=base.Q_soft,approximate_Qhard=row.Q_hard,exact_Qhard=base.Q_hard))
        for _,group in o.groupby('reference_hash',sort=False):
            r=group.iloc[0];gap=(r.TSP_length-base.TSP_length)/base.TSP_length
            rows.append(dict(sample=sample,method='Approximate: optimal path' if abs(gap)<1e-12 else 'Approximate: longer path',runs=len(group),TSP_length=r.TSP_length,TSP_gap_percent=100*gap,Dnet=r.Dnet,eta=r.eta,DI=r.DI,Q_soft=r.Q_soft,Q_hard=r.Q_hard,J_OT=r.J_OT_native,hard_IP_min=r.hard_IP_min,hard_IP_max=r.hard_IP_max,medoid_ids=r.medoid_ids,reference_hash=r.reference_hash))
        rows.append(dict(sample=sample,method='Exact: deterministic path',runs=len(e),TSP_length=base.TSP_length,TSP_gap_percent=0.,Dnet=base.Dnet,eta=base.eta,DI=base.DI,Q_soft=base.Q_soft,Q_hard=base.Q_hard,J_OT=base.J_OT_native,hard_IP_min=base.hard_IP_min,hard_IP_max=base.hard_IP_max,medoid_ids=base.medoid_ids,reference_hash=base.reference_hash))
    write(rows,'02_Comparison.csv');write(repeats,'repeatability.csv');write(reversal,'optimal_orientation_comparisons.csv');write(comparisons,'approximate_vs_exact.csv')
    summary={}
    for sample in ['S1','S2']:
        o=old[old['sample']==sample];e=exact[exact['sample']==sample]
        info=json.loads((P/'enumeration'/f'{sample}_optimal_paths.json').read_text())
        summary[sample]=dict(approximate_runs=len(o),approximate_unique_hard_references=int(o.reference_hash.nunique()),approximate_DI_min=float(o.DI.min()),approximate_DI_max=float(o.DI.max()),approximate_DI_range=float(o.DI.max()-o.DI.min()),exact_runs=len(e),exact_unique_hard_references=int(e.reference_hash.nunique()),exact_DI=float(e.iloc[0].DI),exact_DI_range=float(e.DI.max()-e.DI.min()),optimal_unoriented_paths=len(info['optimal_unoriented_paths']),optimal_oriented_paths=info['optimal_oriented_paths_count'],optimal_length_decimal=info['best_length_decimal'],next_unoriented_path_gap_decimal=info['runner_up_unoriented_length_gap_decimal'])
    (P/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(reversal).to_string(index=False));print(json.dumps(summary,indent=2))

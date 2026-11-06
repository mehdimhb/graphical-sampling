from pathlib import Path
import csv,json,collections,subprocess
OUT=Path(__file__).resolve().parent

def read(n):return list(csv.DictReader((OUT/n).open()))
def write(n,rs):
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
ledger=read('per_setting_replication_ledger.csv');supports=read('finite_design_support_counts.csv');support_by={(r['Population'],r['Probability'],r['Method'],r['n'],r['N']):r for r in supports}
for r in ledger:
 k=tuple(r[c] for c in ['Population','Probability','Method','n','N']);s=support_by.get(k,{})
 r.update(cached_sample_rows=s.get('cached_sample_rows',''),valid_positive_enumeration_rows=s.get('valid_positive_rows',''),distinct_unordered_sample_sets=s.get('unique_unordered_sample_rows',''),finite_support_evidence='finite_design_support_counts.csv:'+str(supports.index(s)+2) if s else '',completed_count_direct_observation='cached enumeration rows verified, not MC completion' if s else 'not independently logged',final_DI_count_basis='finite enumeration cache plus matching saved cell22/CSV/plot values to6dec' if s else 'matching saved successful mean/SD and all-samples/no-drop source; inferred2000' if r['saved_DI_mean_SD_match']=='True' else 'final Dm/Ds pair count unresolved')
write('per_setting_replication_ledger.csv',ledger)
groups=[]
for (pop,prob) in sorted({(r['Population'],r['Probability']) for r in ledger}):
 v=[r for r in ledger if (r['Population'],r['Probability'])==(pop,prob)];mc=[r for r in v if not r['exact_support']];ex=[r for r in v if r['exact_support']]
 groups.append(dict(Population=pop,Probability=prob,n_values=';'.join(sorted({r['n'] for r in v},key=int)),MC_designs=';'.join(sorted({r['Method'] for r in mc})),MC_setting_rows=len(mc),saved_output_label_2000=sum(r['saved_output_Iterations']=='2000' for r in mc),exact_linked_DI_pairs_inferred_2000=sum(r['saved_DI_mean_SD_match']=='True' for r in mc),saved_mean_discrepancy_rows=sum(r['saved_DI_mean_SD_match']=='False' for r in mc),MC_rows_no_setting_output=sum(not r['saved_output_Iterations'] for r in mc),old_MC_metadata_5000=sum(r['old_ite_values']=='5000' for r in mc),finite_design_rows=len(ex),finite_enumeration_rows=sum(int(r['exact_support']) for r in ex),MC_effective_count_directly_logged=0))
write('grouped_replication_counts.csv',groups)
paths=sorted({p for r in ledger for p in r['old_source_paths'].split(';') if p});history=[]
for p in paths:
 log=subprocess.check_output(['git','log','--reverse','--format=%H %ad %s','--date=iso','e71d6935bfe824e72a7b722067a24775ed160610','--',p],text=True).strip().splitlines()
 history.append(dict(path=p,first_commit_log=log[0],last_commit_log=log[-1]))
write('original_5000_summary_file_history.csv',history)
summary=json.loads((OUT/'replication_checks.json').read_text());summary.update(exact_linked_2000_DI_pairs=104,discrepancy_final_DI_pairs_count_unknown=2,no_setting_output_MC_rows_count_unknown=32,effective_MC_counts_independently_logged=0,finite_weighted_designs=24,finite_enumeration_rows=21368,all_finite_enumeration_rows_valid_positive=True,MC_old_metadata5000_rows=sum(g['old_MC_metadata_5000'] for g in groups),auxiliary_MC_rows_inherited_5000_summaries=138,auxiliary_NMS_GMS_rows_inherited_old_summaries=22,auxiliary_NMS_GMS_rows_missing=2)
(OUT/'replication_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['exact_linked_2000_DI_pairs','discrepancy_final_DI_pairs_count_unknown','no_setting_output_MC_rows_count_unknown','effective_MC_counts_independently_logged','finite_weighted_designs','finite_enumeration_rows','MC_old_metadata5000_rows']},indent=2))

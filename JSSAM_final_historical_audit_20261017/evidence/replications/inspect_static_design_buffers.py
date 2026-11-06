"""Decode saved numeric buffers using pickletools+struct; never unpickle or import project code."""
from pathlib import Path
import csv,json,subprocess,pickletools,struct,math,hashlib
OUT=Path(__file__).resolve().parent;REV='e71d6935bfe824e72a7b722067a24775ed160610';out=[]
for r in csv.DictReader((OUT/'historical_saved_design_DI.csv').open()):
 p=f"simulations/best_designs/{r['Population']}/{r['Design']}_design_df_{r['Population']}_{r['n']}_pik_{r['Probability'].lower()}.pkl"
 raw=subprocess.check_output(['git','show',REV+':'+p]);ops=list(pickletools.genops(raw));
 i=next(i for i,(o,a,q) in enumerate(ops) if a=='_all_samples_and_probs');j=next(j for j in range(i+1,len(ops)) if ops[j][1]=='_nht_variance')
 arrays=[(q,bytes(a)) for o,a,q in ops[i:j] if isinstance(a,(bytes,bytearray))]
 assert len(arrays)==2,(p,len(arrays))
 (sample_pos,sample_bytes),(prob_pos,prob_bytes)=arrays
 support=int(r['Support']);n=int(r['n']);N=int(r['N'])
 assert len(sample_bytes)==8*support*n and len(prob_bytes)==4*support,(p,len(sample_bytes),len(prob_bytes))
 vals=[t[0] for t in struct.iter_unpack('<q',sample_bytes)];probs=[t[0] for t in struct.iter_unpack('<f',prob_bytes)];samples=[tuple(vals[k:k+n]) for k in range(0,len(vals),n)]
 valid=[len(set(s))==n and all(0<=v<N for v in s) for s in samples];positive=[math.isfinite(q) and q>0 for q in probs]
 brief=[]
 for o,a,q in ops[i:j]:
  arg=(f'<{len(a)} bytes SHA256={hashlib.sha256(a).hexdigest()}>' if isinstance(a,(bytes,bytearray)) else str(a))
  brief.append(f'{q}: {o.name} {arg}')
 (OUT/(r['Population']+'_'+r['Probability']+'_'+r['Method']+'_'+r['n']+'_cache_opcodes.txt')).write_text('\n'.join(brief)+'\n')
 out.append(dict(Population=r['Population'],Probability=r['Probability'],Method=r['Method'],n=n,N=N,saved_DI_Support=support,cached_sample_rows=len(samples),cached_probability_rows=len(probs),valid_sample_rows=sum(valid),finite_positive_probability_rows=sum(positive),valid_positive_rows=sum(v and q for v,q in zip(valid,positive)),zero_probability_rows=sum(q==0 for q in probs),negative_probability_rows=sum(q<0 for q in probs),nonfinite_probability_rows=sum(not math.isfinite(q) for q in probs),unique_ordered_sample_rows=len(set(samples)),unique_unordered_sample_rows=len({tuple(sorted(s)) for s in samples}),probability_sum=math.fsum(probs),minimum_probability=min(probs),maximum_probability=max(probs),git_commit=REV,git_path=p,git_blob=subprocess.check_output(['git','rev-parse',REV+':'+p],text=True).strip(),sha256=hashlib.sha256(raw).hexdigest(),pickle_cache_key_byte_offset=ops[i][2],sample_buffer_opcode_byte_offset=sample_pos,probability_buffer_opcode_byte_offset=prob_pos,code_executed=False))
with (OUT/'finite_design_support_counts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(dict(designs=len(out),sum_saved_support=sum(r['saved_DI_Support'] for r in out),sum_valid_positive_rows=sum(r['valid_positive_rows'] for r in out),all_sample_rows_valid=all(r['valid_sample_rows']==r['saved_DI_Support'] for r in out),all_probability_rows_finite_positive=all(r['finite_positive_probability_rows']==r['saved_DI_Support'] for r in out),minimum_probability=min(r['minimum_probability'] for r in out),probability_sum_range=[min(r['probability_sum'] for r in out),max(r['probability_sum'] for r in out)],all_cached_supports_match_saved_output=True,unpickling_performed=False),indent=2))

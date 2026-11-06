"""Independent numerical audit of completed runs, sources and exact border assignment."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-hard-ip-balance")
from pathlib import Path
from fractions import Fraction
import math,json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
HOME=Path(__file__).resolve().parent

def verify(allow_unavailable_originals=False):
    checks=[];violations=[];unavailable=[]
    design={c["case_id"]:c for c in json.loads((HOME/"design.json").read_text())["cases"]}
    for f in sorted((HOME/'runs').glob('*/state.npz')):
        s=np.load(f);r=json.loads((f.parent/'summary.json').read_text());pi=s['pi'];n=r['n'];N=r['N'];labels=s['labels'];mu=s['mu'];P=s['P'];x=s['coords'];centers=s['centers'];omega=s['omega'];path=s['path']
        assert r['status']=='completed' and r['case_id']==f.parent.name
        case=design[r['case_id']]
        raw=pd.read_csv(HOME/f'populations/{case["stem"]}.csv')[['x','y']].to_numpy(float)
        normalized=(raw-raw.min(axis=0))/np.ptp(raw,axis=0).max()
        assert np.array_equal(x,normalized) and np.array_equal(s['sample'],case['sample'])
        assert np.array_equal(s['initial_centers'],x[s['sample']])
        assert np.array_equal(s['preliminary_labels'],np.argmax(P,axis=1))
        assert len(s['sample'])==n and len(np.unique(s['sample']))==n
        assert np.all((pi>0)&(pi<=1)) and abs(math.fsum(pi)-n)<1e-10
        assert np.array_equal(np.sort(omega),np.arange(N)) and np.array_equal(np.sort(path),np.arange(n))
        masses=np.array([math.fsum(pi[labels==k]) for k in range(n)]);counts=np.array([sum(labels==k) for k in range(n)])
        # Independent sweep using exact rational endpoints; compare all shares and tie assignments.
        prob=[Fraction(n,N)]*N if r['IP']=='EP' else [Fraction.from_float(float(p)) for p in pi]
        total=sum(prob,Fraction(0));prefix=Fraction(0);exact_labels=np.zeros(N,int);share_error=0.
        for unit in omega:
            hi=prefix+prob[unit];shares=[]
            for j,k in enumerate(path):
                end=Fraction(j+1) if j<n-1 else total
                overlap=max(Fraction(0),min(hi,end)-max(prefix,Fraction(j)))
                share=overlap/prob[unit];shares.append((share,int(k)))
                share_error=max(share_error,abs(float(share)-mu[unit,k]))
            best=max(a for a,k in shares);candidates=[k for a,k in shares if a==best]
            exact_labels[unit]=min(candidates,key=lambda k:int(s['sample'][k]))
            prefix=hi
        assert np.array_equal(exact_labels,labels) and share_error<1e-14
        exact_masses=[sum((prob[i] for i in np.flatnonzero(labels==k)),Fraction(0)) for k in range(n)]
        exact_errors=[abs(m-1) for m in exact_masses]
        assert max(exact_errors)<=max(prob)
        assert abs(100*sum(e>Fraction(1,10) for e in exact_errors)/n-r['pct_gt_010'])<1e-10
        assert abs(100*sum(e>Fraction(1,4) for e in exact_errors)/n-r['pct_gt_025'])<1e-10
        row=float(abs(P.sum(axis=1)-pi).max());col=float(abs(P.sum(axis=0)-1).max())
        fixed=float(np.linalg.norm((P.T@x)/P.sum(axis=0)[:,None]-centers))
        soft=float(abs(pi@mu-1).max());member=float(abs(mu.sum(axis=1)-1).max())
        assert row<=1.001e-10 and col<=1.001e-10 and fixed<=2e-7 and soft<1e-9 and member<1e-12
        assert abs(masses.sum()-n)<1e-10 and counts.sum()==N
        bound_gap=float(abs(masses-1).max()-pi.max());assert bound_gap<=1e-10
        cc=pd.read_csv(f.parent/'clusters.csv',float_precision='round_trip');assert np.array_equal(cc['size'],counts);assert np.allclose(cc.mass,masses,rtol=0,atol=1e-13)
        checks.append(dict(case_id=r['case_id'],row_error=row,column_error=col,fixed_point_residual=fixed,soft_mass_error=soft,row_membership_error=member,independent_overlap_error=share_error,
                           exact_bound_passed=True,exact_threshold_counts_agree=True,hard_mass_sum_error=float(abs(masses.sum()-n)),bound_gap=bound_gap,empty_clusters=int(np.sum(counts==0)),passed=True))
    for row in json.loads((HOME/'protected_hashes.json').read_text()):
        p=HOME.parent/row['path']
        if not p.exists():unavailable.append(row['path'])
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:violations.append(row['path'])
        if '/populations/' in row['path']:
            copied=HOME/'populations'/Path(row['path']).name
            assert hashlib.sha256(copied.read_bytes()).hexdigest()==row['sha256'],str(copied)
        if row['path'].startswith('graphical_sampling/'):
            copied=HOME/'sources'/row['path']
            assert hashlib.sha256(copied.read_bytes()).hexdigest()==row['sha256'],str(copied)
    assert not violations,violations
    assert allow_unavailable_originals or not unavailable,unavailable
    pd.DataFrame(checks).to_csv(HOME/'verification.csv',index=False,float_format='%.17g')
    summary=pd.read_csv(HOME/'summary.csv',float_precision='round_trip');cl=pd.read_csv(HOME/'cluster_measurements.csv',float_precision='round_trip')
    assert len(summary)==60 and set(summary.population)=={'Gridded','Random','Clustered','Meuse','Swiss'}
    for _,r in summary.iterrows():
        c=cl[(cl.population==r.population)&(cl.IP==r.IP)&(cl.n==r.n)]
        assert len(c)==r.completed_replicates*r.n
        if len(c):
            assert abs(c.abs_error.mean()-r.mean_abs_error)<1e-13
            assert abs(c.abs_error.max()-r.max_abs_error)<1e-13
            assert abs(np.std(c.mass,ddof=0)-r.sd_mass)<1e-13
            assert abs(100*np.mean(c.abs_error>.1+1e-12)-r.pct_gt_010)<1e-10
            assert abs(100*np.mean(c.abs_error>.25+1e-12)-r.pct_gt_025)<1e-10
    result=dict(completed_runs=len(checks),measured_clusters=len(cl),summary_settings=len(summary),protected_sources=len(json.loads((HOME/'protected_hashes.json').read_text())),protected_source_changes=violations,unavailable_originals=unavailable,protected_originals_fully_verified=not unavailable,
                all_numerical_checks_passed=True,empty_hard_clusters=sum(c['empty_clusters'] for c in checks),largest_OT_row_residual=max(c['row_error'] for c in checks),
                largest_OT_column_residual=max(c['column_error'] for c in checks),largest_fixed_point_residual=max(c['fixed_point_residual'] for c in checks),largest_soft_mass_error=max(c['soft_mass_error'] for c in checks))
    (HOME/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--allow-unavailable-originals',action='store_true');args=parser.parse_args()
    verify(args.allow_unavailable_originals)

"""Geometry-independent EP predictions for ordered maximum-share hard assignment.

These are exact arithmetic predictions, not measurements from unconverged OT.
With q=N/n and r=N mod n there are n-r clusters of floor(q) units and r
clusters of ceil(q) units. Integer thresholds cannot tie; half thresholds
alternate with integer thresholds, so any exact-half tie rule preserves this
cardinality multiset. Each unit has pi=n/N.
"""
from pathlib import Path
from fractions import Fraction
import math
import pandas as pd
HOME=Path(__file__).resolve().parent

def compute():
    rows=[]
    for _,s in pd.read_csv(HOME/'probability_audit.csv').query("IP=='EP'").iterrows():
        N,n=int(s.N),int(s.n);r=N%n;lo=(N//n)*Fraction(n,N);hi=((N+n-1)//n)*Fraction(n,N)
        mean=Fraction(2*r*(n-r),n*N)
        rows.append(dict(population=s.population,N=N,n=n,fraction=n/N,type='exact EP arithmetic; independent of OT convergence',
            min_mass=float(lo),max_mass=float(hi),mean_abs_error=float(mean),max_abs_error=float(max(abs(lo-1),abs(hi-1))),sd_mass=math.sqrt(r*(n-r))/N,
            low_size=N//n,low_count=n-r,high_size=(N+n-1)//n,high_count=r,
            pct_gt_010=100*((n-r)*(abs(lo-1)>Fraction(1,10))+r*(abs(hi-1)>Fraction(1,10)))/n,
            pct_gt_025=100*((n-r)*(abs(lo-1)>Fraction(1,4))+r*(abs(hi-1)>Fraction(1,4)))/n))
    out=pd.DataFrame(rows);out.to_csv(HOME/'EP_arithmetic_predictions.csv',index=False,float_format='%.17g')
    if (HOME/'cluster_measurements.csv').exists():
        cl=pd.read_csv(HOME/'cluster_measurements.csv');count=0
        for key,c in cl.query("IP=='EP'").groupby(['population','n','replicate']):
            pop,n,rep=key;p=out[(out.population==pop)&(out.n==n)].iloc[0]
            assert int((c['size']==p.low_size).sum())==(n if p.low_size==p.high_size else p.low_count)
            if p.low_size!=p.high_size:assert int((c['size']==p.high_size).sum())==p.high_count
            assert abs(c.mass.min()-p.min_mass)<1e-13 and abs(c.mass.max()-p.max_mass)<1e-13
            assert abs(c.abs_error.mean()-p.mean_abs_error)<1e-13
            count+=1
        print('Exact EP arithmetic verified against',count,'converged EP runs')

if __name__=='__main__':compute()

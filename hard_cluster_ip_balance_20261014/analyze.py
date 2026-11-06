"""Create summary tables and figures from completed, converged balance runs."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-hard-ip-balance")
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HOME=Path(__file__).resolve().parent
COLORS={'EP':'#1768a5','UP':'#ce662b'}
POPS=['Gridded','Random','Clustered','Meuse','Swiss']

def save_csv(df,path):df.to_csv(path,index=False,float_format='%.17g')

def analyze():
    design=json.loads((HOME/'design.json').read_text()); audits=pd.read_csv(HOME/'probability_audit.csv')
    rows=[];clusters=[]
    for c in design['cases']:
        p=HOME/'runs'/c['case_id']/'summary.json'
        if p.exists():
            r=json.loads(p.read_text());assert r['case_id']==c['case_id']
        else:
            first=HOME/'runs'/c['case_id'].replace(f'_r{c["replicate"]}','_r1')/'summary.json'
            reason='not attempted'
            if first.exists() and json.loads(first.read_text())['status']!='completed':reason='not attempted after first-replicate convergence failure'
            r={**{k:v for k,v in c.items() if k not in ('sample','cache','stem','valid')},'fraction':c['n']/c['N'],'status':reason}
        r['sample_unit_ids']=json.dumps([i+1 for i in c['sample']]);rows.append(r)
        p=HOME/'runs'/c['case_id']/'clusters.csv'
        if r['status']=='completed':
            df=pd.read_csv(p,float_precision='round_trip')
            for key in ['case_id','population','N','n','IP','replicate','fraction']:df[key]=r[key]
            clusters.append(df)
    runs=pd.DataFrame(rows);allcl=pd.concat(clusters,ignore_index=True)
    save_csv(runs,HOME/'run_summary.csv');save_csv(allcl,HOME/'cluster_measurements.csv')
    summaries=[]
    for _,a in audits.iterrows():
        rr=runs[(runs.population==a.population)&(runs.n==a.n)&(runs.IP==a.IP)]
        cc=allcl[(allcl.population==a.population)&(allcl.n==a.n)&(allcl.IP==a.IP)]
        out=dict(population=a.population,N=int(a.N),n=int(a.n),IP=a.IP,fraction=a.fraction,
            max_pi=a.pi_max,min_pi=a.pi_min,pi_sum=a.pi_sum,probability_vector_valid=bool(a.valid),
            certainty_units=int(a.certainty_units),simple_proportional_valid=bool(a.simple_proportional_valid) if a.IP=='UP' else True,
            simple_proportional_max_pi=a.simple_proportional_max if a.IP=='UP' else a.pi_max,
            probability_construction=a.construction,completed_replicates=int((rr.status=='completed').sum()),
            attempted_replicates=int(rr.status.isin(['completed','cpu_limit','failed','worker_exit']).sum()),planned_replicates=3,
            incomplete_details='; '.join(f'r{r.replicate}: {r.status}'+(' ('+str(r.issue)+')' if isinstance(r.get('issue'),str) else '') for _,r in rr.iterrows() if r.status!='completed'))
        if len(cc):
            e=cc.abs_error.to_numpy();m=cc.mass.to_numpy()
            out.update(min_mass=float(m.min()),max_mass=float(m.max()),mean_abs_error=float(e.mean()),max_abs_error=float(e.max()),sd_mass=float(np.std(m,ddof=0)),
                pct_gt_010=float(100*np.mean(e>.1+1e-12)),pct_gt_025=float(100*np.mean(e>.25+1e-12)),min_size=int(cc['size'].min()),max_size=int(cc['size'].max()),
                measured_clusters=len(cc),empty_clusters=int((cc['size']==0).sum()),bound_pass=bool(e.max()<=a.pi_max+1e-10),
                mean_error_min_over_replicates=float(rr[rr.status=='completed'].mean_abs_error.min()),mean_error_max_over_replicates=float(rr[rr.status=='completed'].mean_abs_error.max()),
                max_soft_mass_error=float(rr[rr.status=='completed'].soft_mass_max_error.max()))
        summaries.append(out)
    summary=pd.DataFrame(summaries);save_csv(summary,HOME/'summary.csv')
    compact=[]
    for pop in POPS:
        for ip in ['EP','UP']:
            t=summary[(summary.population==pop)&(summary.IP==ip)];ok=t[t.completed_replicates>0];small=ok[ok.fraction<=.1];large=ok[ok.fraction>.1]
            compact.append(dict(population=pop,IP=ip,N=int(t.N.iloc[0]),completed_settings=len(ok),requested_settings=len(t),completed_runs=int(t.completed_replicates.sum()),
                total_runs=18,valid_proportional_settings=int(t.simple_proportional_valid.sum()),
                observed_mass_min=ok.min_mass.min(),observed_mass_max=ok.max_mass.max(),
                at_fraction_le_010_max_MAE=small.mean_abs_error.max(),at_fraction_le_010_worst_error=small.max_abs_error.max(),
                at_fraction_gt_010_max_MAE=large.mean_abs_error.max(),at_fraction_gt_010_worst_error=large.max_abs_error.max(),
                observed_empty_clusters=int(ok.empty_clusters.sum()),max_pi=ok.max_pi.max()))
    save_csv(pd.DataFrame(compact),HOME/'compact_summary.csv')
    figures(allcl,summary)
    print('Completed',int((runs.status=='completed').sum()),'runs;',len(allcl),'clusters;',int((summary.completed_replicates>0).sum()),'settings')
    print(summary[['population','n','IP','completed_replicates','mean_abs_error','max_abs_error','pct_gt_010','pct_gt_025']].to_string(index=False))
    return summary,runs,allcl

def figures(cl,summary):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,2,figsize=(12,14),sharey=True)
    for i,pop in enumerate(POPS):
        for j,ip in enumerate(['EP','UP']):
            ax=axes[i,j];t=summary[(summary.population==pop)&(summary.IP==ip)];ns=t.n.to_numpy()
            ax.axhspan(.75,1.25,color='#d9d9d9',alpha=.35);ax.axhspan(.9,1.1,color='#8fba94',alpha=.18);ax.axhline(1,color='black',lw=.9)
            for position,n in enumerate(ns):
                c=cl[(cl.population==pop)&(cl.IP==ip)&(cl.n==n)]
                if c.empty:
                    ax.text(position,1,'×',ha='center',color='#9b111e',fontsize=15)
                else:
                    values=c.mass.to_numpy();rng=np.random.default_rng(20261014+i*100+j*10+int(n))
                    ax.scatter(position+rng.uniform(-.13,.13,len(c)),values,s=7,color=COLORS[ip],alpha=.35,edgecolors='none')
                    ax.boxplot([values],positions=[position],widths=.45,whis=(0,100),showfliers=False,patch_artist=True,
                        boxprops={'facecolor':'none','edgecolor':COLORS[ip]},medianprops={'color':COLORS[ip]},whiskerprops={'color':COLORS[ip]},capprops={'color':COLORS[ip]})
            ax.set_title(f'{pop} · {ip}',loc='left');ax.set_xticks(np.arange(len(ns)),[str(n) for n in ns]);ax.set_xlabel('Sample size n');ax.set_xlim(-.6,len(ns)-.4)
            if j==0:ax.set_ylabel('Hard-cluster IP mass')
            ax.grid(axis='y',alpha=.15);ax.set_ylim(-.03,2.06)
    fig.suptitle('Hard IPnM balance: target mass = 1\nDots are clusters; boxes show pooled converged replicates; × = no converged run',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ['png','pdf']:fig.savefig(HOME/f'mass_distributions.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7.7),sharey=True)
    for ax,pop in zip(axes.flat,POPS):
        for ip in ['EP','UP']:
            t=summary[(summary.population==pop)&(summary.IP==ip)].sort_values('fraction');valid=t.completed_replicates>0
            ax.plot(t.fraction,t.mean_abs_error,color=COLORS[ip],marker='o',markersize=4,label=f'{ip} mean')
            ax.plot(t.fraction,t.max_abs_error,color=COLORS[ip],ls='--',marker='s',markersize=4,label=f'{ip} maximum')
            ax.scatter(t.loc[~valid,'fraction'],np.full((~valid).sum(),-.020 if ip=='EP' else -.040),color=COLORS[ip],marker='x',s=25)
        ax.set_title(pop,loc='left');ax.set_xlabel('Sampling fraction n/N');ax.set_xlim(0,1);ax.grid(alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Absolute deviation from mass 1')
    handles,labels=axes[0,0].get_legend_handles_labels();axes[1,2].axis('off');axes[1,2].legend(handles,labels,loc='upper left',frameon=False)
    axes[1,2].text(0,.35,'Mean: all clusters in converged replicates\nMaximum: worst observed cluster\n×: no converged OT solution\nMissing settings break the curves.\n\nUP follows the existing Measure code;\ncertainty adjustment is flagged in CSV.',transform=axes[1,2].transAxes,va='top')
    axes[0,0].set_ylim(-.045,1.05);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(HOME/f'imbalance_vs_fraction.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':analyze()

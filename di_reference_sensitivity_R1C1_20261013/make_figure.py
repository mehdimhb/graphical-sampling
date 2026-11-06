"""One figure, three computed references, one fixed observed sample."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
data=pd.read_csv(P/'scored_references.csv',float_precision='round_trip').set_index('run_id')
names=['s2_official_00','s2_official_02','s2_stress_04'];titles=['Official, TSP seed 0','Official, TSP seed 2','Alternative initialization 4 (stress test)']
fig,axes=plt.subplots(1,3,figsize=(13.8,4.8),layout='constrained');settings=[]
for ax,name,title in zip(axes,names,titles):
    st=np.load(P/'runs'/name/'state.npz');xy=st['raw'];s=st['observed_sample'];m=st['medoids'];r=data.loc[name]
    padding=.04*np.ptp(xy,axis=0).max();lo=xy.min(axis=0)-padding;hi=xy.max(axis=0)+padding
    scatter=ax.scatter(*xy.T,c=st['q'],cmap='coolwarm',vmin=-1,vmax=1,s=24,alpha=.9)
    ax.scatter(*xy[s].T,s=112,facecolors='none',edgecolors='black',linewidths=1.6,label='Fixed sample')
    ax.scatter(*xy[m].T,s=84,marker='x',color='black',linewidths=1.5,label='Reference medoids')
    ax.set_aspect('equal');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_xlabel('x (original coordinates)');ax.set_ylabel('y (original coordinates)')
    ax.set_title(f'{title}\nJ={r.J_OT_native:.8g}; Qsoft={r.Q_soft:.8g}\nDnet={r.Dnet:.8g}; eta={r.eta:.8g}; DI={r.DI:.8g}',fontsize=9)
    ax.legend(loc='upper left',fontsize=7)
    settings.append(dict(panel=title,run_id=name,source_population_csv='runs/'+name+'/population_density.csv',source_matching_csv='runs/'+name+'/reference_matching.csv',xmin=lo[0],xmax=hi[0],ymin=lo[1],ymax=hi[1],color_field='signed_contribution',color_vmin=-1,color_vmax=1,colormap='coolwarm'))
fig.colorbar(scatter,ax=axes,label='Signed unit density contribution',shrink=.8)
fig.suptitle('Same observed sample S2 and fixed scoring settings; alternative computed references',fontsize=11)
for suffix in ['pdf','png']:fig.savefig(P/('reference_comparison.'+suffix),dpi=300,bbox_inches='tight')
pd.DataFrame(settings).to_csv(P/'figure_panels.csv',index=False,float_format='%.17g')
(P/'figure_caption.txt').write_text('All panels use the identical S2 observed sample, N=144 clustered population, n=8 UP inclusion vector, covariance-based fixed Scott box bandwidth, and original DI. Circles are the fixed sample and crosses the computed medoids. Population-unit colors show actual signed density contributions evaluated at original versus cluster-translated unit locations. The first two references use official sample-initialized OT with different randomized TSP runs; the third is a separately labeled initialization stress test. Raw coordinates, memberships, matching, translated coordinates and density contributions are saved under runs/.')
print(P/'reference_comparison.pdf')

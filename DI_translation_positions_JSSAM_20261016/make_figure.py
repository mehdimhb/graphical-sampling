"""Four saved placements of the same sample; no optimization."""
import os
os.environ['MPLCONFIGDIR']='/tmp/jssam-comment5-positions-20261016/mpl'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
P=Path(__file__).resolve().parent
original=json.loads((P/'runs/original/summary.json').read_text())
places=json.loads((P/'placements.json').read_text())
entries=[dict(id='original',label='Original: bottom-left',delta=[0,0],comparison=dict(new_to_old=list(range(8)),DI_difference=0,changed_units=0))]+places
raw=np.array([(x,y) for y in range(12) for x in range(12)])
colors=plt.get_cmap('tab10')(np.arange(8))
fig,axes=plt.subplots(2,2,figsize=(12.5,13.8),sharex=True,sharey=True)
for panel,(ax,entry) in enumerate(zip(axes.flat,entries)):
 z=json.loads((P/'runs'/entry['id']/'summary.json').read_text());c=entry['comparison']
 mapping=np.array(c['new_to_old']);labels=mapping[np.array(z['labels'])];grid=labels.reshape(12,12)
 ax.pcolormesh(np.arange(-.5,12.5),np.arange(-.5,12.5),grid,cmap=ListedColormap(colors),
   vmin=0,vmax=7,alpha=.15,shading='flat')
 ax.scatter(raw[:,0],raw[:,1],c=colors[labels],s=15,linewidths=0,zorder=2)
 edges=[]
 for y in range(12):
  for x in range(12):
   if x<11 and grid[y,x]!=grid[y,x+1]:edges.append([(x+.5,y-.5),(x+.5,y+.5)])
   if y<11 and grid[y,x]!=grid[y+1,x]:edges.append([(x-.5,y+.5),(x+.5,y+.5)])
 ax.add_collection(LineCollection(edges,colors='#777',lw=.75,zorder=3))
 meds=raw[z['medoid_ids']];matched=raw[z['matched_ids']]
 for r,t in zip(meds,matched):ax.plot([r[0],t[0]],[r[1],t[1]],ls=(0,(3,2)),color='#666',alpha=.6,lw=.7,zorder=4)
 sample=np.array(z['sample'])
 ax.scatter(sample[:,0],sample[:,1],s=120,facecolors='white',edgecolors='#c72f2f',linewidths=1.8,zorder=6)
 ax.scatter(meds[:,0],meds[:,1],s=65,c='black',marker='x',linewidths=1.6,zorder=7)
 for k,p in enumerate(meds):
  ax.annotate(f'$r_{{{mapping[k]+1}}}$',p,xytext=(4,5),textcoords='offset points',fontsize=7.5,
   bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.3),zorder=8)
 ax.set_aspect('equal');ax.set_xlim(-.6,11.6);ax.set_ylim(-.6,11.6)
 ax.set_xticks([0,3,6,9,11]);ax.set_yticks([0,3,6,9,11]);ax.tick_params(labelsize=9,labelbottom=True,labelleft=True)
 ax.set_xlabel('Grid x',fontsize=10);ax.set_ylabel('Grid y',fontsize=10)
 title=f"({chr(97+panel)}) {entry['label']}  |  shift {tuple(entry['delta'])}"
 ax.set_title(title+f"\nDI = {z['DI']:.10f}"+f"\nChanged memberships: {c['changed_units']}/144;  DI difference: {c['DI_difference']:+.10f}",fontsize=11,pad=11,linespacing=1.4)
fig.suptitle('The same concentrated sample in four positions',fontsize=18,y=.985)
fig.text(.5,.957,r'Fixed $12\times12$ population • $N=144$, $n=8$, $\pi_\ell=1/18$ • independent IPnM fits for each new position',ha='center',fontsize=10.5)
legend=[Line2D([],[],color='none',marker='o',markerfacecolor='#777',markeredgecolor='none',markersize=5,label='Population unit (cluster color)'),
 Line2D([],[],color='none',marker='o',markerfacecolor='white',markeredgecolor='#c72f2f',markeredgewidth=1.8,markersize=8,label='Selected sample unit'),
 Line2D([],[],color='none',marker='x',markeredgecolor='black',markeredgewidth=1.8,markersize=7,label='Medoid'),
 Line2D([],[],color='#666',lw=.8,ls=(0,(3,2)),label='Squared-distance matching')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,fontsize=9.5,frameon=False)
fig.text(.5,.019,'Cluster colors aligned with the original partition. The population stays fixed; only the eight sample locations change.',ha='center',fontsize=9,color='#444')
fig.subplots_adjust(left=.065,right=.985,bottom=.095,top=.885,hspace=.40,wspace=.13)
fig.savefig(P/'four_positions.png',dpi=220,bbox_inches='tight',facecolor='white')
fig.savefig(P/'four_positions.pdf',bbox_inches='tight',facecolor='white')
plt.close(fig)
print(P/'four_positions.png')

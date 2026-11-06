"""Plot only saved states; never reruns reference construction."""
import os
os.environ['MPLCONFIGDIR']='/tmp/jssam-comment5-concentrated-20261016/mpl'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
P=Path(__file__).resolve().parent
s=json.loads((P/'selection.json').read_text())['selected']
a=json.loads((P/'runs'/s['original_id']/'summary.json').read_text())
b=json.loads((P/'runs'/s['translated_id']/'summary.json').read_text())
raw=np.array([(x,y) for y in range(12) for x in range(12)],int)
colors=plt.get_cmap('tab10')(np.arange(8))
fig,axes=plt.subplots(1,2,figsize=(12.5,8.1),sharex=True,sharey=True)
for panel,(ax,z) in enumerate(zip(axes,[a,b])):
    original_labels=np.array(z['labels']);labels=original_labels if panel==0 else np.array(s['new_to_old'])[original_labels]
    grid=labels.reshape(12,12)
    from matplotlib.colors import ListedColormap
    ax.pcolormesh(np.arange(-.5,12.5,1),np.arange(-.5,12.5,1),grid,
       cmap=ListedColormap(colors),vmin=0,vmax=7,alpha=.15,shading='flat',rasterized=False)
    ax.scatter(raw[:,0],raw[:,1],c=colors[labels],s=22,linewidths=0,zorder=2)
    boundaries=[]
    for y in range(12):
      for x in range(12):
        if x<11 and grid[y,x]!=grid[y,x+1]:boundaries.append([(x+.5,y-.5),(x+.5,y+.5)])
        if y<11 and grid[y,x]!=grid[y+1,x]:boundaries.append([(x-.5,y+.5),(x+.5,y+.5)])
    ax.add_collection(LineCollection(boundaries,colors='#6a6a6a',linewidths=.85,alpha=.7,zorder=3))
    med=raw[z['medoid_ids']];matched=raw[z['matched_ids']]
    for r,t in zip(med,matched):ax.plot([r[0],t[0]],[r[1],t[1]],color='#5e5e5e',lw=.8,alpha=.65,ls=(0,(3,2)),zorder=4)
    sample=np.array(z['sample'])
    if panel==1:
      old=np.array(a['sample'])
      for p,q in zip(old[:1],sample[:1]):
        ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',color='#b62828',alpha=.75,lw=1.6,shrinkA=7,shrinkB=8),zorder=5)
    ax.scatter(sample[:,0],sample[:,1],s=150,facecolors='white',edgecolors='#c72f2f',linewidths=2,zorder=6)
    ax.scatter(med[:,0],med[:,1],s=78,c='black',marker='x',linewidths=1.8,zorder=7)
    for k,p in enumerate(med):
      label=k+1 if panel==0 else s['new_to_old'][k]+1
      ax.annotate(f'$r_{{{label}}}$',p,xytext=(5,6),textcoords='offset points',fontsize=9,
          color='#222',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.4),zorder=8)
    ax.set_aspect('equal');ax.set_xlim(-.6,11.6);ax.set_ylim(-.6,11.6)
    ax.set_xticks(np.arange(0,12,2));ax.set_yticks(np.arange(0,12,2))
    ax.tick_params(labelsize=10);ax.set_xlabel('Grid x',fontsize=11)
    ax.set_title(('(a) Concentrated original sample' if panel==0 else '(b) Shifted sample; IPnM fitted again')+
       f"\nDI = {z['DI']:.10f}",fontsize=13,pad=12)
axes[0].set_ylabel('Grid y',fontsize=11)
fig.suptitle('Large translation of a concentrated sample',fontsize=17,y=.975)
fig.text(.5,.925,rf'$12\times12$ grid  •  $N=144$, $n=8$, $\pi_\ell=1/18$  •  shift $\delta=({s["delta"][0]},{s["delta"][1]})$',ha='center',fontsize=12)
handles=[Line2D([],[],color='none',marker='o',markerfacecolor='#777',markeredgecolor='none',markersize=5,label='Population unit (cluster color)'),
 Line2D([],[],color='none',marker='o',markerfacecolor='white',markeredgecolor='#c72f2f',markeredgewidth=1.8,markersize=9,label='Selected sample unit'),
 Line2D([],[],color='none',marker='x',markeredgecolor='black',markeredgewidth=1.8,markersize=8,label='Medoid'),
 Line2D([],[],color='#5e5e5e',lw=1,ls=(0,(3,2)),label='Squared-distance matching')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.13),ncol=4,frameon=False,fontsize=10,
 handlelength=1.8,columnspacing=1.5)
fig.text(.5,.086,f"DI difference = {s['DI_difference']:+.10f}     |     Changed memberships = {s['changed_units']} / 144",ha='center',fontsize=12,weight='bold')
fig.text(.5,.032,'Colors aligned by maximum partition overlap. The red arrow shows the common sample shift.\nEach hard cluster contains 18 population units; medoids and density calculations were recomputed.',
 ha='center',fontsize=9.5,color='#444',linespacing=1.5)
fig.subplots_adjust(left=.065,right=.985,bottom=.23,top=.82,wspace=.12)
fig.savefig(P/'translation_comparison.png',dpi=240,bbox_inches='tight',facecolor='white')
fig.savefig(P/'translation_comparison.pdf',bbox_inches='tight',facecolor='white')
plt.close(fig)
print(P/'translation_comparison.png')

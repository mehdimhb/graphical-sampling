"""Rearrange saved Figure 9 graphics; no scientific computation or fitting.

Requires the existing Python/Matplotlib environment and qpdf/mutool commands.
The source data graphics are embedded directly from the original 2x2 PDF.
"""
import os
os.environ['MPLCONFIGDIR']='/tmp/jssam-figure9-one-row-mpl'
import base64
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
SOURCE=ROOT/'simulations/results/temporary_2D_relative_rotation/grid_20x20_N400_DI_original_shift_rotation90_rotation45_large_2x2_n20.pdf'
PDF=OUT/'translation_rotation_n20_one_row.pdf'
PNG=PDF.with_suffix('.png')
THREADS=max(1,(len(os.sched_getaffinity(0))+1)//2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_axes():
    trace=subprocess.check_output(['mutool','draw','-F','trace',str(SOURCE)],stderr=subprocess.DEVNULL)
    root=ET.fromstring(trace)
    vertical=[]
    for element in root.iter('stroke_path'):
        if element.get('linewidth')!='.8' or element.get('linecap')!='2,2,2':
            continue
        points=[(float(e.get('x')),float(e.get('y'))) for e in element if e.tag in ['moveto','lineto']]
        if len(points)!=2:
            continue
        a,b,c,d,tx,ty=map(float,element.get('transform').split())
        q=np.array(points)@np.array([[a,b],[c,d]])+[tx,ty]
        if abs(q[0,0]-q[1,0])<1e-8 and abs(q[0,1]-q[1,1])>300:
            vertical.append(q)
    xs=sorted(set(float(q[0,0]) for q in vertical))
    ys=sorted(set(float(y) for q in vertical for x,y in q))
    assert len(xs)==len(ys)==4
    return [(xs[0],ys[0],xs[1],ys[1]),(xs[2],ys[0],xs[3],ys[1]),
            (xs[0],ys[2],xs[1],ys[3]),(xs[2],ys[2],xs[3],ys[3])]


def make_labels(path):
    width,height=1022.4,325.
    side=210.
    left,bottom,gap=40.,58.,34.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.linewidth':.8,'axes.grid':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(width/72,height/72))
    headings=['(a) Original sample','(b) Uniformly shifted sample',
              r'(c) Sample rotated $90^\circ$',r'(d) Sample rotated $45^\circ$']
    subtitles=['Starting configuration','Same shift for every sample point',
               'Counterclockwise about the sample center','Counterclockwise about the sample center']
    # Retain the values printed in the saved source; do not invent extra precision.
    stats=['DI = 0.00; MI = −0.47; VI = 0.01; BI = 0.04',
           'DI = 0.00; MI = −0.44; VI = 0.03; BI = 0.08',
           'DI = 0.00; MI = −0.47; VI = 0.01; BI = 0.04',
           'DI = −0.14; MI = −0.23; VI = 0.15; BI = 0.19']
    rectangles=[]
    for k in range(4):
        x=left+k*(side+gap)
        rectangles.append([x,bottom,x+side,bottom+side])
        ax=fig.add_axes([x/width,bottom/height,side/width,side/height],facecolor='none')
        ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel=r'$c_1$',ylabel=r'$c_2$')
        ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
        ax.yaxis.label.set_rotation(0);ax.yaxis.labelpad=8
        ax.tick_params(length=2.5,width=.6,pad=2)
        if k in [2,3]:
            angle=90 if k==2 else 45
            ax.text(.5,.52,rf'${angle}^\circ$',color='#00747B',fontsize=10.7,
                    ha='center',va='center',fontweight='bold',zorder=10,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.82,pad=1))
        center=(x+side/2)/width
        fig.text(center,306/height,headings[k],ha='center',va='center',fontsize=10,fontweight='bold')
        fig.text(center,292/height,subtitles[k],ha='center',va='center',fontsize=8.5)
        fig.text(center,279/height,stats[k],ha='center',va='center',fontsize=8.5,fontweight='bold')
    handles=[
        Line2D([],[],marker='o',ls='',mfc='#E53935',mec='black',mew=.5,ms=5,label='sampled unit'),
        Line2D([],[],marker='o',ls='',mfc='none',mec='.38',mew=1,ms=5.5,label='original sample position'),
        Line2D([],[],marker='x',ls='',color='black',mew=1.2,ms=6,label='IPnM medoid'),
        Line2D([],[],color='#008C95',lw=1,marker='>',ms=4,label='original → transformed'),
        Line2D([],[],color='black',ls='--',lw=.8,label='medoid → matched unit'),
    ]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),
               ncol=5,frameon=False,fontsize=9,columnspacing=1.7,handlelength=2)
    fig.savefig(path,transparent=True)
    plt.close(fig)
    return width,height,rectangles,stats


def main():
    before=digest(SOURCE)
    axes=source_axes()
    with tempfile.TemporaryDirectory(prefix='figure9_layout_') as tmp:
        tmp=Path(tmp)
        labels=tmp/'labels.pdf'
        width,height,destinations,stats=make_labels(labels)
        merged=tmp/'merged.pdf'
        subprocess.run(['qpdf','--empty','--pages',str(SOURCE),'1',str(labels),'1','--',str(merged)],check=True)
        document=json.loads(subprocess.check_output(['qpdf','--json','--json-stream-data=inline',str(merged)]))
        objects=document['qpdf'][1]
        page_refs=[page['object'] for page in document['pages']]
        pages=[objects['obj:'+ref]['value'] for ref in page_refs]
        next_id=document['qpdf'][0]['maxobjectid']+1

        def add_object(value):
            nonlocal next_id
            ref=f'{next_id} 0 R'
            objects['obj:'+ref]=value
            next_id+=1
            return ref

        def form_for(page,strip_old_text=False):
            refs=page['/Contents']
            if isinstance(refs,str):refs=[refs]
            content=b'\n'.join(base64.b64decode(objects['obj:'+ref]['stream']['data']) for ref in refs)
            if strip_old_text:
                # Remove old headings/ticks before cropping, so PDF text extraction
                # cannot recover duplicate hidden labels from the source Form.
                # Geometry, colors, markers, images and matching lines are retained.
                content=re.sub(rb'\bBT\b.*?\bET\b',b'',content,flags=re.S)
                content=re.sub(rb'/F\d+-DejaVuSans[^\s]*\s+Do',b'',content)
            return add_object({'stream':{'dict':{'/Type':'/XObject','/Subtype':'/Form',
                '/BBox':page['/MediaBox'],'/Resources':page['/Resources']},
                'data':base64.b64encode(content).decode()}})

        source_height=pages[0]['/MediaBox'][3]
        source_form=form_for(pages[0],strip_old_text=True)
        labels_form=form_for(pages[1])
        commands=[]
        for (sx0,sy0,sx1,sy1),(dx0,dy0,dx1,dy1) in zip(axes,destinations):
            # Trace coordinates have a top-left origin; PDF coordinates have a
            # bottom-left origin. Crop exactly the original axes data viewport.
            sy_bottom=source_height-sy1
            scale=(dx1-dx0)/(sx1-sx0)
            tx=dx0-scale*sx0;ty=dy0-scale*sy_bottom
            commands.append(f'q {dx0} {dy0} {dx1-dx0} {dy1-dy0} re W n '
                            f'{scale} 0 0 {scale} {tx} {ty} cm /Source Do Q')
        commands.append('q /Labels Do Q')
        stream=add_object({'stream':{'dict':{},'data':base64.b64encode('\n'.join(commands).encode()).decode()}})
        pages[0]['/MediaBox']=[0,0,width,height]
        pages[0]['/Resources']={'/XObject':{'/Source':source_form,'/Labels':labels_form}}
        pages[0]['/Contents']=stream
        pages[0].pop('/Annots',None)
        document['qpdf'][0]['maxobjectid']=next_id-1
        spec=tmp/'document.json';spec.write_text(json.dumps(document))
        staged=tmp/'staged.pdf'
        subprocess.run(['qpdf','--json-input',str(spec),str(staged)],check=True)
        subprocess.run(['qpdf',str(staged),'--pages','.', '1','--',str(PDF)],check=True)
    subprocess.run(['mutool','draw','-q','-r','400','-B','128','-T',str(THREADS),'-o',str(PNG),str(PDF)],check=True)
    assert digest(SOURCE)==before
    (OUT/'caption.txt').write_text(
        'Translation and grid-projected rotation for a fixed 20 × 20 population '
        '(N = 400, n = 20). Panels show (a) the original sample, '
        '(b) translation by (1/19, −1/19), and targets rotated '
        '(c) 90° and (d) 45° counterclockwise about the sample center, projected '
        'to distinct population-grid units. Colored points and pale polygons show '
        'the saved cluster configurations. Red circles denote sampled units, '
        'hollow circles their original positions, black crosses the medoids, '
        'teal arrows the movement, and dashed lines the medoid-to-sample matching. '
        'Plotted configurations and displayed index values are retained from the '
        'original saved figure; this is a layout change only.\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=before,
        layout='one row, four panels; shared legend below',
        scientific_calculations_rerun=0,source_graphics='original PDF axes embedded as Form XObjects',
        displayed_statistics=stats,rendering_threads=THREADS,
        available_cpus=len(os.sched_getaffinity(0)),source_axes_top_origin=axes,
        destination_axes_pdf_origin=destinations),indent=2)+'\n')
    print(PDF);print(PNG)


if __name__=='__main__':main()

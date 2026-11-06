"""Compare saved figure geometry and recover synthetic DI bars; no simulation."""
from pathlib import Path
import os
for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="1"
import re,subprocess,json,xml.etree.ElementTree as E,io,math
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent;EV=OUT/'evidence'
def polygons(path,kind):
 q=[]
 for e in E.parse(path).iter():
  fill=e.get('fill','');numbers=[float(x) for x in re.findall(r'-?\d*\.?\d+(?:e[-+]?\d+)?',e.get('d',''))]
  match=fill.startswith('rgb(') and fill not in ['rgb(100%, 100%, 100%)','rgb(0%, 0%, 0%)'] if kind=='solid' else fill.startswith('url(')
  if e.tag.endswith('path') and match and len(numbers)==10:
   transform=[float(v) for v in re.findall(r'-?\d*\.?\d+',e.get('transform',''))]
   if transform:
    assert len(transform)==6 and transform[1]==transform[2]==0
    for j in [1,3,5,7,9]:numbers[j]=numbers[j]*transform[3]+transform[5]
   q.append(numbers)
 return np.array(q)
def main():
 master=pd.read_csv(io.BytesIO(subprocess.check_output(['git','-C',str(ROOT),'show','e71d6935b:simulations/results/master_summary_DI_nmedoids_updated.csv'])))
 expectations={}
 for prob in ['Equal','Unequal']:
  vals=[]
  for pop in ['Gridded','Random','Clustered']:
   for method in ['MAX','SCP','LPV','WAV']:
    for n in [4,8,16,32]:
     row=master[(master.Population==pop)&(master.Probability==prob)&(master.Method==method)&(master.n==n)];assert len(row)==1
     vals.append(float(row.iloc[0].Dm))
  expectations[prob]=np.array(vals)
 records=[]
 def axis_scale(file):
  t=E.parse(EV/file.replace('.svg','_bbox.html'));words=[]
  for e in t.iter():
   if e.tag.endswith('word') and e.text and re.fullmatch(r'-?0\.[036]',e.text) and 15<float(e.get('xMin',0))<100 and float(e.get('xMax',0))<100:
    words.append((float(e.get('yMin')),float(e.text),(float(e.get('yMin'))+float(e.get('yMax')))/2))
  ticks=sorted(words)[:4];assert [v for y,v,m in ticks]==[.3,0,-.3,-.6]
  slope=float(np.polyfit([v for y,v,m in ticks],[m for y,v,m in ticks],1)[0])
  # Cairo normalizes the SVG canvas through the integer display-page width.
  width=float(E.parse(EV/file).getroot().get('width').replace('pt',''));canvas_scale=width/math.ceil(width)
  return -slope*canvas_scale,[dict(value=v,y=m) for y,v,m in ticks]

 for file in ['original_synthetic.svg','submission_synthetic.svg','proof_synthetic.svg']:
  for prob,kind in [('Equal','solid'),('Unequal','hatched')]:
   q=polygons(EV/file,kind)[:48];assert len(q)==48
   # The first metric row has 48 bars in population/method/n order.
   coordinates=np.round(np.r_[q[:,1],q[:,5]],6);baseline=float(pd.Series(coordinates).mode().iloc[0]);height=2*baseline-q[:,1]-q[:,5];target=expectations[prob];scale,ticks=axis_scale(file);recovered=height/scale;err=float(abs(recovered-target).max())
   assert err<1e-4,(file,prob,err)
   records.append(dict(file=file,probability=prob,bar_count=48,tick_calibrated_points_per_DI_unit=scale,calibration_ticks=ticks,maximum_DI_recovery_difference=err,method='Axis scale comes from the displayed 0.3, 0, -0.3, -0.6 tick positions, independently of the expected DI vector. Transform rectangle coordinates to SVG page coordinates; account for Cairo integer-canvas normalization. Tolerance 1e-4 DI units allows PDF coordinate quantization, not equality at six-decimal CSV precision.'))
 result=dict(all_passed=True,DI_bars_checked_per_figure=96,records=records,proof_pdf='/home/bardia/Downloads/JSSAM-2025-203.R1_Proof_hi_0.pdf',proof_page=31,scope='Geometric identity at rendering precision supports the link to these reported figures; it is not a per-run execution log or proof for every other manuscript output.')
 (OUT/'figure_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

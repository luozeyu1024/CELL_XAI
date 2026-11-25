"""Aggregate full graph profiles, keeping source-training calibration separate."""
import argparse
import numpy as np,pandas as pd
from cumulative_common import *
METRICS=['n_nodes','n_edges','density','isolated_fraction','largest_component_fraction','n_components','selected_mass_fraction','eligible_mass_ceiling']
def summarize(f,scope,fold='all'):
 rows=[]
 for (ds,q,r),g in f.groupby(['dataset','threshold','relation'],sort=False):
  row=dict(dataset=ds,threshold=q,relation=r,scope=scope,fold=fold,n_cells=len(g),attainment_rate=float(g.attained.mean()))
  for name in METRICS:
   v=g[name];row[name+'_mean']=v.mean()
   for label,quant in [('p25',.25),('median',.5),('p75',.75),('p95',.95),('p99',.99),('max',1.)]:row[name+'_'+label]=v.quantile(quant)
  rows.append(row)
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--allow_partial',action='store_true');a=p.parse_args();inputs=pd.read_csv(ROOT/'work/signature_coverage/source_data/inputs.tsv',sep='\t');rows=[]
 for ds in inputs.dataset:
  path=FIG/'source_data/profiles'/f'{ds}.tsv.gz'
  if not path.exists():
   if a.allow_partial:continue
   raise ValueError('Missing graph profiles: '+ds)
  f=pd.read_csv(path,sep='\t');assert not f.duplicated(['cell_id','threshold','relation']).any()
  assert len(f)==16*int(inputs.set_index('dataset').loc[ds,'n_cells'])
  assert (f.n_edges<=f.n_nodes*(f.n_nodes-1)/2).all()
  rows+=summarize(f,'all_cells')
  for job in [j for j in jobs() if j['dataset']==ds]:
   cells=read_cells(job);train=cells.loc[cells.split.eq('train'),'cell_id'];subset=f[f.cell_id.isin(train)];assert subset.cell_id.nunique()==len(train)
   rows+=summarize(subset,'source_training_only',job['fold'])
 pd.DataFrame(rows).to_csv(FIG/'source_data/graph_summary.tsv',sep='\t',index=False)
 print(pd.DataFrame(rows).query("scope=='source_training_only' and relation=='state_15b'")[['dataset','fold','threshold','n_nodes_median','n_edges_median','isolated_fraction_median','largest_component_fraction_median']].round(3).to_string(index=False))
if __name__=='__main__':main()

"""Validate mass-prefix graph construction using raw IG and independent graph traversal."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,networkx as nx
from scipy import sparse
from cumulative_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');a=p.parse_args();records=[]
 inputs=pd.read_csv(ROOT/'work/signature_coverage/source_data/inputs.tsv',sep='\t')
 for r in inputs.itertuples():
  if a.datasets and r.dataset not in a.datasets:continue
  folder=DATA/'candidates'/r.dataset
  if not (folder/'parameters.json').exists():continue
  raw=sparse.load_npz(r.attribution_matrix).tocsr();elig=pd.read_csv(folder/'eligible_genes.tsv',sep='\t').sort_values('gene_symbol');ptr=np.load(folder/'ptr.npy');ids=np.load(folder/'gene_id.npy',mmap_mode='r');counts=np.load(folder/'counts.npy');attained=np.load(folder/'attained.npy');totals=np.asarray(raw.sum(axis=1,dtype=np.float64)).ravel();assert np.all(np.diff(ptr)>=counts.max(axis=1))
  profile=pd.read_csv(FIG/'source_data/profiles'/f'{r.dataset}.tsv.gz',sep='\t');cells=pd.read_csv(folder/'cells.tsv',sep='\t');chosen=sorted(set([0,len(cells)//2,len(cells)-1]))
  failed=np.flatnonzero(~attained.all(axis=1))
  if len(failed):chosen=sorted(set(chosen+[int(failed[0])]))
  checked=0
  for i in chosen:
   values=raw[i,elig.attribution_column.to_numpy(int)].toarray().ravel().astype(np.float64);order=np.argsort(-values,kind='stable');order=order[values[order]>0];expected=elig.gene_id.to_numpy()[order];v=values[order];cum=np.cumsum(v);lo,hi=ptr[i:i+2];assert np.array_equal(ids[lo:hi],expected[:hi-lo])
   for j,q in enumerate(Q):
    k=counts[i,j]
    if attained[i,j]:assert cum[k-1]>=q*totals[i] and (k==1 or cum[k-2]<q*totals[i])
    else:assert cum[-1]<q*totals[i] and k==len(cum)
   for rel in RELATIONS:
    g=sparse.load_npz(DATA/'global_graphs'/f'{rel}.npz').tocsr()
    for j,q in enumerate(Q):
     sub=g[expected[:counts[i,j]]][:,expected[:counts[i,j]]].tocsr();graph=nx.from_scipy_sparse_array(sub);components=list(nx.connected_components(graph));row=profile[profile.cell_id.eq(cells.cell_id.iloc[i])&profile.threshold.eq(q)&profile.relation.eq(rel)].iloc[0]
     assert row.n_edges==graph.number_of_edges();assert row.n_components==len(components);assert np.isclose(row.isolated_fraction,len(list(nx.isolates(graph)))/len(graph));assert np.isclose(row.largest_component_fraction,max(map(len,components))/len(graph));checked+=1
  if (folder/'expression_validation.json').exists():
   d=CumulativeGraphs(r.dataset,'state_15b',.6);graph=d[0];graph.validate(raise_on_error=True);assert graph.x.shape==(graph.num_nodes,1) and torch.isfinite(graph.x).all()
  records.append(dict(dataset=r.dataset,raw_prefix_cells=len(chosen),independent_networkx_graphs=checked,status='pass'))
 pd.DataFrame(records).to_csv(FIG/'source_data/construction_validation.tsv',sep='\t',index=False);print('PASS',len(records),'datasets; raw prefix ordering/minimality and NetworkX components/edges/isolates verified.',flush=True)
if __name__=='__main__':main()

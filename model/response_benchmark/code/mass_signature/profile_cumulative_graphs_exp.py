"""Profile induced graphs for four cumulative SCimilarity-IG thresholds."""
import argparse,filecmp,json,os,sys,shutil,time
from pathlib import Path
from project_paths import project_root
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
ROOT=project_root()
FIG=ROOT/'work/cumulative_signature'
DATA=ROOT/'gene_sim_pathway/data/cumulative_signature_graphs'
RELATIONS=['state_15b','ppi','qwen','esm2_650m'];Q=[.5,.6,.7,.8]
def write_json(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def profile_dataset(row, graph_root):
 ds=row['dataset'];dest=DATA/'candidates'/ds;dest.mkdir(parents=True,exist_ok=True)
 output=FIG/'source_data/profiles'/f'{ds}.tsv.gz';output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists() and (dest/'parameters.json').exists():
  prior=json.loads((dest/'parameters.json').read_text(encoding='utf-8'))
  if (prior['reference_graph'],prior['attribution_matrix'],prior['attribution_manifest']) != (row['graph_package'],row['attribution_matrix'],row['manifest']):
   raise ValueError('Existing candidate profile uses different inputs: '+ds)
  return ds,'existing'
 start=time.perf_counter();elig=pd.read_csv(Path(row['graph_package'])/'eligible_genes.tsv',sep='\t').sort_values('gene_symbol').reset_index(drop=True)
 cells=pd.read_csv(Path(row['graph_package'])/'cells.tsv',sep='\t',keep_default_na=False)
 a=sparse.load_npz(row['attribution_matrix']).tocsr();a.sum_duplicates();a.eliminate_zeros()
 m=json.loads(Path(row['manifest']).read_text(encoding='utf-8'))['input'];assert cells.cell_id.tolist()==m['cell_ids_used']
 assert np.isfinite(a.data).all() and (a.data>=0).all()
 total=np.asarray(a.sum(axis=1,dtype=np.float64)).ravel();e=a[:,elig.attribution_column.to_numpy(int)].tocsr();e.sort_indices()
 geneids=elig.gene_id.to_numpy(np.int32)
 graphs={r:sparse.load_npz(graph_root/'global_graphs'/f'{r}.npz').tocsr() for r in RELATIONS}
 for r,g in graphs.items():assert (g!=g.T).nnz==0 and not np.any(g.diagonal())
 vocab=pd.read_csv(graph_root/'gene_vocab.tsv',sep='\t').gene_symbol.to_numpy()
 assert vocab[geneids].tolist()==elig.gene_symbol.tolist()
 ptr=[0];nodeids=[];scores=[];counts=[];masses=[];flags=[];rows=[]
 for i,cid in enumerate(cells.cell_id):
  lo,hi=e.indptr[i:i+2];v=e.data[lo:hi].astype(np.float64);idx=e.indices[lo:hi];positive=v>0;v=v[positive];idx=idx[positive]
  order=np.argsort(-v,kind='stable');v=v[order];ids=geneids[idx[order]];cum=np.cumsum(v,dtype=np.float64)
  if total[i]<=0 or not len(v):raise ValueError(ds+': invalid/empty eligible signature '+cid)
  cut=np.searchsorted(cum,np.asarray(Q)*total[i],side='left');ok=cut<len(v);n=np.where(ok,cut+1,len(v)).astype(int)
  mass=np.asarray([cum[k-1]/total[i] for k in n]);ceiling=cum[-1]/total[i]
  maxn=int(n.max());ptr.append(ptr[-1]+maxn);nodeids.append(ids[:maxn]);scores.append(v[:maxn].astype(np.float32));counts.append(n);masses.append(mass);flags.append(ok)
  for relation,g in graphs.items():
   big=g[ids[:maxn]][:,ids[:maxn]].tocsr()
   for j,q in enumerate(Q):
    k=n[j];sub=big[:k,:k];degree=np.diff(sub.indptr);components,lab=connected_components(sub,directed=False);edges=sub.nnz//2
    rows.append(dict(dataset=ds,cell_id=cid,threshold=q,relation=relation,n_nodes=k,n_edges=edges,density=2*edges/(k*(k-1)) if k>1 else 0.,isolated_fraction=float(np.mean(degree==0)),largest_component_fraction=float(np.bincount(lab).max()/k),n_components=components,attained=bool(ok[j]),selected_mass_fraction=mass[j],eligible_mass_ceiling=ceiling))
  if (i+1)%5000==0:print(ds,i+1,'cells profiled',flush=True)
 np.save(dest/'ptr.npy',np.asarray(ptr,np.int64));np.save(dest/'gene_id.npy',np.concatenate(nodeids));np.save(dest/'importance.npy',np.concatenate(scores));np.save(dest/'counts.npy',np.asarray(counts,np.int32));np.save(dest/'mass.npy',np.asarray(masses,np.float64));np.save(dest/'attained.npy',np.asarray(flags,bool));np.save(dest/'totals.npy',total)
 cells.to_csv(dest/'cells.tsv',sep='\t',index=False);elig.to_csv(dest/'eligible_genes.tsv',sep='\t',index=False)
 pd.DataFrame(rows).to_csv(output,sep='\t',index=False,compression='gzip')
 write_json(dest/'parameters.json',dict(dataset=ds,thresholds=Q,n_cells=len(cells),attribution_matrix=row['attribution_matrix'],attribution_manifest=row['manifest'],reference_graph=row['graph_package'],denominator='full_original_model_importance',unattainable='all_positive_eligible_nodes_retained_and_flagged',rank_ties='gene_symbol_ascending',graph_statistics_seconds=time.perf_counter()-start))
 print('PROFILE_COMPLETE',ds,'seconds',round(time.perf_counter()-start),flush=True);return ds,'complete'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--graph_path',type=Path,default=ROOT/'work/cell_graphs')
 p.add_argument('--attribution_path',type=Path,default=ROOT/'work/attributions')
 p.add_argument('--workers',type=int,default=2)
 p.add_argument('--datasets',nargs='+',required=True)
 args=p.parse_args()
 rows=[]
 for dataset in args.datasets:
  attr=args.attribution_path/dataset
  rows.append(dict(dataset=dataset, graph_package=str((args.graph_path/dataset).resolve()),
                   attribution_matrix=str((attr/'attributions/scimilarity_ig.npz').resolve()),
                   manifest=str((attr/'manifests/scimilarity_ig_manifest.json').resolve())))
  for key in ('graph_package','attribution_matrix','manifest'):
   if not Path(rows[-1][key]).exists(): raise FileNotFoundError(rows[-1][key])
  rows[-1]['n_cells']=len(json.loads(Path(rows[-1]['manifest']).read_text(encoding='utf-8'))['input']['cell_ids_used'])
 DATA.mkdir(parents=True,exist_ok=True);(DATA/'global_graphs').mkdir(exist_ok=True)
 for relative in ['gene_vocab.tsv',*(f'global_graphs/{r}.npz' for r in RELATIONS)]:
  source=args.graph_path/relative
  target=DATA/relative
  if target.exists():
   if not filecmp.cmp(source,target,shallow=False):
    raise ValueError('Existing cumulative package uses a different global graph: '+relative)
  else:
   shutil.copy2(source,target)
 shutil.copy2(args.graph_path/'global_graphs/parameters.json',DATA/'global_graphs/parameters.json')
 manifest=ROOT/'work/signature_coverage/source_data/inputs.tsv'
 manifest.parent.mkdir(parents=True,exist_ok=True)
 inputs=pd.DataFrame(rows)
 if manifest.exists():
  previous=pd.read_csv(manifest,sep='\t')
  inputs=pd.concat([previous.loc[~previous.dataset.isin(args.datasets)], inputs],ignore_index=True)
 inputs.to_csv(manifest,sep='\t',index=False)
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(profile_dataset,r,args.graph_path) for r in rows]
  for f in as_completed(futures):print(f.result(),flush=True)
if __name__=='__main__':main()

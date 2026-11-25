"""Extract unchanged official aligned log-normalized expression for candidate nodes."""
import argparse,importlib.util,json,os
from pathlib import Path
from project_paths import project_root
import numpy as np,pandas as pd
from scipy import sparse
import anndata as ad
ROOT=project_root();DATA=ROOT/'gene_sim_pathway/data/cumulative_signature_graphs'
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/numba_mass_signature')
Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
def run(ds,counts_root=None):
 p=DATA/'candidates'/ds
 if (p/'expression.npy').exists() and (p/'expression_validation.json').exists():print('EXPRESSION_EXISTS',ds,flush=True);return
 meta=json.loads((p/'parameters.json').read_text(encoding='utf-8'));m=json.loads(Path(meta['attribution_manifest']).read_text(encoding='utf-8'))['input']
 source=Path(m['h5ad'])
 if not source.exists() and counts_root:source=Path(counts_root)/ds/'counts.h5ad'
 if not source.exists():raise FileNotFoundError('Complete raw counts required: '+str(source))
 print('READ_COUNTS',ds,flush=True);raw=ad.read_h5ad(source);raw=raw[np.asarray(m['cell_indices_used'],int)].copy();assert raw.obs_names.astype(str).tolist()==m['cell_ids_used'];raw.var_names_make_unique()
 names=json.loads(Path(meta['attribution_matrix']).with_name('scimilarity_ig_var_names.json').read_text(encoding='utf-8'))
 counts=sparse.csr_matrix(raw.layers['counts'] if 'counts' in raw.layers else raw.X,dtype=np.float32)
 assert np.isfinite(counts.data).all() and (counts.data>=0).all() and np.allclose(counts.data,np.rint(counts.data),rtol=0,atol=1e-6)
 spec=importlib.util.spec_from_file_location('official_signature_utils',ROOT/'assets/SIGnature/src/SIGnature/utils.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 x=ad.AnnData(X=counts,obs=raw.obs.copy(),var=raw.var.copy());x.layers['counts']=counts;x=u.align_dataset(x,names,keep_obsm=False,gene_overlap_threshold=500);x=u.lognorm_counts(x);x=sparse.csr_matrix(x.X)
 assert np.isfinite(x.data).all() and (x.data>=0).all()
 elig=pd.read_csv(p/'eligible_genes.tsv',sep='\t');vocab=pd.read_csv(DATA/'gene_vocab.tsv',sep='\t');mapping=np.full(len(vocab),-1,np.int32);mapping[elig.gene_id]=elig.attribution_column
 ptr=np.load(p/'ptr.npy');ids=np.load(p/'gene_id.npy',mmap_mode='r');expr=np.lib.format.open_memmap(p/'expression.npy',mode='w+',dtype=np.float32,shape=(len(ids),))
 for i in range(len(ptr)-1):
  lo,hi=ptr[i:i+2];cols=mapping[ids[lo:hi]];assert (cols>=0).all();expr[lo:hi]=x[i,cols].toarray().ravel()
 expr.flush()
 # Compare every original selected node for beginning/middle/end cells.
 old=Path(meta['reference_graph']);z=np.load(old/'signatures.npz');checks=0
 for i in sorted(set([0,(len(ptr)-1)//2,len(ptr)-2])):
  a,b=z['ptr'][i:i+2];oldids=z['gene_id'][a:b];oldexpr=z['features'][a:b,0];expected=x[i,mapping[oldids]].toarray().ravel();assert np.allclose(expected,oldexpr,rtol=1e-5,atol=1e-6);checks+=len(oldids)
 (p/'expression_validation.json').write_text(json.dumps(dict(counts_source=str(source),official_alignment_and_lognorm=True,original_nodes_checked=checks,expression='counts aligned to model genes; per-cell 10k and log1p; no signature renormalization'),indent=2)+'\n',encoding='utf-8')
 print('EXPRESSION_COMPLETE',ds,len(ids),'node values',flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');p.add_argument('--counts_root');args=p.parse_args()
 datasets=args.datasets or pd.read_csv(ROOT/'work/signature_coverage/source_data/inputs.tsv',sep='\t').dataset.tolist()
 for ds in datasets:run(ds,args.counts_root)
if __name__=='__main__':main()




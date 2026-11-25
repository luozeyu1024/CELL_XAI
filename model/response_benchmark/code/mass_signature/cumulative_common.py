"""Shared paths, frozen training designs and lazy cumulative-signature graphs."""
import json,sys
from pathlib import Path
from project_paths import project_root
import numpy as np,pandas as pd,torch
from scipy import sparse
from torch.utils.data import Dataset,Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
ROOT=project_root();BENCH=ROOT/'model/response_benchmark';DATA=ROOT/'gene_sim_pathway/data/cumulative_signature_graphs';FIG=ROOT/'work/cumulative_signature';OUT=BENCH/'output/cumulative_signature'
sys.path.insert(0,str(BENCH/'code'))
from gene_embedding_benchmark_exp import jobs
from graph_model import GraphClassifier
Q=[.5,.6,.7,.8];RELATIONS=['state_15b','ppi','qwen','esm2_650m']
from split_registry import load_cells

def key(job):return job['dataset']+'__'+job['fold']
def read_cells(job):return load_cells(Path(job['cohort_root'])/'data/cohorts'/job['dataset']/'cells.tsv',job['dataset'],mode='source')
def policy(job):
 allpolicy=json.loads((FIG/'source_data/selected_policies.json').read_text(encoding='utf-8'));return allpolicy[key(job)]
def dest(job,relation):return OUT/relation/('within' if job['family']=='within' else 'grouped')/job['dataset']/job['fold']/'gat_gine'
class CumulativeGraphs(Dataset):
 def __init__(self,ds,relation,q,cells=None):
  self.ds=ds;self.relation=relation;self.q=q;self.j=Q.index(float(q));self.p=DATA/'candidates'/ds
  self.source_cells=pd.read_csv(self.p/'cells.tsv',sep='\t',keep_default_na=False);self.cells=self.source_cells if cells is None else cells
  assert self.source_cells.cell_id.is_unique and self.cells.cell_id.is_unique
  idx=pd.Series(np.arange(len(self.source_cells)),index=self.source_cells.cell_id);self.index=idx.loc[self.cells.cell_id].to_numpy(int)
  self.ptr=np.load(self.p/'ptr.npy',mmap_mode='r');self.gene_id=np.load(self.p/'gene_id.npy',mmap_mode='r');self.expr=np.load(self.p/'expression.npy',mmap_mode='r');self.counts=np.load(self.p/'counts.npy',mmap_mode='r');self.mass=np.load(self.p/'mass.npy',mmap_mode='r');self.attained=np.load(self.p/'attained.npy',mmap_mode='r')
  self._cache={}
  self.genes=pd.read_csv(DATA/'gene_vocab.tsv',sep='\t').gene_symbol.tolist();self.adjacency=sparse.load_npz(DATA/'global_graphs'/f'{relation}.npz').tocsr()
 def __len__(self):return len(self.cells)
 def __getitem__(self,i):
  if i in self._cache:return self._cache[i]
  row=self.index[i];lo=self.ptr[row];n=int(self.counts[row,self.j]);ids=np.array(self.gene_id[lo:lo+n],dtype=np.int64);x=np.array(self.expr[lo:lo+n],dtype=np.float32).reshape(-1,1);sub=self.adjacency[ids][:,ids].tocoo();cell=self.cells.iloc[i]
  item=Data(x=torch.from_numpy(x),gene_id=torch.from_numpy(ids),edge_index=torch.from_numpy(np.stack([sub.row,sub.col]).astype(np.int64)),edge_attr=torch.from_numpy(sub.data.astype(np.float32)[:,None]),y=torch.tensor([int(cell.get('y',-1))]),num_nodes=n,cell_id=cell.cell_id)
  self._cache[i]=item
  return item
def loader(data,cells,index,kind,batch_size,shuffle=False,seed=42):
 return DataLoader(Subset(data,[int(x) for x in index]),batch_size=batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0)
def stats(data,indices,n_genes):
 total=square=0.;count=0;seen=np.zeros(n_genes,bool)
 for i in indices:
  row=data.index[i];lo=data.ptr[row];n=data.counts[row,data.j];x=data.expr[lo:lo+n].astype(np.float64);total+=x.sum();square+=(x*x).sum();count+=len(x);seen[data.gene_id[lo:lo+n]]=True
 return float(total/count),float(max(square/count-(total/count)**2,1e-12)**.5),seen

from contextlib import contextmanager
@contextmanager
def gpu_slot():
 """Serialize this experiment's GPU training and memory benchmarks on Windows."""
 import ctypes
 from ctypes import wintypes
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR];kernel.CreateMutexW.restype=wintypes.HANDLE;kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD;kernel.ReleaseMutex.argtypes=[wintypes.HANDLE];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
 handle=kernel.CreateMutexW(None,False,'Local\\CumulativeSignatureGPU');result=kernel.WaitForSingleObject(handle,0xFFFFFFFF)
 if result not in (0,0x80):raise ctypes.WinError(ctypes.get_last_error())
 try:yield
 finally:kernel.ReleaseMutex(handle);kernel.CloseHandle(handle)


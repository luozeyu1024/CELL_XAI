"""Shared no-edge controls for the frozen cumulative-IG policies and original 14 designs."""
import argparse,gc,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from torch_geometric.data import Data
from cumulative_common import *
import train_response_exp as core
import train_cumulative_benchmark_exp as full
from edge_null_graphs import remove_edges

CONTROL='no_edges';TABLES=FIG/'source_data/edge_ablation'
ORIGINAL_FORWARD=core.forward

class NoEdgeGraphs(CumulativeGraphs):
 def __init__(self,ds,relation,q,cells=None):
  super().__init__(ds,'state_15b',q,cells)
 def __getitem__(self,i):
  if i in self._cache:return self._cache[i]
  row=self.index[i];lo=self.ptr[row];n=int(self.counts[row,self.j]);cell=self.cells.iloc[i]
  item=Data(x=torch.from_numpy(np.array(self.expr[lo:lo+n],dtype=np.float32).reshape(-1,1)),gene_id=torch.from_numpy(np.array(self.gene_id[lo:lo+n],dtype=np.int64)),edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,1),dtype=torch.float32),y=torch.tensor([int(cell.get('y',-1))]),num_nodes=n,cell_id=cell.cell_id)
  assert n>0 and item.edge_index.shape==(2,0) and item.edge_attr.shape==(0,1)
  self._cache[i]=item;return item
from split_registry import load_cells


def checked_forward(model,batch,kind,device):
 assert kind=='gat_gine' and batch.edge_index.shape==(2,0) and batch.edge_attr.shape==(0,1),'No-edge batch contains edges'
 return ORIGINAL_FORWARD(model,batch,kind,device)

def install(job,relation=CONTROL,threshold=None):
 q=policy(job)['threshold'] if threshold is None else threshold
 def load(root,ds,kind,rel=CONTROL):
  cells=load_cells(Path(root)/'data/cohorts'/ds/'cells.tsv',ds)
  data=NoEdgeGraphs(ds,CONTROL,q,cells);return data,cells,data.genes
 core.load_cohort=load;core.batches=loader;core.training_statistics=stats;core.forward=checked_forward
 core.output_root=lambda args,root:full.destination(job,CONTROL)

def old_folder(job):
 if job['family']=='within':return BENCH/'output/transfer/edge_ablation/no_edges/shared/within'/job['dataset']/'gat_gine'
 return BENCH/'output/grouped_adaptive'/job['dataset']/job['fold']/'no_edges/within'/job['dataset']/'gat_gine'

def preflight():
 TABLES.mkdir(parents=True,exist_ok=True);rows=[];tasks=jobs();assert len(tasks)==14
 cases={(j['dataset'],policy(j)['threshold']) for j in tasks}
 q=policy(next(j for j in tasks if j['dataset']=='GSE223779'))['threshold']
 cases|={(ds,q) for ds in ['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']}
 cases|={('GSE154600_EPI_external',policy(j)['threshold']) for j in tasks if j['dataset']=='GSE165897_pre_EOC'}
 for ds,q in sorted(cases):
  data=NoEdgeGraphs(ds,CONTROL,q);sample=sorted({0,len(data)//2,len(data)-1})
  for rel in RELATIONS:
   reference=CumulativeGraphs(ds,rel,q)
   for i in sample:
    a,b=data[i],reference[i]
    assert torch.equal(a.x,b.x) and torch.equal(a.gene_id,b.gene_id) and a.num_nodes==b.num_nodes and a.cell_id==b.cell_id
    assert a.edge_index.numel()==0 and a.edge_attr.numel()==0
   del reference
  rows.append(dict(dataset=ds,mass_threshold=q,n_cells=len(data),n_node_occurrences=int(data.counts[data.index,data.j].sum()),n_independent_feature_checks=len(sample)*4,n_edges_after=0,status='pass'));del data;gc.collect()
 pd.DataFrame(rows).to_csv(TABLES/'graph_input_validation.tsv',sep='\t',index=False)
 for j in tasks:
  for rel in RELATIONS:
   m=json.loads((full.trial_folder(j,rel)/'metrics.json').read_text());assert m['mass_threshold']==policy(j)['threshold'] and m['gene_feature_dimension']==16 and 'checkpoint_reload_max_margin_error' in m
 print('PREFLIGHT',len(rows),'dataset-policy pairs; nodes/expression match all four full graphs',flush=True)

def train(job,device):
 read_cells(job)
 torch.set_num_threads(4);install(job);folder=full.trial_folder(job,CONTROL);record=folder/'metrics.json'
 if record.exists() and 'checkpoint_reload_max_margin_error' in json.loads(record.read_text()):
  m=json.loads(record.read_text());assert m['mass_threshold']==policy(job)['threshold'] and m['edge_variant']==CONTROL
  print('VERIFIED_CONTROL_EXISTS',key(job),flush=True);return
 args=SimpleNamespace(seed=42,hidden=64,layers=2,heads=4,gene_dim=16,dropout=.2,lr=.001,weight_decay=.0001,graph_batch_size=32,mlp_batch_size=256,epochs=100,patience=15,min_delta=.0001,device=device,threads=4,relation=CONTROL,ablation=CONTROL,signature_method='cumulative_full_model_IG',mass_threshold=policy(job)['threshold'],policy_source=str(FIG/'source_data/selected_policies.json'),signature_candidate_path=str(DATA/'candidates'/job['dataset']),gene_vocab_path=str(DATA/'gene_vocab.tsv'))
 torch.cuda.reset_peak_memory_stats();core.train_one(args,Path(job['cohort_root']),job['dataset'],'gat_gine')
 cells=read_cells(job);pred=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
 for col in cells:
  if col not in pred:pred[col]=cells[col].to_numpy()
 pred.to_csv(folder/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
 model,ck=core.restore(folder/'checkpoint.pt',device);reference=torch.load(full.trial_folder(job,'state_15b')/'checkpoint.pt',map_location='cpu',weights_only=False)
 assert ck['training_cell_ids']==reference['training_cell_ids'] and ck['validation_cell_ids']==reference['validation_cell_ids']
 assert ck['model_config']==reference['model_config'] and ck['gene_symbols']==reference['gene_symbols'] and ck['class_weights']==reference['class_weights']
 assert ck['training_observed_gene_ids']==reference['training_observed_gene_ids']
 data,actual,genes=core.load_cohort(Path(job['cohort_root']),job['dataset'],'gat_gine',CONTROL)
 from torch.utils.data import Subset
 test=np.flatnonzero(cells.split.eq('test'));fresh=core.predict(model,Subset(data,test.tolist()),cells.iloc[test].reset_index(drop=True),'gat_gine',device,32)
 saved=pred.set_index('cell_id').loc[fresh.cell_id];error=float(np.abs(fresh.margin_resistant_minus_sensitive.to_numpy()-saved.margin_resistant_minus_sensitive.to_numpy()).max());assert error<=1e-4
 m=json.loads(record.read_text())
 for split in ['train','validation','test']:
  scores=core.metrics(pred[pred.split.eq(split)])
  for name in ['auroc','balanced_accuracy','auprc_resistant','mcc','brier_score']:assert np.isclose(scores[name],m[split][name],atol=1e-10)
 m.update(edge_variant=CONTROL,shared_no_edges_model=True,signature_method='cumulative_full_model_IG',mass_threshold=args.mass_threshold,graph_relation=CONTROL,gene_feature_dimension=16,trainable_parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_reload_max_margin_error=error,training_and_inference_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,scope='original exploratory random-cell split' if job['family']=='within' else 'original biological-group-held-out split; test cells, with formal group aggregation reported separately')
 core.write_json(record,m);print('VERIFIED_CUMULATIVE_NO_EDGES',key(job),m['test']['auroc'],flush=True)

def complete_old_reference(device):
 """Only GSE140440 is missing among the original fixed-K no-edge controls."""
 ds='GSE140440';job=next(j for j in jobs() if j['dataset']==ds);folder=old_folder(job)
 base=json.loads((BENCH/'output/within'/ds/'gat_gine/run_parameters.json').read_text())
 args=SimpleNamespace(**dict(base,stage='train',datasets=[ds],models=['gat_gine'],relation='state_15b',ablation=CONTROL,device=device))
 standard=core.load_cohort
 def load(root,ds,kind,relation='state_15b'):
  data,cells,genes=standard(root,ds,kind,relation)
  for g in data:remove_edges(g)
  return data,cells,genes
 core.load_cohort=load;core.output_root=lambda args,root:folder.parents[2];core.forward=checked_forward
 core.train_one(args,BENCH,ds,'gat_gine');model,ck=core.restore(folder/'checkpoint.pt',device);data,cells,genes=load(BENCH,ds,'gat_gine')
 fresh=core.predict(model,data,cells,'gat_gine',device,32);saved=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip');assert saved.cell_id.tolist()==cells.cell_id.tolist();error=float(np.abs(fresh.margin_resistant_minus_sensitive-saved.margin_resistant_minus_sensitive).max());assert error<=1e-4
 m=json.loads((folder/'metrics.json').read_text());m.update(edge_variant=CONTROL,shared_no_edges_model=True,fixed_k=1536,checkpoint_reload_max_margin_error=error);core.write_json(folder/'metrics.json',m)
 print('VERIFIED_MISSING_OLD_NO_EDGES',m['test']['auroc'],flush=True)

def transfer(device):
 import transfer_cumulative_exp as target
 full.CumulativeGraphs=NoEdgeGraphs;full.install_loader=install;target.install_loader=install
 full.transfer([CONTROL],device)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','preflight','train','worker','old_reference','transfer'],default='all');p.add_argument('--datasets',nargs='+');p.add_argument('--fold');p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
 if a.stage in ['all','preflight']:preflight()
 if a.stage=='worker':
  tasks=[j for j in jobs() if j['dataset'] in a.datasets and j['fold']==a.fold];assert len(tasks)==1
  with gpu_slot():train(tasks[0],a.device)
  return
 if a.stage in ['all','old_reference']:
  with gpu_slot():complete_old_reference(a.device)
  if a.stage=='all':
   # A fresh process avoids retaining the fixed-K loader for cumulative fitting.
   subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','train','--device',a.device],check=True)
 if a.stage=='train':
  for j in jobs():
   if a.datasets and j['dataset'] not in a.datasets:continue
   folder=full.trial_folder(j,CONTROL);folder.mkdir(parents=True,exist_ok=True)
   if (folder/'metrics.json').exists() and 'checkpoint_reload_max_margin_error' in json.loads((folder/'metrics.json').read_text()):continue
   with (folder/'training.log').open('w',encoding='utf-8') as log:run=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','worker','--datasets',j['dataset'],'--fold',j['fold'],'--device',a.device],stdout=log,stderr=subprocess.STDOUT)
   if run.returncode:raise RuntimeError('No-edge training failed: '+str(folder/'training.log'))
   print('COMPLETE_NO_EDGES',key(j),flush=True)
 if a.stage in ['all','transfer']:
  with gpu_slot():transfer(a.device)

if __name__=='__main__':main()

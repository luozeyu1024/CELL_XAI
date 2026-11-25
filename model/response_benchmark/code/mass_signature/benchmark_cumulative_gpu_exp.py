"""Measure actual GATv2+GINE training-step cost on source-training cell batches."""
import argparse,gc,time
import numpy as np,pandas as pd,torch
from torch_geometric.data import Batch
from cumulative_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');p.add_argument('--batch_size',type=int,default=32);args=p.parse_args();torch.set_num_threads(4)
 path=FIG/'source_data/gpu_cost.tsv';records=pd.read_csv(path,sep='\t').to_dict('records') if path.exists() else []
 done={(r['dataset'],r['fold'],r['relation'],r['threshold'],r['batch_profile']) for r in records}
 device='cuda';assert torch.cuda.is_available();hardware=torch.cuda.get_device_properties(0)
 for job in jobs():
  ds=job['dataset'];fold=job['fold']
  if args.datasets and ds not in args.datasets:continue
  if not (DATA/'candidates'/ds/'expression_validation.json').exists():raise ValueError('Expression not validated: '+ds)
  cells=read_cells(job);train=cells[cells.split.eq('train')].cell_id
  profile=pd.read_csv(FIG/'source_data/profiles'/f'{ds}.tsv.gz',sep='\t');profile=profile[profile.cell_id.isin(train)]
  for q in Q:
   for relation in RELATIONS:
    pf=profile[profile.threshold.eq(q)&profile.relation.eq(relation)].copy()
    actual=CumulativeGraphs(ds,relation,q,cells);mapping=pd.Series(np.arange(len(cells)),index=cells.cell_id)
    typical=pf.sort_values('n_nodes').iloc[np.linspace(0,len(pf)-1,args.batch_size).astype(int)]
    tail=pf.sort_values('n_edges',ascending=False).head(args.batch_size)
    for name,sub in [('typical',typical),('edge_tail',tail)]:
     if (ds,fold,relation,q,name) in done:continue
     cpu=Batch.from_data_list([actual[int(mapping.loc[cid])] for cid in sub.cell_id]);gc.collect();torch.cuda.empty_cache()
     model=GraphClassifier(len(actual.genes)).to(device);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
     row=dict(dataset=ds,fold=fold,relation=relation,threshold=q,batch_profile=name,batch_size=len(sub),n_nodes=cpu.num_nodes,n_directed_edges=cpu.num_edges,device=hardware.name,total_device_GiB=hardware.total_memory/2**30,includes_optimizer_state=True,precision='float32',status='ok')
     try:
      batch=cpu.to(device);target=torch.arange(len(sub),device=device)%2
      def step():
       opt.zero_grad(set_to_none=True);logits=model(batch.x,batch.edge_index,batch.edge_attr,batch.batch,batch.gene_id);loss=torch.nn.functional.cross_entropy(logits,target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
      step();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();elapsed=[]
      for _ in range(2):
       start=time.perf_counter();step();torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
      row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,step_seconds=float(np.median(elapsed)))
     except torch.cuda.OutOfMemoryError:
      row.update(status='out_of_memory',peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,step_seconds=np.nan)
     records.append(row);pd.DataFrame(records).to_csv(path,sep='\t',index=False);print('GPU_COST',ds,fold,q,relation,name,row['status'],round(row['peak_allocated_GiB'],2),flush=True)
     del opt,model,cpu
     if 'batch' in locals():del batch
     gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 with gpu_slot():main()


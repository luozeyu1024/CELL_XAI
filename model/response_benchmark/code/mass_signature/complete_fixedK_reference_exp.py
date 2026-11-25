"""Complete the three missing GSE140440 fixed-K graph references without changing splits."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from project_paths import project_root

ROOT=project_root();BENCH=ROOT/'model/response_benchmark'
sys.path.insert(0,str(BENCH/'code'))
import train_response_exp as core
from cumulative_common import gpu_slot

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--relation',nargs='+',choices=['ppi','qwen','esm2_650m'],default=['ppi','qwen','esm2_650m'])
 p.add_argument('--device',default='cuda');a=p.parse_args()
 base=json.loads((BENCH/'output/within/GSE140440/gat_gine/run_parameters.json').read_text(encoding='utf-8'))
 cells=pd.read_csv(BENCH/'data/cohorts/GSE140440/cells.tsv',sep='\t',keep_default_na=False)
 info=json.loads((BENCH/'data/cohorts/GSE140440/data_sources.json').read_text(encoding='utf-8'))
 assert info['top_k']==1536 and base['gene_dim']==16
 torch.set_num_threads(base['threads'])
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 with gpu_slot():
  for rel in a.relation:
   args=SimpleNamespace(**dict(base,open_path=str(BENCH),stage='train',datasets=['GSE140440'],models=['gat_gine'],relation=rel,device=a.device))
   core.train_one(args,BENCH,'GSE140440','gat_gine')
   folder=core.output_root(args,BENCH)/'within/GSE140440/gat_gine'
   model,ck=core.restore(folder/'checkpoint.pt',a.device)
   assert ck['training_cell_ids']==cells.loc[cells.split.eq('train'),'cell_id'].tolist()
   assert ck['validation_cell_ids']==cells.loc[cells.split.eq('validation'),'cell_id'].tolist()
   assert ck['relation']==rel and ck['model_config']['gene_dim']==16
   data,actual,genes=core.load_cohort(BENCH,'GSE140440','gat_gine',rel)
   fresh=core.predict(model,data,actual,'gat_gine',a.device,args.graph_batch_size)
   saved=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
   assert fresh.cell_id.tolist()==saved.cell_id.tolist()==cells.cell_id.tolist()
   assert saved.y.tolist()==cells.y.tolist() and saved.split.tolist()==cells.split.tolist()
   error=float(np.max(np.abs(fresh.margin_resistant_minus_sensitive.to_numpy()-saved.margin_resistant_minus_sensitive.to_numpy())))
   assert error<=1e-4
   m=json.loads((folder/'metrics.json').read_text())
   for split in ['train','validation','test']:
    scores=core.metrics(saved[saved.split.eq(split)])
    for name in ['auroc','balanced_accuracy','auprc_resistant','mcc','brier_score']:assert np.isclose(scores[name],m[split][name],atol=1e-10)
   m.update(checkpoint_reload_max_margin_error=error,graph_relation=rel,gene_feature_dimension=16,fixed_k=1536)
   core.write_json(folder/'metrics.json',m)
   print('VERIFIED_FIXED_K_REFERENCE',rel,m['test']['auroc'],flush=True)
   del model,data
   if torch.cuda.is_available():torch.cuda.empty_cache()

if __name__=='__main__':main()

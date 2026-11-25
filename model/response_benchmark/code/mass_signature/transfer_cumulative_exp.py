"""Frozen FA34 transfer on cumulative signatures using the source mass policy."""
import gc,json
import numpy as np,pandas as pd,torch
from cumulative_common import *
import train_response_exp as core
from train_cumulative_benchmark_exp import trial_folder,install_loader

from split_registry import apply_splits, prostate_cell_lines

def fa34(variants,device,targets):
 job=next(j for j in jobs() if j['dataset']=='GSE223779');q=policy(job)['threshold'];install_loader(job,'state_15b',q)
 for relation in variants:
  source_record=json.loads((trial_folder(job,relation)/'metrics.json').read_text(encoding='utf-8'))
  assert 'checkpoint_reload_max_margin_error' in source_record and source_record['mass_threshold']==q,'Source checkpoint must be verified under the frozen source policy'
  model,saved=core.restore(trial_folder(job,relation)/'checkpoint.pt',device)
  for target in targets:
   dest=OUT/relation/'transfer'/f'GSE223779_to_{target}'/'gat_gine';dest.mkdir(parents=True,exist_ok=True)
   if (dest/'metrics.json').exists():continue
   if not (DATA/'candidates'/target/'expression_validation.json').exists():raise FileNotFoundError('Target complete expression not ready: '+target)
   data,cells,genes=core.load_cohort(BENCH,target,'gat_gine',relation);assert genes==saved['gene_symbols'];assert not set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids'])
   pred=core.predict(model,data,cells,'gat_gine',device,32)
   for col in cells:
    if col not in pred:pred[col]=cells[col].to_numpy()
   if target=='GSE140440':
    pred['cell_line']=pred.cell_id.map(prostate_cell_lines());assert pred.cell_line.notna().all()
   parts={'all_labeled':pred[pred.y.ge(0)]}
   if target=='GSE140440':parts.update({'cell_line_'+str(k):f for k,f in pred.groupby('cell_line')})
   elif target=='GSE163836':parts.update(primary_FCIBC02_pair=apply_splits(pred,'GSE163836',mode='infer',scope='FCIBC02'),additional_SUM149PR=apply_splits(pred,'GSE163836',mode='infer',scope='SUM149'))
   elif target=='GSE223779_TS485T':parts.update(same_drug_crizotinib=apply_splits(pred,'GSE223779_TS485T',mode='infer',scope='crizotinib'),cross_drug_alectinib=apply_splits(pred,'GSE223779_TS485T',mode='infer',scope='alectinib'))
   elif target=='GSE202234':parts.update({'treatment_'+str(k):f[f.y.ge(0)] for k,f in pred.groupby('treatment')})
   scores={}
   for name,f in parts.items():
    scores[name]=core.metrics(f)
    if f.loc[f.y.ge(0),'y'].nunique()<2:
     for k in ['balanced_accuracy','mcc','macro_f1']:scores[name][k]=None
   seen=np.zeros(len(genes),bool);seen[saved['training_observed_gene_ids']]=True;used=[];unseen=total=0
   for row in data.index:
    lo=data.ptr[row];n=data.counts[row,data.j];ids=data.gene_id[lo:lo+n];used.append(np.unique(ids));unseen+=int((~seen[ids]).sum());total+=len(ids)
   pred.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g');core.write_json(dest/'metrics.json',dict(source='GSE223779',target=target,relation=relation,source_mass_threshold=q,target_mass_threshold=q,target_training=False,target_threshold_tuning=False,checkpoint=str(trial_folder(job,relation)/'checkpoint.pt'),n_target_unique_genes_unseen_in_source=int((~seen[np.unique(np.concatenate(used))]).sum()),fraction_target_node_occurrences_unseen_in_source_training=unseen/total,scopes=scores))
   print('FROZEN_TRANSFER',relation,target,flush=True);del data;gc.collect();torch.cuda.empty_cache()
  del model;gc.collect();torch.cuda.empty_cache()

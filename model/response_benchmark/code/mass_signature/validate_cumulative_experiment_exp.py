"""Audit completed source checkpoints, scalers, biological splits and frozen transfers."""
import argparse,json
import numpy as np,pandas as pd,torch
from cumulative_common import *
from split_registry import prostate_cell_lines
from train_cumulative_benchmark_exp import trial_folder
import train_response_exp as core

def check_metrics(frame,saved):
 values=core.metrics(frame)
 if frame.loc[frame.y.ge(0),'y'].nunique()<2:
  for key in ['balanced_accuracy','mcc','macro_f1']:values[key]=None
 for key in ['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','brier_score']:
  x,y=values.get(key),saved.get(key)
  if x is None or y is None:assert x is None and y is None,(key,x,y)
  else:assert np.isclose(x,y,rtol=1e-10,atol=1e-10),(key,x,y)
 assert values['n_cells']==saved.get('n_cells',saved.get('n_evaluation_units'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--allow_partial',action='store_true');a=p.parse_args();rows=[];calibration={};transfers=[];torch.set_num_threads(4)
 for job in jobs():
  cells=read_cells(job);train=np.flatnonzero(cells.split.eq('train'));valid=cells.split.eq('validation')
  for rel in RELATIONS:
   folder=trial_folder(job,rel)
   if not (folder/'metrics.json').exists() or 'checkpoint_reload_max_margin_error' not in json.loads((folder/'metrics.json').read_text()):
    if a.allow_partial:continue
    raise ValueError('Unverified trial: '+str(folder))
   c=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);m=json.loads((folder/'metrics.json').read_text());q=policy(job)['threshold']
   assert c['relation']==rel and c['model_config']['gene_dim']==16 and c['model_config']['hidden']==64 and c['model_config']['layers']==2 and c['model_config']['heads']==4
   assert c['parameters']['mass_threshold']==q and m['mass_threshold']==q
   assert c['training_cell_ids']==cells.cell_id.iloc[train].tolist() and c['validation_cell_ids']==cells.loc[valid,'cell_id'].tolist()
   if key(job) not in calibration:
    data=CumulativeGraphs(job['dataset'],'state_15b',q,cells);mean,std,observed=stats(data,train,len(data.genes));calibration[key(job)]=(mean,std,observed,data.genes)
   mean,std,observed,genes=calibration[key(job)]
   assert np.isclose(mean,c['model_config']['expr_mean'],atol=1e-12) and np.isclose(std,c['model_config']['expr_std'],atol=1e-12)
   assert np.array_equal(np.flatnonzero(observed),c['training_observed_gene_ids']) and c['gene_symbols']==genes
   weights=len(train)/(2*np.bincount(cells.y.iloc[train],minlength=2));assert np.allclose(weights,c['class_weights'])
   assert 'identity.weight' in c['state_dict'] and c['state_dict']['identity.weight'].shape==(len(genes),16)
   pred=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip');assert pred.cell_id.tolist()==cells.cell_id.tolist();assert pred.y.tolist()==cells.y.tolist()
   info=json.loads((Path(job['cohort_root'])/'data/cohorts'/job['dataset']/'data_sources.json').read_text())
   group=info.get('group_key')
   if group:assert not set(cells.loc[cells.split.eq('test'),group])&set(cells.loc[~cells.split.eq('test'),group])
   for split in ['train','validation','test']:
    values=core.metrics(pred[pred.split.eq(split)])
    for name in ['auroc','balanced_accuracy','auprc_resistant','mcc','brier_score']:assert np.isclose(values[name],m[split][name],rtol=1e-10,atol=1e-10)
   rows.append(dict(dataset=job['dataset'],fold=job['fold'],relation=rel,mass_threshold=q,source_scaler_error=abs(mean-c['model_config']['expr_mean']),checkpoint_reload_max_error=m['checkpoint_reload_max_margin_error'],status='pass'))
 for rel in RELATIONS:
  for target in ['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']:
   path=OUT/rel/'transfer'/f'GSE223779_to_{target}'/'gat_gine/metrics.json'
   if not path.exists():
    if a.allow_partial:continue
    raise ValueError('Missing frozen transfer: '+str(path))
   m=json.loads(path.read_text());assert not m['target_training'] and not m['target_threshold_tuning'];assert m['source_mass_threshold']==m['target_mass_threshold']==policy(next(j for j in jobs() if j['dataset']=='GSE223779'))['threshold']
   source_job=next(j for j in jobs() if j['dataset']=='GSE223779');source_checkpoint=trial_folder(source_job,rel)/'checkpoint.pt';assert Path(m['checkpoint'])==source_checkpoint
   pred=pd.read_csv(path.with_name('predictions.tsv'),sep='\t',float_precision='round_trip');cells=pd.read_csv(BENCH/'data/cohorts'/target/'cells.tsv',sep='\t',keep_default_na=False);assert pred.cell_id.tolist()==cells.cell_id.tolist() and pred.y.tolist()==cells.y.tolist() and pred.split.tolist()==cells.split.tolist()
   parts={'all_labeled':pred[pred.y.ge(0)]}
   if target=='GSE140440':
    assert pred.cell_line.tolist()==pred.cell_id.map(prostate_cell_lines()).tolist();parts.update({'cell_line_'+str(k):f for k,f in pred.groupby('cell_line')})
   elif target=='GSE163836':parts.update(primary_FCIBC02_pair=pred[pred.cell_line.eq('FCIBC02')],additional_SUM149PR=pred[pred.original_group.eq('SUM149PR')])
   elif target=='GSE223779_TS485T':parts.update(same_drug_crizotinib=pred[pred.drug.isin(['untreated','crizotinib'])],cross_drug_alectinib=pred[pred.drug.isin(['untreated','alectinib'])])
   elif target=='GSE202234':parts.update({'treatment_'+str(k):f[f.y.ge(0)] for k,f in pred.groupby('treatment')})
   assert set(parts)==set(m['scopes'])
   for scope,part in parts.items():check_metrics(part,m['scopes'][scope])
   model,checkpoint=core.restore(source_checkpoint,'cpu');data=CumulativeGraphs(target,rel,m['target_mass_threshold'],cells);valid=np.flatnonzero(cells.y.ge(0));sample=sorted(set([int(valid[0]),int(valid[-1])]+np.flatnonzero(cells.y.lt(0))[:1].tolist()))
   from torch.utils.data import Subset
   fresh=core.predict(model,Subset(data,sample),cells.iloc[sample].reset_index(drop=True),'gat_gine','cpu',32);stored=pred.set_index('cell_id').loc[fresh.cell_id];difference=np.abs(fresh.margin_resistant_minus_sensitive.to_numpy()-stored.margin_resistant_minus_sensitive.to_numpy());assert np.allclose(fresh.margin_resistant_minus_sensitive,stored.margin_resistant_minus_sensitive,atol=2e-4,rtol=2e-5)
   transfers.append(dict(relation=rel,dataset=target,n_cells=len(cells),n_scopes_recomputed=len(parts),n_cpu_reload_predictions=len(sample),max_cpu_gpu_margin_difference=float(difference.max()),status='pass'))
  clinical=OUT/rel/'transfer/GSE165897_pre_EOC_to_GSE154600_EPI_external/gat_gine'
  if (clinical/'metrics.json').exists():
   ms=json.loads((clinical/'metrics.json').read_text());expected=[policy(j)['threshold'] for j in jobs() if j['dataset']=='GSE165897_pre_EOC'];assert ms['source_mass_thresholds']==expected
   fs=[pd.read_csv(clinical/f'fold_{i}_cell_predictions.tsv',sep='\t').probability_resistant.to_numpy() for i in range(1,5)];ensemble=pd.read_csv(clinical/'ensemble_cell_predictions.tsv',sep='\t');assert np.allclose(np.mean(fs,axis=0),ensemble.probability_resistant)
   patient=pd.read_csv(clinical/'ensemble_patient_predictions.tsv',sep='\t').set_index('patient_id');assert np.allclose(ensemble.groupby('patient_id').probability_resistant.mean().loc[patient.index],patient.probability_resistant)
   check_metrics(ensemble.assign(margin_resistant_minus_sensitive=ensemble.probability_resistant),ms['scopes']['external_epithelial_cells'])
   check_metrics(patient.assign(margin_resistant_minus_sensitive=patient.probability_resistant),ms['scopes']['external_patients'])
   transfers.append(dict(relation=rel,dataset='GSE154600_EPI_external',n_cells=len(ensemble),n_scopes_recomputed=2,n_cpu_reload_predictions=0,max_cpu_gpu_margin_difference=np.nan,status='pass'))
  elif not a.allow_partial:raise ValueError('Missing clinical transfer '+rel)
 pd.DataFrame(rows).to_csv(FIG/'source_data/completed_experiment_validation.tsv',sep='\t',index=False);pd.DataFrame(transfers).to_csv(FIG/'source_data/frozen_transfer_validation.tsv',sep='\t',index=False);print('PASS',len(rows),'source checkpoints and',len(transfers),'frozen target configurations; source-only scalers, original splits, source policies and independently recomputed target metrics. FA34 target predictions reloaded on CPU.',flush=True)
if __name__=='__main__':main()

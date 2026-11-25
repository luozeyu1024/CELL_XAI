"""Summarize cumulative-signature classifiers at unchanged biological evaluation units."""
import argparse,json
import numpy as np,pandas as pd
from cumulative_common import *
import train_response_exp as core
import report_gene_embedding_exp as oldreport
from train_cumulative_benchmark_exp import destination,trial_folder
METRICS=['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','macro_f1','brier_score']
def main():
 p=argparse.ArgumentParser();p.add_argument('--allow_partial',action='store_true');a=p.parse_args();status=[];splits=[];resources=[]
 for rel in RELATIONS:
  for job in jobs():
   path=trial_folder(job,rel)/'metrics.json';ready=path.exists() and 'checkpoint_reload_max_margin_error' in json.loads(path.read_text())
   status.append(dict(dataset=job['dataset'],fold=job['fold'],relation=rel,complete=ready,status='completed' if ready else ('partial_not_verified' if path.with_name('checkpoint.pt').exists() else 'not_completed')))
   if not ready:continue
   m=json.loads(path.read_text())
   if 'actual_training_peak_allocated_GiB' in m:
    m['training_and_inference_peak_allocated_GiB']=m.pop('actual_training_peak_allocated_GiB')
    core.write_json(path,m)
   pred=pd.read_csv(path.with_name('predictions.tsv'),sep='\t',float_precision='round_trip');cells=read_cells(job);assert pred.cell_id.tolist()==cells.cell_id.tolist() and pred.split.tolist()==cells.split.tolist()
   history=pd.read_csv(path.with_name('training_history.tsv'),sep='\t');best=np.inf;chosen=None
   for row in history.itertuples():
    if row.validation_loss<best-.0001:best=row.validation_loss;chosen=row.epoch
   assert chosen==m['best_epoch']
   resources.append(dict(dataset=job['dataset'],fold=job['fold'],relation=rel,mass_threshold=m['mass_threshold'],n_training_cells=int(cells.split.eq('train').sum()),n_validation_cells=int(cells.split.eq('validation').sum()),n_test_cells=int(cells.split.eq('test').sum()),epochs_run=m['epochs_run'],best_epoch=m['best_epoch'],trainable_parameters=m['trainable_parameter_count'],fit_and_initial_prediction_seconds=m['training_seconds'],training_and_inference_peak_allocated_GiB=m['training_and_inference_peak_allocated_GiB']))
   for split in ['train','validation','test']:
    metric=core.metrics(pred[pred.split.eq(split)])
    for name in METRICS:assert np.isclose(metric[name],m[split][name],rtol=1e-10,atol=1e-10)
    part=pred[pred.split.eq(split)]
    splits.append(dict(dataset=job['dataset'],fold=job['fold'],relation=rel,split=split,mass_threshold=m['mass_threshold'],n_unique_logit_margins=int(part.margin_resistant_minus_sensitive.nunique()),logit_margin_std=float(part.margin_resistant_minus_sensitive.std(ddof=0)),fraction_predicted_positive=float(part.probability_resistant.ge(.5).mean()),**{k:v for k,v in metric.items() if not isinstance(v,(dict,list))}))
 pd.DataFrame(status).to_csv(FIG/'source_data/training_completion.tsv',sep='\t',index=False);pd.DataFrame(splits).to_csv(FIG/'source_data/training_split_metrics.tsv',sep='\t',index=False)
 pd.DataFrame(resources).to_csv(FIG/'source_data/training_resources.tsv',sep='\t',index=False)
 oldreport.OUT=OUT;oldreport.destination=destination
 rows=[]
 for rel in RELATIONS:
  rows.extend(oldreport.grouped_rows(rel,allow_partial=True));rows.extend(oldreport.transfer_rows(rel))
 for rel in RELATIONS:
  for path in (OUT/rel/'grouped').glob('*/summary/oof_patient_predictions.tsv'):
   patient=pd.read_csv(path,sep='\t',float_precision='round_trip')
   if 'margin_resistant_minus_sensitive' in patient:patient.drop(columns='margin_resistant_minus_sensitive').to_csv(path,sep='\t',index=False,float_format='%.17g')
 result=pd.DataFrame(rows);result.to_csv(FIG/'source_data/primary_metrics.tsv',sep='\t',index=False)
 baseline=pd.read_csv(ROOT/'work/transfer_analysis/data/reassessed/primary_comparison.tsv',sep='\t');comparison=[]
 for row in rows:
  match=baseline[baseline.dataset.eq(row['dataset'])&baseline.scope.eq(row['scope'])&baseline.model.eq(row['model'])];assert len(match)==1
  for field in ['n_cells','n_evaluation_units']:
   if field in row and field in match.columns and pd.notna(match.iloc[0][field]):assert int(row[field])==int(match.iloc[0][field]),'Evaluation population differs from fixed-K reference'
  if row.get('unit')=='patient' and 'n_patients' in match.columns and pd.notna(match.iloc[0].n_patients):assert int(row['n_evaluation_units'])==int(match.iloc[0].n_patients)
  for field in ['endpoint','positive_class','evaluation_unit','aggregation','cell_line']:
   if field in match.columns and pd.notna(match.iloc[0][field]):row[field]=match.iloc[0][field]
  for metric in METRICS:comparison.append(dict(dataset=row['dataset'],scope=row['scope'],relation=row['model'],metric=metric,fixed_k=match.iloc[0][metric],cumulative=row[metric],difference=row[metric]-match.iloc[0][metric]))
 pd.DataFrame(rows).to_csv(FIG/'source_data/primary_metrics.tsv',sep='\t',index=False)
 pd.DataFrame(comparison).to_csv(FIG/'source_data/matched_fixedK_differences.tsv',sep='\t',index=False)
 complete=sum(x['complete'] for x in status);print('VERIFIED',complete,'/56 trainings;',len(rows),'/44 formal evaluation scopes',flush=True)
 if not a.allow_partial:assert complete==56 and len(rows)==44
if __name__=='__main__':main()


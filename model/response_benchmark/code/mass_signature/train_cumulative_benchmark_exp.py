"""Train the unchanged 16D gene-identity GATv2+GINE under reviewed mass policies."""
import argparse,gc,json,subprocess,sys
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from cumulative_common import *
import train_response_exp as core
import gene_embedding_benchmark_exp as previous
from split_registry import apply_splits, load_cells, cohort_metadata_root


def destination(job,relation):
 return OUT/relation if job['family']=='within' else OUT/relation/'grouped'/job['dataset']/job['fold']
def trial_folder(job,relation):return destination(job,relation)/'within'/job['dataset']/'gat_gine'
def install_loader(job,relation,threshold=None):
 q=policy(job)['threshold'] if threshold is None else threshold
 def load(root,ds,kind,rel='state_15b'):
  cells=load_cells(Path(root)/'data/cohorts'/ds/'cells.tsv',ds)
  data=CumulativeGraphs(ds,rel,q,cells);return data,cells,data.genes
 core.load_cohort=load;core.batches=loader;core.training_statistics=stats
 core.output_root=lambda args,root:destination(job,relation)
def train(job,relation,device):
 read_cells(job)
 folder=trial_folder(job,relation);complete=folder/'metrics.json'
 if complete.exists() and 'checkpoint_reload_max_margin_error' in json.loads(complete.read_text()):
  existing=json.loads(complete.read_text())
  assert existing['mass_threshold']==policy(job)['threshold'] and existing['graph_relation']==relation and existing['gene_feature_dimension']==16,'Completed trial differs from the selected policy or feature design'
  print('VERIFIED_EXISTS',key(job),relation,flush=True);return
 torch.set_num_threads(4);install_loader(job,relation)
 args=SimpleNamespace(seed=42,hidden=64,layers=2,heads=4,gene_dim=16,dropout=.2,lr=.001,weight_decay=.0001,graph_batch_size=32,mlp_batch_size=256,epochs=100,patience=15,min_delta=.0001,device=device,threads=4,relation=relation,signature_method='cumulative_full_model_IG',mass_threshold=policy(job)['threshold'],policy_source=str(FIG/'source_data/selected_policies.json'),signature_candidate_path=str(DATA/'candidates'/job['dataset']),global_graph_path=str(DATA/'global_graphs'/f'{relation}.npz'),gene_vocab_path=str(DATA/'gene_vocab.tsv'))
 torch.cuda.reset_peak_memory_stats();core.train_one(args,Path(job['cohort_root']),job['dataset'],'gat_gine')
 cells=read_cells(job);prediction=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip');assert prediction.cell_id.tolist()==cells.cell_id.tolist()
 for col in cells:
  if col not in prediction:prediction[col]=cells[col].to_numpy()
 prediction.to_csv(folder/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
 model,checkpoint=core.restore(folder/'checkpoint.pt',device);assert model.identity.weight.requires_grad and model.config['gene_dim']==16
 assert checkpoint['training_cell_ids']==cells.loc[cells.split.eq('train'),'cell_id'].tolist();assert checkpoint['validation_cell_ids']==cells.loc[cells.split.eq('validation'),'cell_id'].tolist()
 data,actual,genes=core.load_cohort(Path(job['cohort_root']),job['dataset'],'gat_gine',relation);ix=np.flatnonzero(cells.split.eq('test'));testcells=cells.iloc[ix].reset_index(drop=True)
 from torch.utils.data import Subset
 fresh=core.predict(model,Subset(data,ix.tolist()),testcells,'gat_gine',device,32);old=prediction.set_index('cell_id').loc[fresh.cell_id]
 error=float(np.max(np.abs(fresh.margin_resistant_minus_sensitive.to_numpy()-old.margin_resistant_minus_sensitive.to_numpy())));assert error<=1e-4
 for split in ['train','validation','test']:
  metrics=core.metrics(prediction[prediction.split.eq(split)]);saved=json.loads(complete.read_text())[split]
  for metric in ['auroc','balanced_accuracy','auprc_resistant','mcc','brier_score']:assert np.isclose(metrics[metric],saved[metric],atol=1e-10)
 result=json.loads(complete.read_text());result.update(signature_method='cumulative_full_model_IG',mass_threshold=args.mass_threshold,graph_relation=relation,gene_feature_dimension=16,trainable_parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_reload_max_margin_error=error,training_and_inference_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,scope='original biological-group or exploratory cell split; unchanged optimizer and learned 16D identity')
 core.write_json(complete,result);print('VERIFIED',key(job),relation,error,flush=True)
 del model,data;gc.collect();torch.cuda.empty_cache()

def transfer(variants,device,targets=None,include_clinical=True):
 sourcejob=next(j for j in jobs() if j['dataset']=='GSE223779');q=policy(sourcejob)['threshold'];install_loader(sourcejob,'state_15b',q)
 previous.OUT=OUT;previous.VARIANTS={r:(r,None) for r in RELATIONS};previous.restore=core.restore
 from transfer_cumulative_exp import fa34
 fa34(variants,device,targets or ['GSE140440','GSE163836','GSE202234','GSE223779_TS485T'])
 if not include_clinical:return
 for relation in variants:
  ds='GSE154600_EPI_external';task='GSE165897_pre_EOC';dest=OUT/relation/'transfer'/f'{task}_to_{ds}'/'gat_gine';dest.mkdir(parents=True,exist_ok=True)
  if (dest/'metrics.json').exists():continue
  cells=pd.read_csv(DATA/'candidates'/ds/'cells.tsv',sep='\t',keep_default_na=False);meta=pd.read_csv(cohort_metadata_root()/ds/'counts_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
  assert not meta.index.has_duplicates and cells.cell_id.isin(meta.index).all(),'Clinical target metadata must match every cell explicitly'
  for col in meta:cells[col]=cells.cell_id.map(meta[col])
  assert cells.label.isin(['sensitive','resistant','unknown']).all() and cells.patient_id.notna().all()
  cells['response']=cells.label;cells['y']=cells.label.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int);cells['split']='external';cells=apply_splits(cells,ds,mode='infer');probs=[];thresholds=[]
  for job in [j for j in jobs() if j['dataset']==task]:
   source_record=json.loads((trial_folder(job,relation)/'metrics.json').read_text(encoding='utf-8'));assert 'checkpoint_reload_max_margin_error' in source_record and source_record['mass_threshold']==policy(job)['threshold']
   threshold=policy(job)['threshold'];thresholds.append(threshold);data=CumulativeGraphs(ds,relation,threshold,cells);model,ck=core.restore(trial_folder(job,relation)/'checkpoint.pt',device)
   assert data.genes==ck['gene_symbols'];assert not set(cells.cell_id)&set(ck['training_cell_ids']+ck['validation_cell_ids'])
   frame=previous.attach_annotations(core.predict(model,data,cells,'gat_gine',device,32),cells);frame.to_csv(dest/f'{job["fold"]}_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g');probs.append(frame.probability_resistant.to_numpy());del model,data;gc.collect();torch.cuda.empty_cache()
  result=cells.copy();result['probability_resistant']=np.mean(probs,axis=0);result.to_csv(dest/'ensemble_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
  patients=result.groupby('patient_id').agg(y=('y','first'),probability_resistant=('probability_resistant','mean'),n_cells=('cell_id','size')).reset_index();assert not result.groupby('patient_id').y.nunique().gt(1).any();patients.to_csv(dest/'ensemble_patient_predictions.tsv',sep='\t',index=False,float_format='%.17g')
  scores={}
  for name,frame in [('external_epithelial_cells',result),('external_patients',patients)]:
   m=core.metrics(frame.assign(margin_resistant_minus_sensitive=frame.probability_resistant));m['ranking_score']='mean_frozen_source_fold_probability_resistant'
   if name=='external_patients':m['n_evaluation_units']=m.pop('n_cells');m['n_sensitive_patients']=m.pop('n_sensitive');m['n_resistant_patients']=m.pop('n_resistant');m['unit']='patient'
   scores[name]=m
  core.write_json(dest/'metrics.json',dict(source=task,target=ds,relation=relation,target_training=False,target_threshold_tuning=False,source_mass_thresholds=thresholds,scopes=scores));print('CLINICAL_TRANSFER',relation,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','worker','transfer','all'],default='all');p.add_argument('--relation',nargs='+',choices=RELATIONS,default=RELATIONS);p.add_argument('--datasets',nargs='+');p.add_argument('--fold');p.add_argument('--device',default='cuda');p.add_argument('--prediction_scope',choices=['all','fa34'],default='all');p.add_argument('--targets',nargs='+',choices=['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']);a=p.parse_args();torch.set_num_threads(4)
 tasks=[j for j in jobs() if (not a.datasets or j['dataset'] in a.datasets) and (not a.fold or j['fold']==a.fold)]
 if a.stage in ['worker']:
  assert len(tasks)==len(a.relation)==1
  with gpu_slot():train(tasks[0],a.relation[0],a.device)
  return
 if a.stage in ['train','all']:
  for job in tasks:
   for relation in a.relation:
    folder=trial_folder(job,relation);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists() and 'checkpoint_reload_max_margin_error' in json.loads((folder/'metrics.json').read_text()):continue
    with (folder/'training.log').open('w',encoding='utf-8') as log:
     run=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--stage','worker','--datasets',job['dataset'],'--fold',job['fold'],'--relation',relation,'--device',a.device],stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:raise RuntimeError('Training failed: '+str(folder/'training.log'))
    print('COMPLETE',key(job),relation,flush=True)
 if a.stage in ['transfer','all']:
  with gpu_slot():transfer(a.relation,a.device,a.targets,a.prediction_scope=='all')
if __name__=='__main__':main()




"""Verify and compare shared no-edge controls for fixed-K and cumulative signatures."""
import argparse,json
import numpy as np,pandas as pd,torch
from cumulative_common import *
import train_response_exp as core
import report_gene_embedding_exp as aggregation
import train_cumulative_benchmark_exp as full
from cumulative_edge_ablation_exp import NoEdgeGraphs,old_folder,CONTROL,TABLES
METRICS=['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','macro_f1','brier_score']

def fixed_reference():
 old=pd.read_csv(ROOT/'work/transfer_analysis/data/reassessed/primary_comparison.tsv',sep='\t');repairs=[]
 # The original PC9 macro writer omitted sensitive AP; recover it from saved folds.
 for i,row in old[old[METRICS].isna().any(axis=1)].iterrows():
  assert row.dataset=='GSE149383_PC9_DTEP_sample_grouped' and row.scope=='sample_set_macro'
  values=[];paths=[]
  for job in [j for j in jobs() if j['dataset']==row.dataset]:
   kind='state_mlp' if row.model=='native_state_embedding' else 'gat_gine';path=BENCH/'output/grouped_adaptive'/row.dataset/job['fold']/row.model/'within'/row.dataset/kind/'predictions.tsv'
   frame=pd.read_csv(path,sep='\t',float_precision='round_trip');cells=read_cells(job)
   for col in ['cell_id','y','split']:assert frame[col].tolist()==cells[col].tolist()
   values.append(core.metrics(frame[frame.split.eq('test')]));paths.append(str(path))
  assert len(values)==2
  for metric in [*METRICS,'threshold']:
   value=float(np.mean([v[metric] for v in values]))
   if pd.isna(row[metric]):
    old.loc[i,metric]=value;repairs.append(dict(dataset=row.dataset,model=row.model,scope=row.scope,metric=metric,value=value,aggregation='equal mean of two held-out sample-set metrics',prediction_paths='; '.join(paths)))
   else:assert np.isclose(value,row[metric],atol=1e-10)
 old.to_csv(TABLES/'fixedK_reference_metrics.tsv',sep='\t',index=False);pd.DataFrame(repairs).to_csv(TABLES/'recomputed_reference_fields.tsv',sep='\t',index=False)
 return old

def scalar_check(frame,saved,single_class=False):
 values=core.metrics(frame)
 if single_class and frame.loc[frame.y.ge(0),'y'].nunique()<2:
  for k in ['balanced_accuracy','mcc','macro_f1']:values[k]=None
 for k in METRICS:
  x,y=values.get(k),saved.get(k)
  assert (x is None and y is None) or (x is not None and y is not None and np.isclose(x,y,rtol=1e-10,atol=1e-10)),(k,x,y)
 assert values['n_cells']==saved.get('n_cells',saved.get('n_evaluation_units'))

def verify_sources():
 rows=[];split_rows=[]
 for job in jobs():
  cells=read_cells(job);q=policy(job)['threshold'];folder=full.trial_folder(job,CONTROL);m=json.loads((folder/'metrics.json').read_text());assert m['edge_variant']==CONTROL and m['mass_threshold']==q and m['checkpoint_reload_max_margin_error']<=1e-4
  scope='original exploratory random-cell split' if job['family']=='within' else 'original biological-group-held-out split; test cells, with formal group aggregation reported separately'
  if m.get('scope')!=scope:m['scope']=scope;core.write_json(folder/'metrics.json',m)
  ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);ref=torch.load(full.trial_folder(job,'state_15b')/'checkpoint.pt',map_location='cpu',weights_only=False)
  assert ck['model_config']==ref['model_config'] and ck['parameters']['ablation']==CONTROL
  assert ck['training_cell_ids']==ref['training_cell_ids']==cells.loc[cells.split.eq('train'),'cell_id'].tolist()
  assert ck['validation_cell_ids']==ref['validation_cell_ids']==cells.loc[cells.split.eq('validation'),'cell_id'].tolist()
  data=NoEdgeGraphs(job['dataset'],CONTROL,q,cells);train=np.flatnonzero(cells.split.eq('train'));mean,std,seen=stats(data,train,len(data.genes))
  assert np.isclose(mean,ck['model_config']['expr_mean'],atol=1e-12) and np.isclose(std,ck['model_config']['expr_std'],atol=1e-12)
  assert ck['training_observed_gene_ids']==np.flatnonzero(seen).tolist()==ref['training_observed_gene_ids']
  assert np.allclose(ck['class_weights'],len(train)/(2*np.bincount(cells.y.iloc[train],minlength=2)))
  new=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip');old=old_folder(job);previous=pd.read_csv(old/'predictions.tsv',sep='\t',float_precision='round_trip');prior=json.loads((old/'metrics.json').read_text())
  oldck=torch.load(old/'checkpoint.pt',map_location='cpu',weights_only=False);assert oldck['model_config']['gene_dim']==16 and oldck['gene_symbols']==ck['gene_symbols']
  oldfull=(BENCH/'output/within'/job['dataset']/'gat_gine/checkpoint.pt') if job['family']=='within' else (BENCH/'output/grouped_adaptive'/job['dataset']/job['fold']/'state_15b/within'/job['dataset']/'gat_gine/checkpoint.pt')
  oldfullck=torch.load(oldfull,map_location='cpu',weights_only=False)
  assert oldck['model_config']==oldfullck['model_config'] and oldck['class_weights']==oldfullck['class_weights'] and oldck['training_observed_gene_ids']==oldfullck['training_observed_gene_ids']
  assert oldck['training_cell_ids']==ck['training_cell_ids'] and oldck['validation_cell_ids']==ck['validation_cell_ids']
  for column in ['cell_id','y','split']:assert new[column].tolist()==previous[column].tolist()==cells[column].tolist()
  for split in ['train','validation','test']:
   scalar_check(new[new.split.eq(split)],m[split]);scalar_check(previous[previous.split.eq(split)],prior[split])
   for method,record in [('fixed_k',prior),('cumulative',m)]:split_rows.append(dict(dataset=job['dataset'],fold=job['fold'],method=method,split=split,**{k:v for k,v in record[split].items() if not isinstance(v,(list,dict))}))
  sample=sorted({int(ix) for s in ['train','validation','test'] for ix in [np.flatnonzero(cells.split.eq(s))[0],np.flatnonzero(cells.split.eq(s))[-1]]})
  from torch.utils.data import Subset
  model,checkpoint=core.restore(folder/'checkpoint.pt','cpu');fresh=core.predict(model,Subset(data,sample),cells.iloc[sample].reset_index(drop=True),'gat_gine','cpu',32);saved=new.set_index('cell_id').loc[fresh.cell_id]
  error=float(np.abs(fresh.margin_resistant_minus_sensitive.to_numpy()-saved.margin_resistant_minus_sensitive.to_numpy()).max());assert np.allclose(fresh.margin_resistant_minus_sensitive,saved.margin_resistant_minus_sensitive,atol=2e-4,rtol=2e-5)
  for i in sample:assert data[i].edge_index.shape==(2,0) and data[i].edge_attr.shape==(0,1)
  info=json.loads((Path(job['cohort_root'])/'data/cohorts'/job['dataset']/'data_sources.json').read_text());old_k=info.get('selected_k',info.get('top_k'));assert old_k is not None
  rows.append(dict(dataset=job['dataset'],fold=job['fold'],mass_threshold=q,fixed_k=int(old_k),n_train=len(train),n_test=int(cells.split.eq('test').sum()),n_cpu_reload_predictions=len(sample),max_cpu_margin_difference=error,peak_allocated_GiB=m['training_and_inference_peak_allocated_GiB'],epochs=m['epochs_run'],fit_and_initial_prediction_seconds=m['training_seconds'],fixed_k_predictions=str(old/'predictions.tsv'),status='pass'))
 pd.DataFrame(rows).to_csv(TABLES/'source_validation.tsv',sep='\t',index=False);pd.DataFrame(split_rows).to_csv(TABLES/'split_metrics.tsv',sep='\t',index=False)

def verify_transfers():
 rows=[];source=next(j for j in jobs() if j['dataset']=='GSE223779');q=policy(source)['threshold'];model,ck=core.restore(full.trial_folder(source,CONTROL)/'checkpoint.pt','cpu')
 for ds in ['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']:
  folder=OUT/CONTROL/'transfer'/f'GSE223779_to_{ds}'/'gat_gine';m=json.loads((folder/'metrics.json').read_text());assert m['source_mass_threshold']==m['target_mass_threshold']==q and not m['target_training'] and not m['target_threshold_tuning'];assert Path(m['checkpoint'])==full.trial_folder(source,CONTROL)/'checkpoint.pt'
  cells=pd.read_csv(BENCH/'data/cohorts'/ds/'cells.tsv',sep='\t',keep_default_na=False);pred=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
  ref=pd.read_csv(OUT/'state_15b/transfer'/f'GSE223779_to_{ds}'/'gat_gine/predictions.tsv',sep='\t',float_precision='round_trip')
  oldpred=pd.read_csv(old_folder(source).parents[2]/'transfer'/f'GSE223779_to_{ds}'/'gat_gine/predictions.tsv',sep='\t',float_precision='round_trip')
  for c in ['cell_id','y','split']:assert pred[c].tolist()==cells[c].tolist()==ref[c].tolist()
  for c in ['cell_id','y','split']:assert pred[c].tolist()==oldpred[c].tolist(),'Old fixed-K target population differs: '+ds+'/'+c
  assert not set(cells.cell_id)&set(ck['training_cell_ids']+ck['validation_cell_ids'])
  parts={'all_labeled':pred[pred.y.ge(0)]}
  if ds=='GSE140440':
   assert pred.cell_line.tolist()==ref.cell_line.tolist();parts.update({'cell_line_'+str(k):f for k,f in pred.groupby('cell_line')})
  elif ds=='GSE163836':parts.update(primary_FCIBC02_pair=pred[pred.cell_line.eq('FCIBC02')],additional_SUM149PR=pred[pred.original_group.eq('SUM149PR')])
  elif ds=='GSE223779_TS485T':parts.update(same_drug_crizotinib=pred[pred.drug.isin(['untreated','crizotinib'])],cross_drug_alectinib=pred[pred.drug.isin(['untreated','alectinib'])])
  elif ds=='GSE202234':parts.update({'treatment_'+str(k):f[f.y.ge(0)] for k,f in pred.groupby('treatment')})
  assert set(parts)==set(m['scopes'])
  for scope,part in parts.items():scalar_check(part,m['scopes'][scope],True)
  data=NoEdgeGraphs(ds,CONTROL,q,cells);valid=np.flatnonzero(cells.y.ge(0));sample=sorted({int(valid[0]),int(valid[-1]),*np.flatnonzero(cells.y.lt(0))[:1].tolist()})
  from torch.utils.data import Subset
  fresh=core.predict(model,Subset(data,sample),cells.iloc[sample].reset_index(drop=True),'gat_gine','cpu',32);stored=pred.set_index('cell_id').loc[fresh.cell_id];assert np.allclose(fresh.margin_resistant_minus_sensitive,stored.margin_resistant_minus_sensitive,atol=2e-4,rtol=2e-5)
  rows.append(dict(dataset=ds,n_scopes=len(parts),n_cpu_reload_predictions=len(sample),status='pass'))
 folder=OUT/CONTROL/'transfer/GSE165897_pre_EOC_to_GSE154600_EPI_external/gat_gine';m=json.loads((folder/'metrics.json').read_text());tasks=[j for j in jobs() if j['dataset']=='GSE165897_pre_EOC'];assert m['source_mass_thresholds']==[policy(j)['threshold'] for j in tasks] and not m['target_training'] and not m['target_threshold_tuning']
 ensemble=pd.read_csv(folder/'ensemble_cell_predictions.tsv',sep='\t',float_precision='round_trip');data_cells=pd.read_csv(DATA/'candidates/GSE154600_EPI_external/cells.tsv',sep='\t',keep_default_na=False);assert ensemble.cell_id.tolist()==data_cells.cell_id.tolist()
 oldclinical=BENCH/'output/transfer/ovarian_clinical/no_edges/GSE165897_pre_EOC_to_GSE154600_EPI_external/gat_gine'
 oldensemble=pd.read_csv(oldclinical/'ensemble_cell_predictions.tsv',sep='\t',float_precision='round_trip')
 for col in ['cell_id','y','patient_id']:assert ensemble[col].tolist()==oldensemble[col].tolist(),'Old/new clinical population differs: '+col
 predictions=[]
 for job in tasks:
  frame=pd.read_csv(folder/f'{job["fold"]}_cell_predictions.tsv',sep='\t',float_precision='round_trip');assert frame.cell_id.tolist()==ensemble.cell_id.tolist() and frame.y.tolist()==ensemble.y.tolist();predictions.append(frame.probability_resistant.to_numpy())
  model,ck=core.restore(full.trial_folder(job,CONTROL)/'checkpoint.pt','cpu');assert not set(ensemble.cell_id)&set(ck['training_cell_ids']+ck['validation_cell_ids']);data=NoEdgeGraphs('GSE154600_EPI_external',CONTROL,policy(job)['threshold'],ensemble);ix=[0,len(ensemble)-1]
  from torch.utils.data import Subset
  fresh=core.predict(model,Subset(data,ix),ensemble.iloc[ix].reset_index(drop=True),'gat_gine','cpu',32);stored=frame.set_index('cell_id').loc[fresh.cell_id];assert np.allclose(fresh.margin_resistant_minus_sensitive,stored.margin_resistant_minus_sensitive,atol=2e-4,rtol=2e-5)
 assert np.allclose(np.mean(predictions,axis=0),ensemble.probability_resistant)
 patient=pd.read_csv(folder/'ensemble_patient_predictions.tsv',sep='\t',float_precision='round_trip').set_index('patient_id');assert np.allclose(ensemble.groupby('patient_id').probability_resistant.mean().loc[patient.index],patient.probability_resistant)
 for frame,name in [(ensemble,'external_epithelial_cells'),(patient,'external_patients')]:scalar_check(frame.assign(margin_resistant_minus_sensitive=frame.probability_resistant),m['scopes'][name])
 rows.append(dict(dataset='GSE154600_EPI_external',n_scopes=2,n_cpu_reload_predictions=len(tasks)*2,status='pass'));pd.DataFrame(rows).to_csv(TABLES/'transfer_validation.tsv',sep='\t',index=False)

def report():
 aggregation.OUT=OUT;aggregation.destination=full.destination
 rows=aggregation.grouped_rows(CONTROL)+aggregation.transfer_rows(CONTROL);assert len(rows)==11
 for p in (OUT/CONTROL/'grouped').glob('*/summary/oof_patient_predictions.tsv'):
  a=pd.read_csv(p,sep='\t',float_precision='round_trip')
  if 'margin_resistant_minus_sensitive' in a:a.drop(columns='margin_resistant_minus_sensitive').to_csv(p,sep='\t',index=False,float_format='%.17g')
 new=pd.DataFrame(rows);new.to_csv(TABLES/'primary_no_edges_metrics.tsv',sep='\t',index=False)
 old=fixed_reference()
 # Independently reaggregate the old controls rather than trusting table filenames.
 for ds in sorted({j['dataset'] for j in jobs() if j['family']=='grouped'}):
  tasks=[j for j in jobs() if j['dataset']==ds];parts=[];scores=[]
  for job in tasks:
   cells=read_cells(job);frame=pd.read_csv(old_folder(job)/'predictions.tsv',sep='\t',float_precision='round_trip')
   if 'patient_id' in cells:frame['patient_id']=cells.patient_id.to_numpy()
   part=frame[frame.split.eq('test')].copy();parts.append(part);scores.append(core.metrics(part))
  reference=old[old.dataset.eq(ds)&old.model.eq(CONTROL)];assert len(reference)==1
  if 'patient_id' in parts[0]:
   patients=pd.concat(parts).groupby('patient_id').agg(y=('y','first'),probability_resistant=('probability_resistant','mean'));values=core.metrics(patients.assign(margin_resistant_minus_sensitive=patients.probability_resistant))
  else:values={metric:np.mean([s[metric] for s in scores]) for metric in METRICS}
  for metric in METRICS:assert np.isclose(values[metric],reference.iloc[0][metric],atol=1e-10),('Old grouped reference metric changed',ds,metric)
 current=pd.read_csv(FIG/'source_data/primary_metrics.tsv',sep='\t');contrasts=[];controls=[]
 for row in new.to_dict('records'):
  ds,scope=row['dataset'],row['scope'];previous=old[old.dataset.eq(ds)&old.scope.eq(scope)&old.model.eq(CONTROL)];assert len(previous)==1;previous=previous.iloc[0]
  for field in ['n_cells','n_evaluation_units','n_patients']:
   if field in row and field in previous and pd.notna(row[field]) and pd.notna(previous[field]):assert int(row[field])==int(previous[field])
  controls.append(dict(dataset=ds,scope=scope,**{field:row[field] for field in ['n_cells','n_evaluation_units','unit','resistant_prevalence'] if field in row},**{f'{method}_{metric}':record[metric] for method,record in [('fixed_k',previous),('cumulative',row)] for metric in METRICS}))
  for rel in RELATIONS:
   f=current[current.dataset.eq(ds)&current.scope.eq(scope)&current.model.eq(rel)];o=old[old.dataset.eq(ds)&old.scope.eq(scope)&old.model.eq(rel)];assert len(f)==len(o)==1
   for metric in METRICS:
    old_gain=o.iloc[0][metric]-previous[metric];gain=f.iloc[0][metric]-row[metric]
    contrasts.append(dict(dataset=ds,scope=scope,relation=rel,metric=metric,fixed_k_full=o.iloc[0][metric],fixed_k_no_edges=previous[metric],cumulative_full=f.iloc[0][metric],cumulative_no_edges=row[metric],fixed_k_edge_gain=old_gain,cumulative_edge_gain=gain,edge_gain_change=gain-old_gain,no_edges_change=row[metric]-previous[metric]))
 pd.DataFrame(controls).to_csv(TABLES/'fixedK_vs_cumulative_no_edges.tsv',sep='\t',index=False);pd.DataFrame(contrasts).to_csv(TABLES/'formal_edge_effects.tsv',sep='\t',index=False)
 training=pd.read_csv(FIG/'source_data/training_design_comparison.tsv',sep='\t');designs=[]
 for job in jobs():
  newm=json.loads((full.trial_folder(job,CONTROL)/'metrics.json').read_text())['test'];oldm=json.loads((old_folder(job)/'metrics.json').read_text())['test']
  for rel in RELATIONS:
   f=training[training.dataset.eq(job['dataset'])&training.fold.eq(job['fold'])&training.relation.eq(rel)];assert len(f)==1;f=f.iloc[0]
   for metric in ['auroc','balanced_accuracy']:
    old_gain=f['fixed_k_'+metric]-oldm[metric];gain=f['cumulative_'+metric]-newm[metric]
    designs.append(dict(dataset=job['dataset'],fold=job['fold'],relation=rel,metric=metric,n_test_cells=f.n_test_cells,fixed_k_full=f['fixed_k_'+metric],fixed_k_no_edges=oldm[metric],cumulative_full=f['cumulative_'+metric],cumulative_no_edges=newm[metric],fixed_k_edge_gain=old_gain,cumulative_edge_gain=gain,edge_gain_change=gain-old_gain))
 pd.DataFrame(designs).to_csv(TABLES/'training_design_edge_effects.tsv',sep='\t',index=False)
 print('COMPLETE_EDGE_COMPARISON 14 shared controls; 11 formal scopes; 44 paired graph effects; old/new fixed populations',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','validate','report'],default='all');a=p.parse_args();torch.set_num_threads(4);TABLES.mkdir(exist_ok=True,parents=True)
 if a.stage in ['all','validate']:verify_sources();verify_transfers();print('PASS 14 cumulative sources, 14 old controls and 5 target configurations',flush=True)
 if a.stage in ['all','report']:report()

if __name__=='__main__':main()

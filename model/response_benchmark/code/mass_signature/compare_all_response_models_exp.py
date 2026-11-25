"""Unify completed model families at the same formal scopes and original training splits."""
import json
import numpy as np,pandas as pd
from cumulative_common import *
import train_response_exp as core
import train_cumulative_benchmark_exp as full
from cumulative_edge_ablation_exp import old_folder,CONTROL,TABLES

def definitions():
 result=[]
 for method in ['fixed_k','cumulative']:
  for rel in RELATIONS:result.append(dict(model_id=method+'__'+rel,node_selection=method,graph=rel,node_features='trainable_gene_identity_16D_plus_expression',source_model=rel,source_group=method,display_name=('Fixed K' if method=='fixed_k' else 'Cumulative')+'\n'+{'state_15b':'15B graph','ppi':'PPI graph','qwen':'Qwen graph','esm2_650m':'650M graph'}[rel]))
 for method in ['fixed_k','cumulative']:result.append(dict(model_id=method+'__no_edges',node_selection=method,graph='none',node_features='trainable_gene_identity_16D_plus_expression',source_model='no_edges',source_group=method,display_name=('Fixed K' if method=='fixed_k' else 'Cumulative')+'\nNo edges'))
 result.extend([
  dict(model_id='fixed_k__random_signature',node_selection='matched_random_signature',graph='state_15b',node_features='trainable_gene_identity_16D_plus_expression',source_model='random_signature',source_group='fixed_k',display_name='Random genes\n15B graph'),
  dict(model_id='native_se__mlp',node_selection='not_applicable',graph='none',node_features='native_STATE_SE_cell_embedding_2048D',source_model='native_state_embedding',source_group='fixed_k',display_name='Native SE\nMLP'),
  dict(model_id='fixed_k__state_graph__state_15b',node_selection='fixed_k',graph='state_15b',node_features='frozen_ESM2_15B_gene_embedding_5120D_plus_expression',source_model='state_graph__state_15b',source_group='pretrained',display_name='Fixed K: 15B graph\n15B gene emb')])
 return result

def folder(job,model):
 ds,fold=job['dataset'],job['fold'];rel=model['source_model'];simple=job['family']=='within'
 if model['source_group']=='cumulative':return full.trial_folder(job,rel)
 if model['source_group']=='pretrained':
  base=BENCH/'output/gene_embedding'/rel
  if not simple:base=base/'grouped'/ds/fold
  return base/'within'/ds/'gat_gine'
 if rel==CONTROL:return old_folder(job)
 if simple:
  if rel=='native_state_embedding':return BENCH/'output/within'/ds/'state_mlp'
  if rel=='random_signature':return BENCH/'output/transfer/signature_ablation/random_genes/state_15b/within'/ds/'gat_gine'
  return (BENCH/'output' if rel=='state_15b' else BENCH/'output/transfer'/rel)/'within'/ds/'gat_gine'
 return BENCH/'output/grouped_adaptive'/ds/fold/rel/'within'/ds/('state_mlp' if rel=='native_state_embedding' else 'gat_gine')

def main():
 tables=FIG/'source_data';defs=definitions();assert len(defs)==13
 pd.DataFrame(defs).to_csv(tables/'all_model_definitions.tsv',sep='\t',index=False)
 paths={'fixed_k':TABLES/'fixedK_reference_metrics.tsv','cumulative':tables/'primary_metrics.tsv','pretrained':ROOT/'work/transfer_analysis/data/gene_embedding/pretrained_primary_metrics.tsv'}
 frames={key:pd.read_csv(p,sep='\t') for key,p in paths.items()};noedge=pd.read_csv(TABLES/'primary_no_edges_metrics.tsv',sep='\t');frames['cumulative']=pd.concat([frames['cumulative'],noedge],ignore_index=True)
 scopes=list(noedge[['dataset','scope']].itertuples(index=False,name=None));assert len(scopes)==11
 rows=[]
 for ds,scope in scopes:
  reference=frames['fixed_k'];reference=reference[reference.dataset.eq(ds)&reference.scope.eq(scope)&reference.model.eq('state_15b')].iloc[0]
  for model in defs:
   frame=frames[model['source_group']];part=frame[frame.dataset.eq(ds)&frame.scope.eq(scope)&frame.model.eq(model['source_model'])];assert len(part)==1,(ds,scope,model['model_id']);record=part.iloc[0].to_dict()
   for field in ['n_cells','n_evaluation_units','n_patients','n_sensitive','n_resistant','n_sensitive_patients','n_resistant_patients']:
    if field in record and field in reference and pd.notna(record[field]) and pd.notna(reference[field]):assert int(record[field])==int(reference[field]),(ds,scope,model['model_id'],field)
   assert record['threshold']==.5 and pd.notna(record['auroc']) and pd.notna(record['balanced_accuracy'])
   rows.append(dict(record,model_id=model['model_id'],node_selection=model['node_selection'],graph=model['graph'],node_features=model['node_features'],source_table=str(TABLES/'primary_no_edges_metrics.tsv' if model['model_id']=='cumulative__no_edges' else paths[model['source_group']]),result_status='completed'))
 table=pd.DataFrame(rows);assert len(table)==143 and not table.duplicated(['dataset','scope','model_id']).any();table.to_csv(tables/'all_models_formal_comparison.tsv',sep='\t',index=False)
 training=[]
 for job in jobs():
  cells=read_cells(job)
  for model in defs:
   loc=folder(job,model);record=dict(dataset=job['dataset'],fold=job['fold'],model_id=model['model_id'],metrics_path=str(loc/'metrics.json'),result_status='missing_result')
   if (loc/'metrics.json').exists():
    saved=json.loads((loc/'metrics.json').read_text());pred=pd.read_csv(loc/'predictions.tsv',sep='\t',float_precision='round_trip')
    for col in ['cell_id','y','split']:assert pred[col].tolist()==cells[col].tolist(),(job['dataset'],job['fold'],model['model_id'],col)
    scores=core.metrics(pred[pred.split.eq('test')])
    for metric in ['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','macro_f1','brier_score']:assert np.isclose(scores[metric],saved['test'][metric],atol=1e-10)
    record.update({k:v for k,v in scores.items() if not isinstance(v,(dict,list))});record['result_status']='completed'
   training.append(record)
 pd.DataFrame(training).to_csv(tables/'all_models_training_comparison.tsv',sep='\t',index=False)
 print('ALL_MODEL_COMPARISON',len(defs),'configurations x',len(scopes),'formal scopes:',len(table),'results;',sum(r['result_status']=='completed' for r in training),'/',len(training),'individual design results',flush=True)

if __name__=='__main__':main()

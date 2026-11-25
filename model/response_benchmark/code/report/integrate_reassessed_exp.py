"""Build the active comparison after original-condition reassessment."""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True);p.add_argument('--strata_path',required=True);p.add_argument('--save_path',required=True);a=p.parse_args()
source=Path(a.open_path);out=Path(a.save_path)
old=pd.read_csv(source/'mechanism_combined_metrics.tsv',sep='\t')
exclude=['patient_grouped_OOF_cells','replicate_grouped_OOF_cells','frozen_ovarian_external_epithelial_cells']
old=old[~old.scope.isin(exclude)&~old.scope.str.startswith('held_out_replicate_')]
old=old[~old.dataset.isin(['GSE140440','GSE149383','GSE186960','GSE202234'])].copy()
strata=pd.read_csv(a.strata_path,sep='\t')
strata['scope']='frozen_cell_line_'+strata.cell_line
state=pd.read_csv(source/'pc9_state_grouped_metrics.tsv',sep='\t');state=state[state.scope.eq('sample_set_macro')].copy()
if len(state)!=7:raise ValueError('PC9 state experiment incomplete')
table=pd.concat([old,strata,state],ignore_index=True)
if table.duplicated(['dataset','model','scope']).any() or not table.groupby(['dataset','scope']).size().eq(7).all():raise ValueError('Incomplete comparison')
table['endpoint']='response_or_fate';table.loc[table.dataset.str.contains('PC9_DTEP'),'endpoint']='DTEP_treatment_state'
table.loc[table.dataset.eq('GSE223003_nk_pre'),'endpoint']='NK_survival_fate'
out.mkdir(parents=True,exist_ok=True);table.to_csv(out/'primary_comparison.tsv',sep='\t',index=False,float_format='%.17g')
rows=[]
for (dataset,scope),part in table.groupby(['dataset','scope']):
 indexed=part.set_index('model')
 for model,control,name in [('state_15b','no_edges','STATE_graph_minus_no_edges'),('esm2_650m','no_edges','ESM_graph_minus_no_edges'),('qwen','no_edges','Qwen_graph_minus_no_edges'),('ppi','no_edges','PPI_graph_minus_no_edges'),('state_15b','random_signature','IG_minus_random_signature')]:
  row=dict(dataset=dataset,scope=scope,model=model,control=control,contrast=name)
  for metric in ['auroc','balanced_accuracy','auprc_resistant']:row['delta_'+metric]=indexed.loc[model,metric]-indexed.loc[control,metric]
  rows.append(row)
pd.DataFrame(rows).to_csv(out/'matched_ablation_differences.tsv',sep='\t',index=False,float_format='%.17g')
print('ACTIVE_PRIMARY_ROWS',len(table),'SCOPES',table[['dataset','scope']].drop_duplicates().shape[0])

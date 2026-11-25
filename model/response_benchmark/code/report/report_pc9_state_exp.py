"""Summarize held-out PC9 sample sets with DTEP as the positive class."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

from pathlib import Path
import json
import pandas as pd
from train_response_exp import metrics,write_json
from pc9_state_grouped_exp import TASK,MODELS
ROOT=BENCH_ROOT
def main():
 rows=[]
 for model in MODELS:
  kind='state_mlp' if model=='native_state_embedding' else 'gat_gine';parts=[];fold_values=[]
  for fold in [1,2]:
   folder=ROOT/'output/grouped_adaptive'/TASK/f'fold_{fold}'/model/'within'/TASK/kind
   frame=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
   cells=pd.read_csv(ROOT/'data/grouped'/TASK/f'fold_{fold}'/'data/cohorts'/TASK/'cells.tsv',sep='\t')
   if frame.cell_id.tolist()!=cells.cell_id.tolist() or not frame.y.eq(cells.y).all():raise ValueError('Prediction identity or label mismatch')
   for column in ['sample_id','state_label','outer_sample_partition']:frame[column]=cells[column].to_numpy()
   test=frame[frame.split.eq('test')].copy();test['fold']=fold;parts.append(test)
   value=metrics(test);fold_values.append(value)
   rows.append(dict(dataset=TASK,model=model,scope=f'held_out_sample_set_{fold}',positive_class='DTEP_day11',**{k:v for k,v in value.items() if not isinstance(v,list)}))
  merged=pd.concat(parts,ignore_index=True)
  if not merged.cell_id.is_unique or merged.groupby('sample_id').fold.nunique().gt(1).any():raise ValueError('Outer sample leakage')
  out=ROOT/'output/grouped_adaptive'/TASK/'summary'/model;out.mkdir(parents=True,exist_ok=True)
  merged.to_csv(out/'oof_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
  macro={key:sum(v[key] for v in fold_values)/2 for key in ['auroc','auprc_resistant','balanced_accuracy','mcc','macro_f1','brier_score','resistant_prevalence']}
  macro.update(n_cells=len(merged),n_samples=merged.sample_id.nunique(),evaluation_unit='equal mean of two held-out sample-set metrics')
  rows.append(dict(dataset=TASK,model=model,scope='sample_set_macro',positive_class='DTEP_day11',n_sample_sets=2,**macro))
  write_json(out/'state_metrics.json',dict(positive_class='DTEP_day11',negative_class='untreated_day0',endpoint='treatment state, not stable resistance',sample_sets=fold_values,macro=macro,limitation='two samples per state; source inner validation is random cells; one seed'))
 table=pd.DataFrame(rows);table.to_csv(ROOT/'output/summary/pc9_state_grouped_metrics.tsv',sep='\t',index=False,float_format='%.17g')
 print(table[table.scope.eq('sample_set_macro')].to_string(index=False),flush=True)
if __name__=='__main__':main()

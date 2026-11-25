"""Sample-held-out PC9 DTEP-state classification with the established model pipeline."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from split_registry import apply_splits, source_folds
import mechanism_grouped_exp as prep
TASK='GSE149383_PC9_DTEP_sample_grouped'
MODELS=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']
def cohorts(root,include_embeddings):
 cells=pd.read_csv(prep.INDEX_ROOT/TASK/'cells.tsv',sep='\t')
 labels=pd.read_csv(prep.METADATA_ROOT/TASK/'counts_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
 for column in labels:cells[column]=cells.cell_id.map(labels[column])
 cells['y']=cells.state_y.astype(int);cells['response']=cells.state_label;cells['graph_row']=np.arange(len(cells))
 if cells.sample_id.nunique()!=4 or cells.y.nunique()!=2:raise ValueError('Invalid state task')
 x=None
 if include_embeddings:
  metadata=pd.read_csv(prep.SE_ROOT/TASK/'cell_metadata.tsv',sep='\t');index=pd.Index(metadata.cell_id).get_indexer(cells.cell_id)
  if (index<0).any():raise ValueError('SE cell mismatch')
  x=np.load(prep.SE_ROOT/TASK/'cell_embeddings.npy')[index,:2048]
 for fold_name in source_folds(TASK):
  fold=int(fold_name.split('_')[1]);frame=apply_splits(cells,TASK,fold_name,mode='source')
  test=frame.split.eq('test')
  if set(frame.loc[test,'sample_id'])&set(frame.loc[~test,'sample_id']):raise ValueError('Outer sample leakage')
  folder=prep.BENCH/'data/grouped'/TASK/f'fold_{fold}'/'data/cohorts'/TASK;folder.mkdir(parents=True,exist_ok=True)
  frame.to_csv(folder/'cells.tsv',sep='\t',index=False)
  if x is not None:np.save(folder/'state_cell_embeddings.npy',x)
  info=dict(task=TASK,graph_source=str(prep.INDEX_ROOT/TASK),group_key='sample_id',outer_folds=2,fold=fold,
   positive_class='DTEP_day11',negative_class='untreated_day0',endpoint='treatment state, not stable resistance',
   test_samples=sorted(frame.loc[test,'sample_id'].unique()),source_samples=sorted(frame.loc[~test,'sample_id'].unique()),
   inner_validation='20% stratified cells within source samples; not independent culture validation',
   limitation='two samples per state; no claim of matched culture pairing; one training seed',seed=42)
  (folder/'data_sources.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
  print('PC9_SAMPLE_SPLIT',fold,frame.groupby(['split','y']).size().to_dict(),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
 p.add_argument('--stage',choices=['index','split_profile','state','cohorts','graphs','random','train'],required=True)
 a=p.parse_args();root=Path(a.open_path)
 if a.stage=='index':prep.index_stage(root,TASK)
 elif a.stage=='split_profile':cohorts(root,False);prep.profile_stage(root,TASK)
 elif a.stage=='state':prep.state_stage(root,TASK)
 elif a.stage=='cohorts':cohorts(root,True)
 elif a.stage=='graphs':prep.graphs_stage(root,TASK)
 elif a.stage=='random':prep.random_stage(root,TASK)
 elif a.stage=='train':
  prep.train_stage(root,TASK,MODELS)
  for path in (prep.BENCH/'output/grouped_adaptive'/TASK).rglob('metrics.json'):
   value=json.loads(path.read_text());value['scope']='sample-held-out PC9 DTEP treatment-state prediction';value['positive_class']='DTEP_day11';value['negative_class']='untreated_day0'
   path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()

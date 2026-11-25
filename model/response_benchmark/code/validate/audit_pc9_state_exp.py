"""Verify PC9 state inputs and all saved model predictions without retraining."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from mechanism_grouped_exp import BENCH,INDEX_ROOT,POLICY_ROOT,SE_ROOT,graph_package
from audit_grouped_inputs_exp import verify_predictions
from pc9_state_grouped_exp import TASK
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs_only',action='store_true');a=p.parse_args()
expected=pd.read_csv(INDEX_ROOT/TASK/'cells.tsv',sep='\t').cell_id.tolist();all_test=[];records=[]
state=pd.read_csv(SE_ROOT/TASK/'cell_metadata.tsv',sep='\t');index=pd.Index(state.cell_id).get_indexer(expected)
if not state.cell_id.is_unique or (index<0).any():raise ValueError('Native SE identities differ')
x=np.load(SE_ROOT/TASK/'cell_embeddings.npy',mmap_mode='r')[index,:2048]
if x.shape!=(len(expected),2048) or not np.isfinite(x).all():raise ValueError('Native SE values invalid')
for fold in [1,2]:
 name=f'fold_{fold}';folder=BENCH/'data/grouped'/TASK/name/'data/cohorts'/TASK;cells=pd.read_csv(folder/'cells.tsv',sep='\t')
 if cells.cell_id.tolist()!=expected or not cells.y.eq(cells.state_y).all():raise ValueError('Cell order or state labels differ')
 if not cells.response.eq(cells.state_label).all():raise ValueError('Response is not the declared state endpoint')
 test=cells.split.eq('test');all_test.extend(cells.loc[test,'cell_id'])
 if set(cells.loc[test,'sample_id'])&set(cells.loc[~test,'sample_id']):raise ValueError('Outer sample leakage')
 for _,part in cells.groupby('split'):
  if part.y.nunique()!=2:raise ValueError('Missing state class')
 training=cells.loc[cells.split.eq('train'),'cell_id'].tolist();cal=pd.read_csv(POLICY_ROOT/TASK/name/'calibration_cells.tsv',sep='\t')
 if training!=cal.cell_id.tolist():raise ValueError('Nontraining K calibration')
 k=json.loads((POLICY_ROOT/TASK/name/'dataset_policy.json').read_text())['datasets'][TASK]['k']
 if k!=768 or not np.array_equal(x,np.load(folder/'state_cell_embeddings.npy')):raise ValueError('K or SE order differs')
 if not a.inputs_only:verify_predictions(TASK,name,cells)
 for random in [False,True]:
  package=graph_package(k,random)/TASK;gcells=pd.read_csv(package/'cells.tsv',sep='\t')
  if gcells.cell_id.tolist()!=expected:raise ValueError('Graph order differs')
  with np.load(package/'signatures.npz') as z:
   ptr=z['ptr'];nodes=z['gene_id'];features=z['features'];sizes=np.diff(ptr)
   if len(ptr)!=len(expected)+1 or ptr[0]!=0 or ptr[-1]!=len(nodes) or (sizes<=0).any():raise ValueError('Invalid signature pointers')
   if not random and (sizes>k).any():raise ValueError('Signature exceeds K')
   if len(features)!=len(nodes) or not np.isfinite(features).all() or (features[:,0]<=0).any():raise ValueError('Invalid node features')
   for lo,hi in zip(ptr[:-1],ptr[1:]):
    if len(set(nodes[lo:hi]))!=hi-lo:raise ValueError('Duplicate node')
  if random:
   qc=pd.read_csv(package/'matching_qc.tsv',sep='\t')
   if len(qc)!=len(expected) or not qc.matching_passed.astype(str).str.lower().eq('true').all():raise ValueError('Random matching failed')
  records.append(dict(dataset=TASK,fold=name,signature='random' if random else 'IG',n_cells=len(expected),min_nodes=int(sizes.min()),median_nodes=float(np.median(sizes)),max_nodes=int(sizes.max()),k=k,input_checks_passed=True,n_prediction_models_checked=0 if a.inputs_only else 7))
if len(all_test)!=len(expected) or set(all_test)!=set(expected):raise ValueError('OOF test coverage differs')
pd.DataFrame(records).to_csv(BENCH/'output/summary/pc9_state_input_audit.tsv',sep='\t',index=False)
print(pd.DataFrame(records).to_string(index=False))

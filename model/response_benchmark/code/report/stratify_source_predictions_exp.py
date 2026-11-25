"""Evaluate saved frozen predictions within original cell-line strata."""
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_paths import BENCH_ROOT, PROJECT_ROOT
from train_response_exp import metrics
def main():
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--benchmark_path", type=Path, default=BENCH_ROOT)
 parser.add_argument("--conditions_path", type=Path, default=PROJECT_ROOT / "dataset_util/datasets_download_preprocess/26_11_20/resources/sample_conditions.tsv")
 parser.add_argument("--save_path", type=Path, required=True)
 args = parser.parse_args()
 conditions = pd.read_csv(args.conditions_path, sep="\t")
 labels = conditions.loc[conditions.dataset.eq("GSE140440")].set_index("sample_id", verify_integrity=True)
 base=args.benchmark_path/'output';rows=[]
 paths={'state_15b':base,'native_state_embedding':base,'no_edges':base/'transfer/edge_ablation/no_edges/shared','random_signature':base/'transfer/signature_ablation/random_genes/state_15b'}
 for model in ['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']:
  folder=paths.get(model,base/'transfer'/model);kind='state_mlp' if model=='native_state_embedding' else 'gat_gine'
  path=folder/'transfer/GSE223779_to_GSE140440'/kind/'predictions.tsv'
  frame=pd.read_csv(path,sep='\t',float_precision='round_trip');frame['cell_line']=frame.cell_id.map(labels['cell line'])
  if frame.cell_line.isna().any() or not frame.cell_id.is_unique:raise ValueError('Unmapped prediction')
  for line,part in frame.groupby('cell_line'):
   values=metrics(part);rows.append(dict(dataset='GSE140440',cell_line=line,model=model,scope='all_labeled_within_cell_line',prediction_source=str(path),**{k:v for k,v in values.items() if not isinstance(v,list)}))
 table=pd.DataFrame(rows);args.save_path.parent.mkdir(parents=True,exist_ok=True);table.to_csv(args.save_path,sep='\t',index=False,float_format='%.17g')
 print(table[['cell_line','model','n_cells','auroc','balanced_accuracy']].to_string(index=False))
if __name__=='__main__': main()

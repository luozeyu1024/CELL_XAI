"""Export logits and float64 probabilities from existing checkpoints; do not refit anything."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_response_exp import DATASETS,load_cohort,restore,predict,metrics,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();root=Path(a.open_path).resolve()
    torch.set_num_threads(4)
    for ds in DATASETS:
        for kind in ['gat_gine','state_mlp']:
            data,cells,genes=load_cohort(root,ds,kind)
            local=root/'output/within'/ds/kind
            model,ckpt=restore(local/'checkpoint.pt',a.device)
            size=ckpt['parameters']['graph_batch_size'] if kind=='gat_gine' else ckpt['parameters']['mlp_batch_size']
            pred=predict(model,data,cells,kind,a.device,size)
            pred.to_csv(local/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
            saved=json.loads((local/'metrics.json').read_text())
            for split in ['train','validation','test']:saved[split]=metrics(pred[pred.split.eq(split)])
            write_json(local/'metrics.json',saved)
            del model
            if ds!='GSE223779':
                source=root/'output/within/GSE223779'/kind/'checkpoint.pt'
                frozen,source_ckpt=restore(source,a.device)
                assert not set(cells.cell_id)&set(source_ckpt['training_cell_ids']+source_ckpt['validation_cell_ids'])
                if kind=='gat_gine':assert genes==source_ckpt['gene_symbols']
                pred=predict(frozen,data,cells,kind,a.device,size)
                dest=root/'output/transfer'/f'GSE223779_to_{ds}'/kind
                pred.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
                saved=json.loads((dest/'metrics.json').read_text())
                saved['target_test']=metrics(pred[pred.split.eq('test')]);saved['target_all_labeled']=metrics(pred)
                write_json(dest/'metrics.json',saved)
                del frozen
            del data
            torch.cuda.empty_cache()
            print('Exported frozen predictions:',ds,kind,flush=True)


if __name__=='__main__':main()

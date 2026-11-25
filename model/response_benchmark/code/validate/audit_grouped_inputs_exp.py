"""Check cell identity, graph signatures, grouped folds and training-only K calibration."""

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
import pandas as pd
from mechanism_grouped_exp import ROOT,BENCH,INDEX_ROOT,POLICY_ROOT,SE_ROOT,selected_k,graph_package

MODELS=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']


def verify_predictions(task,fold,cells):
    from train_response_exp import metrics
    for model in MODELS:
        kind='state_mlp' if model=='native_state_embedding' else 'gat_gine'
        folder=BENCH/'output/grouped_adaptive'/task/fold/model/'within'/task/kind
        frame=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
        saved=json.loads((folder/'metrics.json').read_text())
        if frame.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Prediction cell identity/order mismatch')
        for column in ['y','response','split']:
            if frame[column].astype(str).tolist()!=cells[column].astype(str).tolist(): raise ValueError('Prediction labels/splits changed')
        values=frame[['probability_resistant','logit_sensitive','logit_resistant','margin_resistant_minus_sensitive']].to_numpy()
        if not np.isfinite(values).all() or not frame.probability_resistant.between(0,1).all(): raise ValueError('Invalid prediction values')
        if not np.allclose(frame.margin_resistant_minus_sensitive,frame.logit_resistant-frame.logit_sensitive,rtol=0,atol=1e-12):
            raise ValueError('Prediction score direction mismatch')
        if not frame.predicted.eq(frame.probability_resistant.ge(.5).astype(int)).all(): raise ValueError('Threshold changed')
        for split in ['train','validation','test']:
            recalculated=metrics(frame[frame.split.eq(split)])
            for name,value in recalculated.items():
                previous=saved[split][name]
                if isinstance(value,(float,int)):
                    if not np.isclose(value,previous,rtol=0,atol=1e-12): raise ValueError('Saved metric differs from predictions: '+name)
                elif value!=previous: raise ValueError('Saved metric field mismatch: '+name)


def audit(task,allow_pending_se):
    candidates=pd.read_csv(INDEX_ROOT/task/'cells.tsv',sep='\t',keep_default_na=False)
    expected=candidates.cell_id.tolist()
    if not candidates.cell_id.is_unique: raise ValueError('Duplicate candidate cell IDs')
    eligible=pd.read_csv(INDEX_ROOT/task/'eligible_genes.tsv',sep='\t')
    allowed=eligible.gene_id.to_numpy()
    se=SE_ROOT/task
    se_ready=(se/'validation.json').exists()
    if se_ready:
        validation=json.loads((se/'validation.json').read_text())
        if validation['status']!='passed': raise ValueError('Native SE validation did not pass')
        metadata=pd.read_csv(se/'cell_metadata.tsv',sep='\t',keep_default_na=False)
        index=pd.Index(metadata.cell_id).get_indexer(expected)
        if not metadata.cell_id.is_unique or (index<0).any(): raise ValueError('SE cells do not match candidates')
        embedding=np.load(se/'cell_embeddings.npy',mmap_mode='r')
        if embedding.shape!=(len(expected),2048) or not np.isfinite(embedding).all(): raise ValueError('Invalid SE matrix')
    elif not allow_pending_se: raise ValueError('Native SE inference is not complete')
    rows=[];test_cells=[]
    for fold in sorted((BENCH/'data/grouped'/task).glob('fold_*')):
        number=int(fold.name.split('_')[1]);k=selected_k(task,number)
        folder=fold/'data/cohorts'/task
        cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False)
        if cells.cell_id.tolist()!=expected: raise ValueError('Fold cell order mismatch')
        if not cells.split.isin(['train','validation','test']).all(): raise ValueError('Unexpected split')
        if not cells.y.isin([0,1]).all(): raise ValueError('Unknown label entered supervised task')
        if not cells.response.map({'sensitive':0,'resistant':1}).eq(cells.y).all(): raise ValueError('Response direction mismatch')
        for name,part in cells.groupby('split'):
            if part.y.nunique()!=2: raise ValueError('Both classes required in each split: '+name)
        calibration=pd.read_csv(POLICY_ROOT/task/fold.name/'calibration_cells.tsv',sep='\t')
        if calibration.cell_id.tolist()!=cells.loc[cells.split.eq('train'),'cell_id'].tolist():
            raise ValueError('K calibration includes nontraining cells or differs in order')
        if task.startswith('GSE223003_'):
            if cells.loc[cells.split.eq('test'),'biological_replicate'].nunique()!=1:
                raise ValueError('Mixed outer test replicates')
            group_sets=[set(cells.loc[cells.split.eq(name),'biological_replicate'].astype(str)) for name in ['train','validation','test']]
            if group_sets[0]!=group_sets[1] or group_sets[0]&group_sets[2]: raise ValueError('Outer replicate leakage')
            known=cells[cells.y.eq(1)&cells.prer_r_group.astype(str).str.lower().isin(['','nan','na','none']).eq(False)&~cells.split.eq('test')]
            if (known.groupby('prer_r_group').split.nunique()>1).any(): raise ValueError('Resistant lineage crossed training/validation')
        else:
            if (cells.groupby('patient_id').split.nunique()>1).any(): raise ValueError('Patient crossed fold partitions')
        if se_ready:
            folded=np.load(folder/'state_cell_embeddings.npy',mmap_mode='r')
            if not np.array_equal(folded,embedding[index,:2048]): raise ValueError('Fold SE values/order changed')
        if not allow_pending_se: verify_predictions(task,fold.name,cells)
        for random in [False,True]:
            package=graph_package(k,random);target=package/task
            graph_cells=pd.read_csv(target/'cells.tsv',sep='\t',keep_default_na=False)
            if graph_cells.cell_id.tolist()!=expected: raise ValueError('Graph cell order mismatch')
            with np.load(target/'signatures.npz') as z:
                ptr=z['ptr'];nodes=z['gene_id'];features=z['features']
                counts=np.diff(ptr)
                if len(ptr)!=len(expected)+1 or ptr[0]!=0 or ptr[-1]!=len(nodes): raise ValueError('Invalid signature pointers')
                if len(features)!=len(nodes) or not np.isfinite(features).all() or (features[:,0]<=0).any():
                    raise ValueError('Invalid node expression features')
                if not np.isin(nodes,allowed).all(): raise ValueError('Node outside fixed eligible vocabulary')
                if (counts<=0).any() or (not random and (counts>k).any()): raise ValueError('Invalid node counts')
                for lo,hi in zip(ptr[:-1],ptr[1:]):
                    if np.unique(nodes[lo:hi]).size!=hi-lo: raise ValueError('Repeated gene within a cell signature')
            if random:
                qc=pd.read_csv(target/'matching_qc.tsv',sep='\t')
                if len(qc)!=len(expected) or not qc.matching_passed.astype(str).str.lower().eq('true').all():
                    raise ValueError('Random structure matching incomplete')
            rows.append(dict(dataset=task,fold=fold.name,signature='matched_random' if random else 'IG',k=k,n_cells=len(expected),
                             median_nodes=float(np.median(counts)),min_nodes=int(counts.min()),max_nodes=int(counts.max()),
                             cell_order_passed=True,split_and_calibration_passed=True,graph_features_passed=True,se_ready=se_ready,
                             n_prediction_models_checked=0 if allow_pending_se else len(MODELS)))
        test_cells.extend(cells.loc[cells.split.eq('test'),'cell_id'])
    if len(test_cells)!=len(expected) or set(test_cells)!=set(expected): raise ValueError('Outer tests do not cover every cell exactly once')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',nargs='+',required=True)
    parser.add_argument('--save_path',required=True)
    parser.add_argument('--allow_pending_se',action='store_true')
    args=parser.parse_args();rows=[]
    for task in args.tasks: rows.extend(audit(task,args.allow_pending_se))
    out=Path(args.save_path);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,sep='\t',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__': main()

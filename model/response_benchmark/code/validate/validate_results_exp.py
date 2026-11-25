"""Verify shared splits, train-only scaling, saved metrics and reloaded local/transfer predictions."""

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
import anndata as ad
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,matthews_corrcoef
from sklearn.metrics import f1_score,brier_score_loss,confusion_matrix
from train_response_exp import DATASETS,load_cohort,restore,predict


def check_metrics(frame,saved):
    f=frame[frame.y.ge(0)];y=f.y.to_numpy();s=f.probability_resistant.to_numpy();pred=s>=.5
    score=f.margin_resistant_minus_sensitive.to_numpy()
    assert len(f)==saved['n_cells']
    assert saved['ranking_score']=='resistant_minus_sensitive_logit'
    expected={'auroc':roc_auc_score(y,score),'auprc_resistant':average_precision_score(y,score),
        'auprc_sensitive':average_precision_score(1-y,-score),'balanced_accuracy':balanced_accuracy_score(y,pred),
        'mcc':matthews_corrcoef(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
        'brier_score':brier_score_loss(y,s),'resistant_prevalence':y.mean()}
    for key,v in expected.items():assert np.isclose(v,saved[key],rtol=1e-7,atol=1e-10),(key,v,saved[key])
    assert confusion_matrix(y,pred,labels=[0,1]).tolist()==saved['confusion_matrix_true_rows_sensitive_resistant']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();root=Path(a.open_path).resolve()
    torch.set_num_threads(4);checks=[];source='GSE223779'
    for ds in DATASETS:
        folder=root/'data/cohorts'/ds;cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False)
        assert cells.cell_id.is_unique
        known=cells.y.ge(0);assert (cells.loc[~known,'split']=='excluded_unknown_response').all()
        assert cells.loc[known,'split'].isin(['train','validation','test']).all()
        train=np.flatnonzero(cells.split.eq('train'));validation=np.flatnonzero(cells.split.eq('validation'))
        test=np.flatnonzero(cells.split.eq('test'))
        info=json.loads((folder/'data_sources.json').read_text())
        expected_train,holdout=train_test_split(np.flatnonzero(known),test_size=.30,
            stratify=cells.loc[known,'y'],random_state=info['seed'])
        expected_validation,expected_test=train_test_split(holdout,test_size=1/3,
            stratify=cells.y.iloc[holdout],random_state=info['seed'])
        assert set(train)==set(expected_train) and set(validation)==set(expected_validation) and set(test)==set(expected_test)
        labels=pd.read_excel(info['label_source'],dtype=str).set_index('Cell_ID').label
        expected_y=cells.cell_id.map(labels).map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)
        assert np.array_equal(cells.y,expected_y)
        native=ad.read_h5ad(info['state_source'],backed='r')
        native_index=pd.Index(native.obs_names.astype(str)).get_indexer(cells.cell_id)
        assert np.array_equal(native_index,cells.state_row) and (native_index>=0).all()
        prepared=np.load(folder/'state_cell_embeddings.npy',mmap_mode='r')
        native_values=np.asarray(native.obsm[info['state_key']])
        assert np.array_equal(prepared,native_values[native_index,:2048])
        native.file.close();del native_values
        assert max(abs(len(ix)/known.sum()-ratio) for ix,ratio in [(train,.7),(validation,.2),(test,.1)])<=2/known.sum()
        for kind in ['gat_gine','state_mlp']:
            path=root/'output/within'/ds/kind
            model,ckpt=restore(path/'checkpoint.pt',a.device)
            assert ckpt['training_cell_ids']==cells.cell_id.iloc[train].tolist()
            assert ckpt['validation_cell_ids']==cells.cell_id.iloc[validation].tolist()
            assert not set(cells.cell_id.iloc[test])&set(ckpt['training_cell_ids']+ckpt['validation_cell_ids'])
            weights=len(train)/(2*np.bincount(cells.y.iloc[train],minlength=2))
            assert np.allclose(weights,ckpt['class_weights'])
            hist=pd.read_csv(path/'training_history.tsv',sep='\t');best=hist[hist.epoch.eq(ckpt['best_epoch'])].iloc[0]
            assert np.isclose(best.validation_loss,ckpt['validation_loss'])
            assert hist.validation_loss.min()>=ckpt['validation_loss']-ckpt['parameters']['min_delta']-1e-10
            pred=pd.read_csv(path/'predictions.tsv',sep='\t')
            assert pred.cell_id.tolist()==cells.cell_id.tolist()
            assert pred.y.tolist()==cells.y.tolist() and pred.split.tolist()==cells.split.tolist()
            assert pred.probability_resistant.between(0,1).all()
            assert np.allclose(pred.margin_resistant_minus_sensitive,pred.logit_resistant-pred.logit_sensitive,atol=1e-12)
            expected_probability=1/(1+np.exp(-np.clip(pred.margin_resistant_minus_sensitive.to_numpy(),-700,700)))
            assert np.allclose(expected_probability,pred.probability_resistant,atol=1e-14,rtol=1e-12)
            assert np.array_equal(pred.predicted,pred.probability_resistant.ge(.5).astype(int))
            saved=json.loads((path/'metrics.json').read_text())
            for split in ['train','validation','test']:check_metrics(pred[pred.split.eq(split)],saved[split])
            data,aligned,genes=load_cohort(root,ds,kind)
            assert aligned.cell_id.tolist()==cells.cell_id.tolist()
            if kind=='state_mlp':
                assert torch.allclose(model.feature_mean.cpu(),data[train].mean(0),rtol=1e-6,atol=1e-7)
                assert torch.allclose(model.feature_std.cpu(),data[train].std(0,unbiased=False).clamp_min(1e-6),rtol=1e-6,atol=1e-7)
            else:
                assert genes==ckpt['gene_symbols']
                values=np.concatenate([data[i].x[:,0].numpy() for i in train]).astype(np.float64)
                assert np.isclose(model.expr_mean.item(),values.mean(),rtol=1e-6)
                assert np.isclose(model.expr_std.item(),values.std(),rtol=1e-6)
                assert set(ckpt['training_observed_gene_ids'])==set(np.concatenate([data[i].gene_id.numpy() for i in train]))
            sample=np.concatenate([train[:3],validation[:3],test,np.flatnonzero(~known)[:1]])
            subset=data[sample] if kind=='state_mlp' else [data[i] for i in sample]
            fresh=predict(model,subset,cells.iloc[sample].reset_index(drop=True),kind,a.device,32)
            expected=pred.set_index('cell_id').loc[fresh.cell_id,'probability_resistant'].to_numpy()
            assert np.allclose(expected,fresh.probability_resistant,atol=2e-5,rtol=1e-5)
            checks.append(dict(dataset=ds,model=kind,n_cells=len(cells),n_checkpoint_predictions_verified=len(sample),
                n_fresh_test_predictions_verified=len(test),status='passed'))
            del model
            if ds!=source:
                transfer_path=root/'output/transfer'/f'{source}_to_{ds}'/kind
                t=pd.read_csv(transfer_path/'predictions.tsv',sep='\t');m=json.loads((transfer_path/'metrics.json').read_text())
                assert t.cell_id.tolist()==cells.cell_id.tolist() and m['source_dataset']==source
                assert m['source_scaler_unchanged'] and not m['target_training'] and not m['target_threshold_tuning']
                check_metrics(t[t.split.eq('test')],m['target_test']);check_metrics(t,m['target_all_labeled'])
                frozen,source_ckpt=restore(root/'output/within'/source/kind/'checkpoint.pt',a.device)
                assert not set(cells.cell_id)&set(source_ckpt['training_cell_ids']+source_ckpt['validation_cell_ids'])
                if kind=='gat_gine':assert genes==source_ckpt['gene_symbols']
                z=predict(frozen,subset,cells.iloc[sample].reset_index(drop=True),kind,a.device,32)
                expected=t.set_index('cell_id').loc[z.cell_id,'probability_resistant'].to_numpy()
                assert np.allclose(expected,z.probability_resistant,atol=2e-5,rtol=1e-5)
                del frozen
            del data
            torch.cuda.empty_cache()
            print('Validated',ds,kind,flush=True)
    result=dict(status='passed',within_models_checked=8,transfer_models_checked=6,
        checks=checks,source_dataset=source,
        scope='shared cohorts and labels, train-only scaling, checkpoint selection, all saved metrics, fresh local/transfer predictions on every test cell plus training/validation samples; not independent biological validation')
    (root/'output/summary').mkdir(parents=True,exist_ok=True)
    (root/'output/summary/validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('All paired-training and direct-transfer checks passed.',flush=True)


if __name__=='__main__':main()

"""Recompute frozen-transfer metrics and verify cell/fold/patient aggregation."""

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
from train_response_exp import metrics

MODELS=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']


def read(path):
    return pd.read_csv(path,sep='\t',keep_default_na=False,float_precision='round_trip')


def compare(actual,saved):
    for name,value in actual.items():
        previous=saved[name]
        if isinstance(value,(int,float)):
            if not np.isclose(value,previous,rtol=0,atol=1e-12): raise ValueError('Metric differs: '+name)
        elif value!=previous: raise ValueError('Metric field differs: '+name)


def check_frame(frame,cells):
    if not frame.cell_id.is_unique or frame.cell_id.tolist()!=cells.cell_id.tolist():
        raise ValueError('Prediction cell identity/order differs')
    for column in ['y','response','split']:
        if frame[column].astype(str).tolist()!=cells[column].astype(str).tolist():
            raise ValueError('Prediction metadata differs: '+column)
    if not np.isfinite(frame.probability_resistant.to_numpy(dtype=float)).all() or not frame.probability_resistant.between(0,1).all():
        raise ValueError('Invalid probability')
    if 'logit_resistant' in frame:
        if not np.allclose(frame.margin_resistant_minus_sensitive,frame.logit_resistant-frame.logit_sensitive,rtol=0,atol=1e-12):
            raise ValueError('Logit direction differs')
        if not frame.predicted.eq(frame.probability_resistant.ge(.5).astype(int)).all():
            raise ValueError('Classification threshold differs')


def check_summary(table,model,scope,values):
    row=table[table.model.eq(model)&table.scope.eq(scope)]
    if len(row)!=1: raise ValueError('Missing or duplicated summary scope')
    for name,value in values.items():
        if isinstance(value,(int,float)):
            if not np.isclose(float(row.iloc[0][name]),value,rtol=0,atol=1e-12):
                raise ValueError('Summary differs: '+name)


def alk(root):
    task='GSE223779_TS485T';cells=read(root/'data/cohorts'/task/'cells.tsv')
    summary=read(root/'output/summary/alk_frozen_transfer_metrics.tsv');records=[]
    bases={'state_15b':root/'output','native_state_embedding':root/'output',
           'no_edges':root/'output/transfer/edge_ablation/no_edges/shared',
           'random_signature':root/'output/transfer/signature_ablation/random_genes/state_15b'}
    for model in MODELS:
        base=bases.get(model,root/'output/transfer'/model)
        kind='state_mlp' if model=='native_state_embedding' else 'gat_gine'
        folder=base/'transfer'/('GSE223779_to_'+task)/kind
        frame=read(folder/'predictions.tsv');check_frame(frame,cells)
        saved=json.loads((folder/'metrics.json').read_text())
        if saved['target_training'] or saved['target_threshold_tuning']: raise ValueError('Target fitting enabled')
        for column in ['drug','patient_id','sample_id']:
            if frame[column].astype(str).tolist()!=cells[column].astype(str).tolist(): raise ValueError('Target stratum differs')
        for scope,drug in [('same_drug_crizotinib','crizotinib'),('cross_drug_alectinib','alectinib')]:
            part=frame[frame.drug.isin(['untreated',drug])];values=metrics(part)
            compare(values,saved['scopes'][scope]);check_summary(summary,model,scope,values)
            records.append(dict(dataset=task,model=model,scope=scope,n_predictions=len(part),n_evaluated=int(part.y.ge(0).sum()),aggregation_passed=True,metrics_passed=True))
    return records


def ovarian(root):
    task='GSE154600_EPI_external';source='GSE165897_pre_EOC'
    from mechanism_grouped_exp import METADATA_ROOT
    metadata=read(METADATA_ROOT/task/'counts_labels.tsv').set_index('Cell_ID')
    summary=read(root/'output/summary/ovarian_frozen_transfer_metrics.tsv');records=[]
    for model in MODELS:
        kind='state_mlp' if model=='native_state_embedding' else 'gat_gine'
        folder=root/'output/transfer/ovarian_clinical'/model/(source+'_to_'+task)/kind
        cells=read(folder/'ensemble_cell_predictions.tsv');folds=[]
        if not metadata.index.is_unique or set(cells.cell_id)!=set(metadata.index): raise ValueError('Clinical counts identities differ')
        expected_labels=cells.cell_id.map(metadata.label)
        if not cells.response.eq(expected_labels).all() or not cells.y.eq(expected_labels.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)).all():
            raise ValueError('Clinical original labels differ')
        for column in ['patient_id','sample_id','original_response']:
            if cells[column].astype(str).tolist()!=cells.cell_id.map(metadata[column]).astype(str).tolist():
                raise ValueError('Clinical original metadata differs: '+column)
        if cells.groupby('patient_id').y.nunique().gt(1).any(): raise ValueError('Patient labels inconsistent')
        for fold in range(1,5):
            frame=read(folder/f'fold_{fold}_cell_predictions.tsv');check_frame(frame,cells)
            if not frame.source_fold.eq(fold).all(): raise ValueError('Fold identity differs')
            folds.append(frame.probability_resistant.to_numpy())
        if not np.allclose(cells.probability_resistant,np.mean(folds,axis=0),rtol=0,atol=1e-12):
            raise ValueError('Fold probability averaging differs')
        patients=read(folder/'ensemble_patient_predictions.tsv').sort_values('patient_id').reset_index(drop=True)
        expected=cells.groupby('patient_id').agg(y=('y','first'),original_response=('original_response','first'),
            probability_resistant=('probability_resistant','mean'),n_cells=('cell_id','size')).reset_index().sort_values('patient_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(patients,expected,check_dtype=False,atol=1e-12,rtol=0)
        if len(patients)!=5 or int(patients.y.ge(0).sum())!=4: raise ValueError('Clinical evaluation scope differs')
        saved=json.loads((folder/'metrics.json').read_text())
        if saved['target_training'] or saved['target_threshold_tuning'] or saved['selected_k']!=768:
            raise ValueError('Frozen clinical policy differs')
        for scope,frame in [('external_epithelial_cells',cells),('external_patients',patients)]:
            values=metrics(frame.assign(margin_resistant_minus_sensitive=frame.probability_resistant))
            values['ranking_score']='mean_frozen_source_fold_probability_resistant'
            if scope=='external_patients':
                values['n_evaluation_units']=values.pop('n_cells');values['n_sensitive_patients']=values.pop('n_sensitive');values['n_resistant_patients']=values.pop('n_resistant')
            compare(values,saved['scopes'][scope]);check_summary(summary,model,scope,values)
            records.append(dict(dataset=task,model=model,scope=scope,n_predictions=len(frame),n_evaluated=int(frame.y.ge(0).sum()),aggregation_passed=True,metrics_passed=True))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    parser.add_argument('--save_path',required=True)
    args=parser.parse_args();root=Path(args.open_path)
    table=pd.DataFrame(alk(root)+ovarian(root))
    table.to_csv(args.save_path,sep='\t',index=False)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__': main()

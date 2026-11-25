"""Combine grouped out-of-fold predictions with existing frozen-transfer tables."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from train_response_exp import metrics,write_json


def grouped(root):
    records=[]
    directory=root/'output/grouped_adaptive'
    if not directory.exists(): return records
    for task_folder in sorted(directory.iterdir()):
        # Treatment-state endpoints use report_pc9_state_exp.py, not response-fate aggregation.
        if task_folder.name=='GSE149383_PC9_DTEP_sample_grouped': continue
        variants=sorted({p.name for fold in task_folder.glob('fold_*') for p in fold.iterdir() if p.is_dir()})
        for variant in variants:
            kind='state_mlp' if variant=='native_state_embedding' else 'gat_gine'
            paths=sorted(task_folder.glob(f'fold_*/{variant}/within/{task_folder.name}/{kind}/predictions.tsv'))
            source_info=root/'data/grouped'/task_folder.name/'fold_1/data/cohorts'/task_folder.name/'data_sources.json'
            if not source_info.exists(): continue
            design=json.loads(source_info.read_text())
            if len(paths)!=design['outer_folds']: continue
            frames=[]
            for path in paths:
                f=pd.read_csv(path,sep='\t',float_precision='round_trip');f=f[f.split.eq('test')].copy();f['fold']=path.parents[4].name
                frames.append(f)
            test=pd.concat(frames,ignore_index=True)
            if not test.cell_id.is_unique: raise ValueError('Overlapping outer-test cells')
            group_key=design['group_key']
            if (test.groupby(group_key).fold.nunique()!=1).any(): raise ValueError('Independent group crossed outer folds')
            out=root/'output/grouped_adaptive'/task_folder.name/'summary'/variant
            out.mkdir(parents=True,exist_ok=True)
            test.to_csv(out/'oof_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
            cell_metrics=metrics(test)
            cell_scope='patient_grouped_OOF_cells' if group_key=='patient_id' else 'replicate_grouped_OOF_cells'
            records.append(dict(dataset=task_folder.name,model=variant,scope=cell_scope,**{k:v for k,v in cell_metrics.items() if not isinstance(v,list)}))
            if group_key!='patient_id':
                replicate_metrics=[]
                for replicate,part in test.groupby(group_key,sort=True):
                    values=metrics(part)
                    replicate_metrics.append(values)
                    records.append(dict(dataset=task_folder.name,model=variant,scope='held_out_replicate_'+str(replicate),
                                        **{k:v for k,v in values.items() if not isinstance(v,list)}))
                metric_names=['auroc','auprc_resistant','auprc_sensitive','balanced_accuracy','mcc','macro_f1','brier_score']
                macro={name:float(np.mean([value[name] for value in replicate_metrics]))
                       if all(value.get(name) is not None for value in replicate_metrics) else None for name in metric_names}
                macro.update(n_replicates=len(replicate_metrics),unit='independent experiment replicate',
                    aggregation='equal arithmetic mean of held-out replicate metrics; descriptive with two replicates',threshold=.5)
                write_json(out/'replicate_macro_metrics.json',macro)
                records.append(dict(dataset=task_folder.name,model=variant,scope='replicate_macro',**macro))
                write_json(out/'replicate_metrics.json',{'scope':'two independent replicate outer tests',
                    'label_scope':design['label_scope'],'limitation':design['limitation'],
                    'outer_test_metrics':{str(k):metrics(f) for k,f in test.groupby(group_key)}})
                continue
            if (test.groupby('patient_id').y.nunique()!=1).any(): raise ValueError('Mixed patient outcome labels')
            patients=test.groupby('patient_id',sort=True).agg(y=('y','first'),
                probability_resistant=('probability_resistant','mean'),n_cells=('cell_id','size'),fold=('fold','first')).reset_index()
            patients.to_csv(out/'oof_patient_predictions.tsv',sep='\t',index=False,float_format='%.17g')
            # Ranking score for the clinical unit is the mean cell probability.
            evaluation=patients.assign(margin_resistant_minus_sensitive=patients.probability_resistant)
            result=metrics(evaluation);result['ranking_score']='mean_cell_probability_resistant'
            result['n_patients']=len(patients);result['unit']='patient'
            result['aggregation']='arithmetic mean of pretreatment malignant-cell probabilities per held-out patient'
            result['n_evaluation_units']=result.pop('n_cells')
            result['n_sensitive_patients']=result.pop('n_sensitive');result['n_resistant_patients']=result.pop('n_resistant')
            rng=np.random.default_rng(42);bootstrap=[]
            strata=[np.flatnonzero(patients.y.eq(y)) for y in [0,1]]
            for _ in range(1000):
                index=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strata])
                bootstrap.append(roc_auc_score(patients.y.iloc[index],patients.probability_resistant.iloc[index]))
            result['auroc_stratified_patient_bootstrap_95_interval']=np.quantile(bootstrap,[.025,.975]).tolist()
            result['bootstrap_scope']='conditional on saved outer-fold models; not full retraining or independent cohort uncertainty'
            write_json(out/'patient_metrics.json',result)
            records.append(dict(dataset=task_folder.name,model=variant,scope='patient_grouped_OOF_patients',
                                **{k:v for k,v in result.items() if not isinstance(v,(dict,list))}))
    return records


def combined(root,grouped_records):
    summary=root/'output/summary'
    old=pd.read_csv(summary/'four_graph_comparison_metrics.tsv',sep='\t')
    old=old[old.scope.eq('target_all_labeled')].copy()
    old['dataset']=old.target;old['model']=old.relation;old['scope']='frozen_transfer_all_labeled'
    edge=pd.read_csv(summary/'edge_ablation_metrics.tsv',sep='\t')
    edge=edge[edge.scope.eq('target_all_labeled')&edge.variant.eq('no_edges')].drop_duplicates('target').copy()
    edge['dataset']=edge.target;edge['model']='no_edges';edge['scope']='frozen_transfer_all_labeled'
    random=pd.read_csv(summary/'signature_ablation_metrics.tsv',sep='\t')
    random=random[random.scope.eq('target_all_labeled')&random.variant.eq('random_signature_state')].copy()
    random['dataset']=random.target;random['model']='random_signature';random['scope']='frozen_transfer_all_labeled'
    external=pd.read_csv(summary/'external_transfer_metrics.tsv',sep='\t')
    external=external[((external.dataset=='GSE202234')&(external.scope=='all_labeled'))|
                      ((external.dataset=='GSE163836')&(external.scope=='primary_FCIBC02_pair'))].copy()
    external['scope']='frozen_transfer_all_labeled_or_primary_pair'
    columns=['dataset','model','scope','n_cells','n_sensitive','n_resistant','auroc','auprc_resistant',
             'auprc_sensitive','resistant_prevalence','balanced_accuracy','mcc','macro_f1','brier_score','threshold','ranking_score']
    frames=[f.reindex(columns=columns) for f in [old,edge,random,external]]
    alk_path=summary/'alk_frozen_transfer_metrics.tsv'
    if alk_path.exists():
        alk=pd.read_csv(alk_path,sep='\t')
        alk['scope']='frozen_TS485T_'+alk.scope
        frames.append(alk.reindex(columns=columns))
    ovarian_path=summary/'ovarian_frozen_transfer_metrics.tsv'
    if ovarian_path.exists():
        ovarian=pd.read_csv(ovarian_path,sep='\t')
        ovarian['scope']='frozen_ovarian_'+ovarian.scope
        frames.append(ovarian)
    if grouped_records: frames.append(pd.DataFrame(grouped_records))
    table=pd.concat(frames,ignore_index=True)
    if table.duplicated(['dataset','model','scope']).any(): raise ValueError('Duplicate comparison rows')
    table.to_csv(summary/'mechanism_combined_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    print('COMBINED_ROWS',len(table),'GROUPED_ROWS',len(grouped_records),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    args=parser.parse_args();root=Path(args.open_path)
    combined(root,grouped(root))


if __name__=='__main__': main()

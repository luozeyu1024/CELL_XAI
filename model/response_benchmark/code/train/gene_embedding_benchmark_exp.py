"""Paired pretrained-gene-feature experiments using the existing cohorts and training loop."""
import os

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import uuid
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import train_response_exp as core
from gene_embedding_model import GeneEmbeddingClassifier, restore

ROOT=PROJECT_ROOT
BENCH=ROOT/'model/response_benchmark'
OUT=BENCH/'output/gene_embedding'
TABLES=Path(os.environ.get('CELL_XAI_GENE_FEATURES_ROOT', BENCH/'data/gene_embeddings')).resolve()
# Each graph uses the gene features from its own protein representation model.
VARIANTS={'state_graph__state_15b':('state_15b','state_15b'),
          'esm2_graph__esm2_650m':('esm2_650m','esm2_650m'),
          'qwen_graph__qwen':('qwen','qwen')}
DEFAULT_VARIANTS=['state_graph__state_15b']

from split_registry import apply_splits, load_cells, prostate_cell_lines, cohort_metadata_root


from split_registry import split_root

from split_registry import apply_splits

def jobs():
    cohort_root=Path(os.environ.get('CELL_XAI_BENCHMARK_ROOT', BENCH)).resolve()
    result=[]
    for dataset in ['GSE140440','GSE223779']:
        result.append(dict(dataset=dataset, cohort_root=str(cohort_root), family='within', fold='single'))
    from split_registry import source_folds
    for dataset_folder in sorted(split_root().iterdir()):
        dataset=dataset_folder.name
        if not dataset_folder.is_dir() or dataset in ['GSE140440','GSE223779'] or (dataset_folder/'infer.json').exists():continue
        for fold in source_folds(dataset):
            folder=cohort_root/'data/grouped'/dataset/fold
            result.append(dict(dataset=dataset,cohort_root=str(folder),family='grouped',fold=fold))
    return result


def destination(job, variant):
    if job['family']=='within': return OUT/variant
    return OUT/variant/'grouped'/job['dataset']/job['fold']


def preflight(datasets=None, include_targets=True):
    from cell_graph_dataset import CellGraphDataset
    records=[]
    gene_order=pd.read_csv(TABLES/'gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    unique={}
    selected_jobs=[job for job in jobs() if not datasets or job['dataset'] in datasets]
    if not selected_jobs: raise ValueError('No registered source designs selected')
    for job in selected_jobs:
        root=Path(job['cohort_root']);dataset=job['dataset']
        info=json.loads((root/'data/cohorts'/dataset/'data_sources.json').read_text())
        package=Path(info['graph_source']).parent
        unique[(str(package),dataset)]=package
    for dataset in (['GSE163836','GSE202234','GSE223779_TS485T','GSE154600_EPI_external'] if include_targets else []):
        file=BENCH/'data/cohorts'/dataset/'data_sources.json'
        if file.exists(): unique[(str(Path(json.loads(file.read_text())['graph_source']).parent),dataset)]=Path(json.loads(file.read_text())['graph_source']).parent
    ovarian_package=ROOT/'gene_sim_pathway/data/cell_signature_graphs_grouped/K768'
    if include_targets:
        unique[(str(ovarian_package),'GSE154600_EPI_external')]=ovarian_package
    for (package_str,dataset),package in unique.items():
        raw=CellGraphDataset(package,dataset,'state_15b')
        if raw.genes.tolist()!=gene_order: raise ValueError('Gene order differs: '+dataset)
        selected=np.unique(raw.gene_id)
        for model in ['esm2_650m','qwen','state_15b']:
            coverage=pd.read_csv(TABLES/f'{model}_coverage.tsv',sep='\t').mapped.to_numpy(bool)
            absent=selected[~coverage[selected]]
            records.append(dict(dataset=dataset,graph_package=package_str,model=model,
                                n_cells=len(raw),n_used_genes=len(selected),n_missing_used_genes=len(absent)))
            if len(absent): raise ValueError(f'{dataset}/{model}: missing used genes {np.array(gene_order)[absent].tolist()}')
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(OUT/'input_coverage.tsv',sep='\t',index=False)
    pd.DataFrame(selected_jobs).to_csv(OUT/'training_cohorts.tsv',sep='\t',index=False)
    print('PREFLIGHT',len(selected_jobs),'cohorts;',len(VARIANTS),'variants; all used genes mapped',flush=True)


@contextmanager
def trial_lock(folder):
    """Serialize identical Windows trials without adding result metadata files."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateMutexW.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.ReleaseMutex.argtypes=[wintypes.HANDLE]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    name='Local\\CellGeneEmbedding_'+uuid.uuid5(uuid.NAMESPACE_URL,str(folder.resolve()).lower()).hex
    handle=kernel.CreateMutexW(None,False,name)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    acquired=False
    try:
        result=kernel.WaitForSingleObject(handle,0xFFFFFFFF)
        if result not in [0,0x80]: raise ctypes.WinError(ctypes.get_last_error())
        acquired=True
        yield
    finally:
        if acquired: kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def train_worker(job,variant,device):
    load_cells(Path(job['cohort_root'])/'data/cohorts'/job['dataset']/'cells.tsv',job['dataset'],mode='source')
    folder=destination(job,variant)/'within'/job['dataset']/'gat_gine'
    with trial_lock(folder):
        complete=folder/'metrics.json'
        if complete.exists() and 'checkpoint_reload_max_margin_error' in json.loads(complete.read_text()):
            print('VERIFIED_RESULT_EXISTS',variant,job['dataset'],job['fold'],flush=True)
            return
        _train_worker(job,variant,device)


def _train_worker(job, variant, device):
    relation,embedding=VARIANTS[variant]
    table=TABLES/f'{embedding}.npy'
    core.GraphClassifier=partial(GeneEmbeddingClassifier,embedding_path=str(table))
    core.restore=restore
    core.output_root=lambda args,root:destination(job,variant)
    args=SimpleNamespace(seed=42,hidden=64,layers=2,heads=4,gene_dim=16,dropout=.2,
        lr=.001,weight_decay=.0001,graph_batch_size=32,mlp_batch_size=256,
        epochs=100,patience=15,min_delta=.0001,device=device,threads=4,relation=relation,
        variant=variant,embedding_model=embedding,embedding_path=str(table),embedding_trainable=False)
    torch.set_num_threads(4)
    core.train_one(args,Path(job['cohort_root']),job['dataset'],'gat_gine')
    folder=destination(job,variant)/'within'/job['dataset']/'gat_gine'
    cells=load_cells(Path(job['cohort_root'])/'data/cohorts'/job['dataset']/'cells.tsv',job['dataset'],mode='source')
    pred=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
    if pred.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Prediction cell order mismatch')
    for col in cells:
        if col not in pred: pred[col]=cells[col].to_numpy()
    pred.to_csv(folder/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
    model,saved=restore(folder/'checkpoint.pt',device)
    if model.identity.weight.requires_grad or 'identity.weight' in saved['state_dict']:
        raise ValueError('Frozen gene features were stored or made trainable')
    expected=cells.loc[cells.split.eq('train'),'cell_id'].tolist()
    if saved['training_cell_ids']!=expected: raise ValueError('Training split changed')
    data,actual,genes=core.load_cohort(Path(job['cohort_root']),job['dataset'],'gat_gine',relation)
    test_index=np.flatnonzero(actual.split.eq('test'))
    check_cells=actual.iloc[test_index].reset_index(drop=True)
    check=core.predict(model,[data[i] for i in test_index],check_cells,'gat_gine',device,32)
    previous=pred.set_index('cell_id').loc[check.cell_id]
    error=float(np.max(np.abs(check.margin_resistant_minus_sensitive.to_numpy()-previous.margin_resistant_minus_sensitive.to_numpy())))
    if error>1e-4: raise ValueError(f'Checkpoint reload mismatch: {error}')
    table_genes=pd.read_csv(TABLES/'gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    if genes!=table_genes: raise ValueError('Feature table order mismatch')
    record=json.loads((folder/'metrics.json').read_text())
    record.update(variant=variant,embedding_model=embedding,graph_relation=relation,
                  gene_feature_dimension=model.identity.weight.shape[1],embedding_trainable=False,
                  trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_gene_feature_values=model.identity.weight.numel(),
                  scope=job['family']+'; unchanged saved cell splits and K',
                  checkpoint_reload_max_margin_error=error)
    core.write_json(folder/'metrics.json',record)
    print('VERIFIED',variant,job['dataset'],job['fold'],error,flush=True)


def run_training(selected_variants,selected_datasets,device):
    for job in jobs():
        if selected_datasets and job['dataset'] not in selected_datasets: continue
        for variant in selected_variants:
            folder=destination(job,variant)/'within'/job['dataset']/'gat_gine'
            if (folder/'metrics.json').exists() and 'checkpoint_reload_max_margin_error' in json.loads((folder/'metrics.json').read_text()):
                print('VERIFIED_RESULT_EXISTS',variant,job['dataset'],job['fold'],flush=True);continue
            folder.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(Path(__file__).resolve()),'--stage','worker','--variant',variant,
                     '--job',json.dumps(job),'--device',device]
            print('START',variant,job['dataset'],job['fold'],flush=True)
            with (folder/'training.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',variant,job['dataset'],job['fold'],flush=True)


def attach_annotations(pred,cells):
    if pred.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Annotation order mismatch')
    for column in cells:
        if column not in pred: pred[column]=cells[column].to_numpy()
    return pred


def predict_external(variants,device,include_ovarian=True):
    """FA34 frozen transfer and the original four-fold ovarian ensemble."""
    torch.set_num_threads(4)
    for variant in variants:
        relation,_=VARIANTS[variant]
        source=OUT/variant/'within/GSE223779/gat_gine/checkpoint.pt'
        model,saved=restore(source,device)
        for target in ['GSE140440','GSE163836','GSE202234','GSE223779_TS485T']:
            dest=OUT/variant/'transfer'/('GSE223779_to_'+target)/'gat_gine'
            dest.mkdir(parents=True,exist_ok=True)
            if (dest/'metrics.json').exists(): continue
            data,cells,genes=core.load_cohort(BENCH,target,'gat_gine',relation)
            if genes!=saved['gene_symbols']: raise ValueError('Transfer vocabulary differs')
            if set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids']):
                raise ValueError('External target overlaps fitting cells')
            seen=np.zeros(len(genes),dtype=bool)
            seen[saved['training_observed_gene_ids']]=True
            occurrences=np.concatenate([graph.gene_id.numpy() for graph in data])
            feature_coverage=dict(n_target_unique_genes_unseen_in_source=int((~seen[np.unique(occurrences)]).sum()),
                fraction_target_node_occurrences_unseen_in_source_training=float((~seen[occurrences]).mean()),
                gene_feature_dimension=int(model.identity.weight.shape[1]))
            pred=attach_annotations(core.predict(model,data,cells,'gat_gine',device,32),cells)
            pred.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
            parts={'all_labeled':pred[pred.y.ge(0)]}
            if target=='GSE140440':
                pred['cell_line']=pred.cell_id.map(prostate_cell_lines())
                if pred.cell_line.isna().any(): raise ValueError('Unmapped GSE140440 cell line')
                parts.update({'cell_line_'+str(k):f for k,f in pred.groupby('cell_line')})
            elif target=='GSE163836':
                parts['primary_FCIBC02_pair']=apply_splits(pred,'GSE163836',mode='infer',scope='FCIBC02')
                parts['additional_SUM149PR']=apply_splits(pred,'GSE163836',mode='infer',scope='SUM149')
            elif target=='GSE223779_TS485T':
                parts['same_drug_crizotinib']=apply_splits(pred,'GSE223779_TS485T',mode='infer',scope='crizotinib')
                parts['cross_drug_alectinib']=apply_splits(pred,'GSE223779_TS485T',mode='infer',scope='alectinib')
            elif target=='GSE202234':
                parts.update({'treatment_'+str(k):f[f.y.ge(0)] for k,f in pred.groupby('treatment')})
            scores={}
            for scope,frame in parts.items():
                values=core.metrics(frame)
                if frame.loc[frame.y.ge(0),'y'].nunique()<2:
                    for key in ['balanced_accuracy','mcc','macro_f1']: values[key]=None
                scores[scope]=values
            pred.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
            core.write_json(dest/'metrics.json',dict(source='GSE223779',target=target,variant=variant,
                checkpoint=str(source),target_training=False,target_threshold_tuning=False,scopes=scores,**feature_coverage))
            print('TRANSFER',variant,target,flush=True)
            del data;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
        if not include_ovarian: continue
        target='GSE154600_EPI_external';source_task='GSE165897_pre_EOC'
        dest=OUT/variant/'transfer'/f'{source_task}_to_{target}'/'gat_gine'
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists(): continue
        from cell_graph_dataset import CellGraphDataset
        package=ROOT/'gene_sim_pathway/data/cell_signature_graphs_grouped/K768'
        raw=CellGraphDataset(package,target,relation)
        cells=raw.cells.copy()
        metadata=pd.read_csv(cohort_metadata_root()/target/'counts_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
        for col in metadata: cells[col]=cells.cell_id.map(metadata[col])
        cells['response']=cells.label
        cells['y']=cells.label.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)
        cells['split']='external'
        cells=apply_splits(cells,target,mode='infer')
        data=[]
        for i,row in enumerate(cells.itertuples()):
            graph=raw.to_pyg(i);graph.x=graph.x[:,:1].clone();graph.y=torch.tensor([row.y]);data.append(graph)
        probabilities=[]
        for fold in range(1,5):
            path=OUT/variant/'grouped'/source_task/f'fold_{fold}'/'within'/source_task/'gat_gine/checkpoint.pt'
            model,saved=restore(path,device)
            if raw.genes.tolist()!=saved['gene_symbols']: raise ValueError('Ovarian gene order differs')
            if set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids']): raise ValueError('Ovarian target leakage')
            frame=attach_annotations(core.predict(model,data,cells,'gat_gine',device,32),cells)
            frame.to_csv(dest/f'fold_{fold}_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
            probabilities.append(frame.probability_resistant.to_numpy())
            del model;gc.collect();torch.cuda.empty_cache()
        ensemble=cells.copy();ensemble['probability_resistant']=np.mean(probabilities,axis=0)
        ensemble.to_csv(dest/'ensemble_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
        patients=ensemble.groupby('patient_id').agg(y=('y','first'),probability_resistant=('probability_resistant','mean'),n_cells=('cell_id','size')).reset_index()
        patients.to_csv(dest/'ensemble_patient_predictions.tsv',sep='\t',index=False,float_format='%.17g')
        scopes={}
        for scope,frame in [('external_epithelial_cells',ensemble),('external_patients',patients)]:
            score=core.metrics(frame.assign(margin_resistant_minus_sensitive=frame.probability_resistant))
            score['ranking_score']='mean_frozen_source_fold_probability_resistant'
            score['unit']='patient' if scope=='external_patients' else 'cell'
            if scope=='external_patients':
                score['n_evaluation_units']=score.pop('n_cells')
                score['n_sensitive_patients']=score.pop('n_sensitive')
                score['n_resistant_patients']=score.pop('n_resistant')
            scopes[scope]=score
        core.write_json(dest/'metrics.json',dict(source=source_task,target=target,variant=variant,
            target_training=False,target_threshold_tuning=False,scopes=scopes))
        del data;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','train','worker','predict','all'],default='all')
    parser.add_argument('--variant',nargs='+',choices=list(VARIANTS),default=DEFAULT_VARIANTS)
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--job')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--prediction_scope',choices=['all','fa34'],default='all')
    args=parser.parse_args()
    if args.stage=='worker': train_worker(json.loads(args.job),args.variant[0],args.device);return
    if args.stage in ['preflight','train','predict','all']:
        preflight(args.datasets, include_targets=args.stage in ['predict','all'])
    if args.stage in ['train','all']: run_training(args.variant,args.datasets,args.device)
    if args.stage in ['predict','all']: predict_external(args.variant,args.device,include_ovarian=args.prediction_scope=='all')
    if args.stage=='all' and not args.datasets and args.prediction_scope=='all':
        subprocess.run([sys.executable,str(script_path('report_gene_embedding_exp.py')),
                        '--variant',*args.variant],check=True)
        subprocess.run([os.environ.get('CELL_XAI_PLOT_PYTHON', sys.executable),
            str(script_path('plot_gene_embedding_exp.py')),
            '--open_path',str(ROOT/'work/transfer_analysis/data/gene_embedding'),
            '--save_path',str(ROOT/'work/transfer_analysis/figures'),
            '--variant',*args.variant],check=True)


if __name__=='__main__': main()

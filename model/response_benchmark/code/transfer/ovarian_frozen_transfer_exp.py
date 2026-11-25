"""External chemotherapy-response stress test using frozen patient-grouped ovarian models."""

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
import numpy as np
import pandas as pd
import mechanism_grouped_exp as prep

ROOT=prep.ROOT
BENCH=prep.BENCH
SOURCE='GSE165897_pre_EOC'
TASK='GSE154600_EPI_external'
MODELS=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']

from split_registry import apply_splits


def counts_stage(root):
    source = root / TASK / 'counts.h5ad'
    if not source.is_file():
        raise FileNotFoundError('Prepare the registered target with the cohort module: ' + str(source))
    return source


def source_k():
    choices={prep.selected_k(SOURCE,fold) for fold in range(1,5)}
    if choices!={768}: raise ValueError('Source fold K policies differ; per-fold target graph packages required')
    return 768


def graphs_stage():
    from types import SimpleNamespace
    build=prep.module('ovarian_external_graph_builder',ROOT/'gene_sim_pathway/code/emb_KNN_graph_G/build_cell_signature_graphs_exp.py')
    utility=prep.module('ovarian_external_utils',ROOT/'assets/SIGnature/src/SIGnature/utils.py')
    package=prep.graph_package(source_k())
    genes=pd.read_csv(package/'gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    with np.load(package/'global_graphs/coverage.npz') as z: masks={key:z[key] for key in z.files}
    mapper=build.GeneMapper(ROOT/'assets/hgnc_complete_set.txt')
    if not (package/TASK/'signatures.npz').exists():
        build.run_dataset(ROOT/'work/attributions'/TASK,
                          SimpleNamespace(save_path=str(package),dataset_top_k=source_k()),mapper,genes,masks,utility)


def random_stage():
    from types import SimpleNamespace
    import shutil
    from scipy import sparse
    reference=prep.graph_package(source_k());output=prep.graph_package(source_k(),True)
    plans=prep.PLAN_ROOT/'grouped'/f'K{source_k()}'
    plans.mkdir(parents=True,exist_ok=True)
    for name in ['state_15b.npz','gene_vocab.tsv']:
        source=reference/('global_graphs/'+name if name.endswith('.npz') else name)
        if not (plans/name).exists(): shutil.copy2(source,plans/name)
    build=prep.module('ovarian_random_builder',ROOT/'gene_sim_pathway/code/state_signature_ablation/build_random_state_signatures_exp.py')
    args=SimpleNamespace(reference_path=str(reference),save_path=str(output),graph_path=str(plans),benchmark_path=str(BENCH),
        signature_utils=str(ROOT/'assets/SIGnature/src/SIGnature/utils.py'),seed=42,count_jitter=.05,
        density_tolerance=.02,degree_cdf_tolerance=.05,isolation_tolerance=.02,max_steps=2000,restarts=12,pilot_cells=0,rebuild=False)
    build.build_dataset(args,TASK,sparse.load_npz(reference/'global_graphs/state_15b.npz').tocsr(),pd.read_csv(reference/'gene_vocab.tsv',sep='\t'))
    qc=pd.read_csv(output/TASK/'matching_qc.tsv',sep='\t')
    if len(qc)!=7793 or not qc.matching_passed.all(): raise ValueError('Unmatched random signatures')


def prediction_stage():
    import torch
    import train_response_exp as core
    from cell_graph_dataset import CellGraphDataset
    from edge_null_graphs import remove_edges
    torch.set_num_threads(4);core.seed_all(42)
    metadata=pd.read_csv(prep.METADATA_ROOT/TASK/'counts_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
    se=prep.SE_ROOT/TASK
    state_ids=pd.Index(pd.read_csv(se/'cell_metadata.tsv',sep='\t').cell_id)
    cells=pd.read_csv(prep.graph_package(source_k())/TASK/'cells.tsv',sep='\t',keep_default_na=False)
    if set(cells.cell_id)!=set(metadata.index): raise ValueError('Counts/graph identity mismatch')
    for col in metadata.columns: cells[col]=cells.cell_id.map(metadata[col])
    cells['response']=cells.label;cells['y']=cells.label.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)
    cells['split']='external';cells['graph_row']=np.arange(len(cells))
    cells=apply_splits(cells,TASK,mode='infer')
    index=state_ids.get_indexer(cells.cell_id)
    if (index<0).any() or not state_ids.is_unique: raise ValueError('Native SE order mismatch')
    x=np.load(se/'cell_embeddings.npy')[index,:2048]
    if x.shape!=(len(cells),2048) or not np.isfinite(x).all(): raise ValueError('Invalid native SE values')
    rows=[]
    for variant in MODELS:
        kind='state_mlp' if variant=='native_state_embedding' else 'gat_gine'
        if kind=='state_mlp': data=torch.from_numpy(x.astype(np.float32));genes=None
        else:
            package=prep.graph_package(source_k(),variant=='random_signature')
            relation=variant if variant in ['esm2_650m','qwen','ppi'] else 'state_15b'
            raw=CellGraphDataset(package,TASK,relation);genes=raw.genes.tolist()
            if raw.cells.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Graph row mismatch')
            data=[]
            for i,row in enumerate(cells.itertuples()):
                graph=raw.to_pyg(i);graph.x=graph.x[:,:1].clone();graph.y=torch.tensor([row.y])
                if variant=='no_edges': remove_edges(graph)
                data.append(graph)
        destination=BENCH/'output/transfer/ovarian_clinical'/variant/(SOURCE+'_to_'+TASK)/kind
        destination.mkdir(parents=True,exist_ok=True);probabilities=[]
        for fold in range(1,5):
            checkpoint=BENCH/'output/grouped_adaptive'/SOURCE/f'fold_{fold}'/variant/'within'/SOURCE/kind/'checkpoint.pt'
            model,saved=core.restore(checkpoint,'cuda')
            if genes is not None and genes!=saved['gene_symbols']: raise ValueError('Frozen source gene identity mismatch')
            if set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids']): raise ValueError('External cells in source fitting')
            prediction=core.predict(model,data,cells,kind,'cuda',32 if kind=='gat_gine' else 256)
            prediction['source_fold']=fold
            prediction.to_csv(destination/f'fold_{fold}_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
            probabilities.append(prediction.probability_resistant.to_numpy())
            del model;gc.collect();torch.cuda.empty_cache()
        ensemble=cells[['cell_id','patient_id','sample_id','response','original_response','y','split']].copy()
        ensemble['probability_resistant']=np.mean(probabilities,axis=0)
        ensemble.to_csv(destination/'ensemble_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
        patients=ensemble.groupby('patient_id').agg(y=('y','first'),original_response=('original_response','first'),
            probability_resistant=('probability_resistant','mean'),n_cells=('cell_id','size')).reset_index()
        patients.to_csv(destination/'ensemble_patient_predictions.tsv',sep='\t',index=False,float_format='%.17g')
        results={}
        for scope,frame in [('external_epithelial_cells',ensemble),('external_patients',patients)]:
            values=core.metrics(frame.assign(margin_resistant_minus_sensitive=frame.probability_resistant))
            values['ranking_score']='mean_frozen_source_fold_probability_resistant'
            if scope=='external_patients':
                values['n_evaluation_units']=values.pop('n_cells');values['n_sensitive_patients']=values.pop('n_sensitive');values['n_resistant_patients']=values.pop('n_resistant')
            results[scope]=values
            rows.append(dict(dataset=TASK,model=variant,scope=scope,**{k:v for k,v in values.items() if not isinstance(v,list)}))
        core.write_json(destination/'metrics.json',{'source':SOURCE,'target':TASK,'source_models':'four frozen outer-fold models, equal probability ensemble',
            'target_training':False,'target_threshold_tuning':False,'selected_k':source_k(),'scopes':results,
            'limitation':'four labeled patients and one refractory unknown; GEO chemotherapy response differs from source PFI; regimen/timepoint equivalence unconfirmed; author epithelial is not independent malignant CNV annotation'})
        print('OVARIAN_FROZEN',variant,{k:v['auroc'] for k,v in results.items()},flush=True)
        del data;gc.collect()
    pd.DataFrame(rows).to_csv(BENCH/'output/summary/ovarian_frozen_transfer_metrics.tsv',sep='\t',index=False,float_format='%.17g')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    parser.add_argument('--stage',choices=['counts','attribution','prepare','graphs','random','predict','all'],default='all')
    parser.add_argument('--wait_pid',type=int,default=0)
    args=parser.parse_args();root=Path(args.open_path)
    if args.wait_pid:
        import psutil
        try:
            process=psutil.Process(args.wait_pid)
            if not any(str(x).endswith('alk_frozen_transfer_exp.py') for x in process.cmdline()): raise ValueError('Unexpected wait process')
            print('WAIT_ALK_TRANSFER',args.wait_pid,flush=True);process.wait()
        except psutil.NoSuchProcess: pass
    if args.stage=='counts': counts_stage(root);return
    if args.stage=='attribution':
        counts_stage(root);prep.attribution_stage(root,TASK);prep.index_stage(root,TASK);return
    if args.stage in ['prepare','all']:
        counts_stage(root);prep.attribution_stage(root,TASK);prep.index_stage(root,TASK);prep.state_stage(root,TASK)
        if args.stage=='prepare': return
    if args.stage=='graphs': graphs_stage();return
    if args.stage=='all':
        subprocess.run([prep.PYTHON['attribution'],str(Path(__file__)), '--open_path',str(root),'--stage','graphs'],check=True)
    if args.stage in ['random','all']: random_stage()
    if args.stage in ['predict','all']: prediction_stage()


if __name__=='__main__': main()

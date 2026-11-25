"""Grouped validation of drug-specific tasks with the existing graph/SE models."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=PROJECT_ROOT
BENCH=ROOT/'model/response_benchmark'
GRAPH_ROOT=Path(os.environ.get('CELL_XAI_GRAPH_ROOT', ROOT/'work/cell_graphs'))
RANDOM_ROOT=ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation'
PLAN_ROOT=ROOT/'gene_sim_pathway/data/emb_KNN_graph/state_ablation'
SE_ROOT=BENCH/'data/state_embeddings'
INDEX_ROOT=ROOT/'gene_sim_pathway/data/cell_signature_candidate_index'
POLICY_ROOT=ROOT/'gene_sim_pathway/output/signature_gene_selection_grouped'
METADATA_ROOT=Path(os.environ.get('CELL_XAI_COHORT_METADATA', ROOT/'work/cohort_metadata'))
PYTHON={'counts':sys.executable,
        'attribution':os.environ.get('CELL_XAI_ATTR_PYTHON', sys.executable),
        'state':os.environ.get('CELL_XAI_SE_PYTHON', sys.executable)}
TASKS=['GSE165897_pre_EOC','GSE223003_carboplatin_pre','GSE223003_olaparib_pre','GSE223003_nk_pre']
for key in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ.setdefault(key,'4')
for name in ['numba','tmp']:
    (ROOT/'signature_attr_run/.cache'/name).mkdir(parents=True,exist_ok=True)
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'signature_attr_run/.cache/numba')
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'signature_attr_run/.cache/tmp')

from split_registry import load_cells, split_root


def module(name,path):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def primary_counts(data_root, task):
    source = data_root / task / 'counts.h5ad'
    if not source.is_file():
        raise FileNotFoundError('Prepare the registered task with the cohort module: ' + str(source))
    return source


def attribution_stage(data_root,task):
    counts=primary_counts(data_root,task)
    out=ROOT/'work/attributions'/task
    if not (out/'manifests/scimilarity_ig_manifest.json').exists():
        subprocess.run([PYTHON['attribution'],str(ROOT/'signature_attr_run/code/run_attribution.py'),
            '--model','scimilarity','--h5ad',str(counts),'--out-dir',str(out),
            '--model-dir',str(ROOT/'assets/scimilarity'),
            '--method','ig','--batch-size','100','--device','cuda','--counts-policy','auto'],check=True)


def index_stage(data_root,task):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    folder=INDEX_ROOT/task
    folder.mkdir(parents=True,exist_ok=True)
    if not (INDEX_ROOT/'gene_vocab.tsv').exists(): shutil.copy2(GRAPH_ROOT/'gene_vocab.tsv',INDEX_ROOT/'gene_vocab.tsv')
    if (folder/'eligible_genes.tsv').exists(): return
    build=module('mechanism_graph_builder',ROOT/'gene_sim_pathway/code/emb_KNN_graph_G/build_cell_signature_graphs_exp.py')
    attribution=ROOT/'work/attributions'/task
    manifest=json.loads((attribution/'manifests/scimilarity_ig_manifest.json').read_text())['input']
    data=ad.read_h5ad(manifest['h5ad'],backed='r')
    if data.obs_names.astype(str).tolist()!=manifest['cell_ids_used']: raise ValueError('Full-cell attribution order mismatch')
    names=json.loads((attribution/'attributions/scimilarity_ig_var_names.json').read_text())
    mapper=build.GeneMapper(ROOT/'assets/hgnc_complete_set.txt')
    raw_map,_=build.unique_canonical(data.var_names,mapper);model_map,_=build.unique_canonical(names,mapper)
    genes=pd.read_csv(GRAPH_ROOT/'gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    with np.load(GRAPH_ROOT/'global_graphs/coverage.npz') as z:
        common=np.logical_and.reduce([z[r] for r in ['esm2_650m','qwen','state_15b','ppi']])
    records=[{'gene_id':i,'gene_symbol':g,'attribution_column':model_map[g]} for i,g in enumerate(genes)
             if common[i] and g in raw_map and g in model_map and names[model_map[g]] in data.var_names]
    mapping=pd.DataFrame(records);mapping.to_csv(folder/'eligible_genes.tsv',sep='\t',index=False)
    full=sparse.load_npz(attribution/'attributions/scimilarity_ig.npz').tocsr()
    positive=np.asarray((full[:,mapping.attribution_column.to_numpy(int)]>0).sum(axis=1)).ravel()
    pd.DataFrame({'cell_id':manifest['cell_ids_used'],'source_row':manifest['cell_indices_used'],
                  'y':-1,'label_status':'not_assigned_in_candidate_index','has_signature':positive>0}).to_csv(folder/'cells.tsv',sep='\t',index=False)
    data.file.close()


def selected_k(task,fold):
    path=POLICY_ROOT/task/f'fold_{fold}'/'dataset_policy.json'
    if not path.exists(): raise ValueError(f'Review training coverage and write a dataset-specific policy first: {path}')
    value=json.loads(path.read_text())['datasets'][task]
    if not value.get('rationale') or not 1<=int(value['k'])<=4096: raise ValueError('Invalid reviewed K policy')
    return int(value['k'])


def graph_package(k,random=False):
    return ROOT/'gene_sim_pathway/data'/('cell_signature_graphs_grouped_random' if random else 'cell_signature_graphs_grouped')/f'K{k}'


def profile_stage(data_root,task):
    from profile_mechanism_k_exp import profile
    profile(task,INDEX_ROOT,POLICY_ROOT/task)


def graphs_stage(data_root,task):
    import numpy as np
    import pandas as pd
    build=module('mechanism_graph_builder',ROOT/'gene_sim_pathway/code/emb_KNN_graph_G/build_cell_signature_graphs_exp.py')
    utility=module('mechanism_signature_utils',ROOT/'assets/SIGnature/src/SIGnature/utils.py')
    genes=pd.read_csv(GRAPH_ROOT/'gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    with np.load(GRAPH_ROOT/'global_graphs/coverage.npz') as z: masks={k:z[k] for k in z.files}
    mapper=build.GeneMapper(ROOT/'assets/hgnc_complete_set.txt')
    from split_registry import source_folds
    folds=[BENCH/'data/grouped'/task/fold for fold in source_folds(task)]
    for fold in folds:
        k=selected_k(task,int(fold.name.split('_')[1]));package=graph_package(k)
        (package/'global_graphs').mkdir(parents=True,exist_ok=True)
        for relative in ['gene_vocab.tsv']+['global_graphs/'+n for n in ['coverage.npz','esm2_650m.npz','qwen.npz','state_15b.npz','ppi.npz']]:
            if not (package/relative).exists(): shutil.copy2(GRAPH_ROOT/relative,package/relative)
        if not (package/task/'signatures.npz').exists():
            build.run_dataset(ROOT/'work/attributions'/task,
                              SimpleNamespace(save_path=str(package),dataset_top_k=k),mapper,genes,masks,utility)
        path=fold/'data/cohorts'/task/'data_sources.json';info=json.loads(path.read_text())
        info.update(graph_source=str(package/task),candidate_index_source=str(INDEX_ROOT/task),adaptive_graph_source=str(package/task),selected_k=k,
                    k_policy=str(POLICY_ROOT/task/fold.name/'dataset_policy.json'),k_calibration='outer training cells only')
        path.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')


def state_stage(data_root,task):
    out=SE_ROOT/task
    if not (out/'validation.json').exists():
        subprocess.run([PYTHON['state'],str(ROOT/'env/state/test/run_se_test.py'),
            '--open_path',str(data_root/'analysis_counts'/task/'counts.h5ad'),
            '--model_path',str(ROOT/'assets/SE-600M'),'--save_path',str(out),
            '--counts_save_path',str(data_root/'analysis_counts'/task/'se_input_counts.h5ad'),
            '--state_h5ad_path',str(data_root/'analysis_counts'/task/'se_encoded_counts.h5ad'),
            '--batch_size','8','--max_cells','0','--seed','42'],check=True)


def random_stage(data_root,task):
    from split_registry import source_folds
    folds=[BENCH/'data/grouped'/task/fold for fold in source_folds(task)]
    for k in sorted({selected_k(task,int(f.name.split('_')[1])) for f in folds}):
        random_for_k(data_root,task,k)


def random_for_k(data_root,task,k):
    import pandas as pd
    from scipy import sparse
    reference=graph_package(k)
    output=graph_package(k,random=True)
    if (output/task/'signatures.npz').exists(): return
    (output/'global_graphs').mkdir(parents=True,exist_ok=True)
    for name in ['gene_vocab.tsv','global_graphs/state_15b.npz']:
        if not (output/name).exists(): shutil.copy2(reference/name,output/name)
    graph_path=PLAN_ROOT/'grouped'/f'K{k}'
    graph_path.mkdir(parents=True,exist_ok=True)
    for src,name in [(reference/'gene_vocab.tsv','gene_vocab.tsv'),(reference/'global_graphs/state_15b.npz','state_15b.npz')]:
        if not (graph_path/name).exists(): shutil.copy2(src,graph_path/name)
    build=module('mechanism_random_builder',ROOT/'gene_sim_pathway/code/state_signature_ablation/build_random_state_signatures_exp.py')
    args=SimpleNamespace(reference_path=str(reference),save_path=str(output),graph_path=str(graph_path),
        benchmark_path=str(BENCH),signature_utils=str(ROOT/'assets/SIGnature/src/SIGnature/utils.py'),
        seed=42,count_jitter=.05,density_tolerance=.02,degree_cdf_tolerance=.05,isolation_tolerance=.02,
        max_steps=2000,restarts=12,pilot_cells=0,rebuild=False)
    build.build_dataset(args,task,sparse.load_npz(reference/'global_graphs/state_15b.npz').tocsr(),
                        pd.read_csv(reference/'gene_vocab.tsv',sep='\t'))
    qc=pd.read_csv(output/task/'matching_qc.tsv',sep='\t')
    if not qc.matching_passed.all(): raise ValueError('Incomplete random graph matching')


def cohorts_stage(data_root,task,include_embeddings=True):
    import anndata as ad
    import numpy as np
    import pandas as pd
    package=INDEX_ROOT/task
    labels_path=METADATA_ROOT/task/'counts_labels.tsv'
    labels=pd.read_csv(labels_path,sep='\t',keep_default_na=False).set_index('Cell_ID')
    cells=pd.read_csv(package/'cells.tsv',sep='\t')
    cells['graph_row']=np.arange(len(cells))
    for key in labels.columns: cells[key]=cells.cell_id.map(labels[key])
    cells['y']=cells.label.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)
    cells['response']=cells.label
    assert cells.y.ge(0).all() and cells.has_signature.all()
    x=None
    if include_embeddings:
        state_cells=pd.read_csv(SE_ROOT/task/'cell_metadata.tsv',sep='\t').cell_id.astype(str)
        state_index=pd.Index(state_cells).get_indexer(cells.cell_id)
        assert (state_index>=0).all() and state_cells.is_unique
        x=np.asarray(np.load(SE_ROOT/task/'cell_embeddings.npy'))[state_index,:2048].astype(np.float32)
        assert x.shape==(len(cells),2048) and np.isfinite(x).all()
    from split_registry import source_folds, apply_splits
    for fold_name in source_folds(task):
        frame=apply_splits(cells,task,fold_name,mode='source')
        fold=int(fold_name.split('_')[1])
        folder=BENCH/'data/grouped'/task/fold_name/'data/cohorts'/task
        folder.mkdir(parents=True,exist_ok=True)
        frame.to_csv(folder/'cells.tsv',sep='\t',index=False)
        if x is not None:np.save(folder/'state_cell_embeddings.npy',x)
        group_key='biological_replicate' if task.startswith('GSE223003_') else 'patient_id'
        group_sets={role:set(frame.loc[frame.split.eq(role),group_key].astype(str)) for role in ['train','validation','test']}
        if group_sets['test'] & (group_sets['train'] | group_sets['validation']):raise ValueError('Outer group leakage')
        if group_key=='patient_id' and group_sets['train'] & group_sets['validation']:raise ValueError('Patient validation leakage')
        info=dict(graph_source=str(package),task=task,group_key=group_key,fold=fold,
                  split_source=str(split_root()/task/fold_name),
                  n_cells=len(cells),seed=42)
        (folder/'data_sources.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        print('JSON_SPLIT',task,fold_name,frame.groupby(['split','y']).size().to_dict(),flush=True)


def train_stage(data_root,task,models):
    import numpy as np
    import pandas as pd
    import torch
    from cell_graph_dataset import CellGraphDataset
    base=module('mechanism_training',script_path('train_response_exp.py'))
    torch.set_num_threads(4)
    from split_registry import source_folds
    folds=[BENCH/'data/grouped'/task/fold for fold in source_folds(task)]
    if not folds: raise ValueError('No validated grouped folds')
    for fold_folder in folds:
        fold=int(fold_folder.name.split('_')[1])
        k=selected_k(task,fold)
        working=BENCH/'data/grouped'/task/f'fold_{fold}'
        source_path=working/'data/cohorts'/task/'data_sources.json'
        source_info=json.loads(source_path.read_text())
        source_info.update(graph_source=str(graph_package(k)/task),candidate_index_source=str(INDEX_ROOT/task),
                           adaptive_graph_source=str(graph_package(k)/task),selected_k=k,
                           k_policy=str(POLICY_ROOT/task/f'fold_{fold}'/'dataset_policy.json'),k_calibration='outer training cells only')
        source_path.write_text(json.dumps(source_info,indent=2)+'\n',encoding='utf-8')
        # Keep the established architecture/optimizer/checkpoint/prediction schema.
        for variant in models:
            relation=variant if variant in ['state_15b','esm2_650m','qwen','ppi'] else 'state_15b'
            kind='state_mlp' if variant=='native_state_embedding' else 'gat_gine'
            original_load=base.load_cohort
            def load(root,ds,model_kind,ignored_relation='state_15b'):
                if model_kind=='state_mlp': return original_load(root,ds,model_kind,relation)
                cells=load_cells(root/'data/cohorts'/ds/'cells.tsv',ds,mode='source')
                raw=CellGraphDataset(graph_package(k,random=variant=='random_signature'),ds,relation)
                result=[]
                for row in cells.itertuples():
                    graph=raw.to_pyg(int(row.graph_row));assert graph.cell_id==row.cell_id
                    graph.x=graph.x[:,:1].clone();graph.y=torch.tensor([row.y],dtype=torch.long)
                    if variant=='no_edges':
                        graph.edge_index=torch.empty((2,0),dtype=torch.long);graph.edge_attr=torch.empty((0,1))
                    result.append(graph)
                return result,cells,raw.genes.tolist()
            base.load_cohort=load
            # Preserve the established /within/<task>/<kind> output schema.
            base.output_root=lambda args,root:BENCH/'output/grouped_adaptive'/task/f'fold_{fold}'/variant
            args=SimpleNamespace(seed=42,hidden=64,layers=2,heads=4,gene_dim=16,dropout=.2,
                lr=.001,weight_decay=.0001,graph_batch_size=32,mlp_batch_size=256,
                epochs=100,patience=15,min_delta=.0001,device='cuda',threads=4,relation=relation,variant=variant,
                selected_k=k,k_policy=str(POLICY_ROOT/task/f'fold_{fold}'/'dataset_policy.json'))
            existing=base.output_root(args,working)/'within'/task/kind/'run_parameters.json'
            if existing.exists() and json.loads(existing.read_text()).get('selected_k')!=k:
                raise ValueError('Completed output uses a different K; preserve it and select a separate experiment output')
            base.train_one(args,working,task,kind)
            actual=base.output_root(args,working)/'within'/task/kind
            record=json.loads((actual/'metrics.json').read_text())
            record['scope']='patient-grouped pretreatment clinical outcome' if task=='GSE165897_pre_EOC' else 'independent replicate outer validation of author-inferred pretreatment fate; source-sensitive-lineage grouping incomplete'
            base.write_json(actual/'metrics.json',record)
            prediction=pd.read_csv(actual/'predictions.tsv',sep='\t')
            annotations=pd.read_csv(working/'data/cohorts'/task/'cells.tsv',sep='\t').set_index('cell_id')
            for column in ['patient_id','biological_replicate','drug','label_scope','prer_r_group']:
                if column in annotations: prediction[column]=prediction.cell_id.map(annotations[column])
            prediction.to_csv(actual/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
            base.load_cohort=original_load
            gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True,help='Prepared counts root containing TASK/counts.h5ad')
    parser.add_argument('--task',choices=TASKS,default='GSE165897_pre_EOC')
    parser.add_argument('--wait_pid',type=int,default=0,help='Wait for a known counts conversion process before checking its completed output')
    parser.add_argument('--wait_training_pid',type=int,default=0,help='Wait for an identified grouped training process before starting this task')
    parser.add_argument('--stage',choices=['attribution','index','profile','split_profile','graphs','state','cohorts','random','train','prepare','all'],default='all')
    parser.add_argument('--models',nargs='+',choices=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding'],
                        default=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding'])
    args=parser.parse_args();data_root=Path(args.open_path)
    if args.wait_training_pid:
        import psutil
        try:
            process=psutil.Process(args.wait_training_pid)
            command=process.cmdline()
            if not any(str(x).endswith('mechanism_grouped_exp.py') for x in command) or '--stage' not in command or command[command.index('--stage')+1]!='train':
                raise ValueError('Wait PID is not an identified grouped training process')
            print('WAIT_GROUPED_TRAINING',args.wait_training_pid,flush=True);process.wait()
        except psutil.NoSuchProcess: pass
    if args.wait_pid:
        import psutil
        try:
            process=psutil.Process(args.wait_pid)
            if not any(str(x).endswith('prepare_mechanism_counts_exp.py') for x in process.cmdline()):
                raise ValueError('Wait PID is not the expected counts converter')
            print('WAIT_COUNTS_CONVERTER',args.wait_pid,flush=True)
            process.wait()
        except psutil.NoSuchProcess: pass
        if not (data_root/args.task/'counts.h5ad').is_file():
            raise ValueError('Counts converter did not produce validated output')
    if args.stage=='split_profile':
        cohorts_stage(data_root,args.task,include_embeddings=False);profile_stage(data_root,args.task);return
    if args.stage in ['prepare','all']:
        attribution_stage(data_root,args.task)
        index_stage(data_root,args.task);state_stage(data_root,args.task);cohorts_stage(data_root,args.task)
        profile_stage(data_root,args.task)
        if args.stage=='prepare': return
        subprocess.run([PYTHON['attribution'],str(Path(__file__)), '--open_path',str(data_root),'--task',args.task,'--stage','graphs'],check=True)
        if 'random_signature' in args.models: random_stage(data_root,args.task)
        train_stage(data_root,args.task,args.models)
        subprocess.run([PYTHON['counts'],str(script_path('report_mechanism_exp.py')),'--open_path',str(BENCH)],check=True)
    elif args.stage=='train': train_stage(data_root,args.task,args.models)
    else: globals()[{'attribution':'attribution_stage','index':'index_stage','profile':'profile_stage','graphs':'graphs_stage','state':'state_stage','cohorts':'cohorts_stage','random':'random_stage'}[args.stage]](data_root,args.task)


if __name__=='__main__': main()

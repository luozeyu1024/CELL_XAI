"""Frozen FA34 classifiers evaluated on the independent TS485T organoid background."""

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
from types import SimpleNamespace
import mechanism_grouped_exp as preparation

ROOT=preparation.ROOT
BENCH=preparation.BENCH
TASK='GSE223779_TS485T'

from split_registry import load_cells


from split_registry import apply_splits

def counts_stage(root):
    source = root / TASK / 'counts.h5ad'
    if not source.is_file():
        raise FileNotFoundError('Prepare the registered target with the cohort module: ' + str(source))
    return source


def graph_stage(root):
    import numpy as np
    import pandas as pd
    package=preparation.GRAPH_ROOT
    policy=json.loads((package/'dataset_policy.json').read_text())['datasets']['GSE223779']
    k=int(policy['k'])
    if k!=1024: raise ValueError('Frozen source K policy changed; inspect source experiment before proceeding')
    build=preparation.module('alk_graph_builder',ROOT/'gene_sim_pathway/code/emb_KNN_graph_G/build_cell_signature_graphs_exp.py')
    utility=preparation.module('alk_signature_utils',ROOT/'assets/SIGnature/src/SIGnature/utils.py')
    mapper=build.GeneMapper(ROOT/'assets/hgnc_complete_set.txt')
    genes=pd.read_csv(package/'gene_vocab.tsv',sep='\t').gene_symbol.tolist()
    with np.load(package/'global_graphs/coverage.npz') as z: masks={key:z[key] for key in z.files}
    if not (package/TASK/'signatures.npz').exists():
        build.run_dataset(ROOT/'work/attributions'/TASK,
                          SimpleNamespace(save_path=str(package),dataset_top_k=k),mapper,genes,masks,utility)


def random_stage():
    import external_transfer_exp as external
    import pandas as pd
    external.TARGETS=[TASK]
    external.random_stage()
    qc=pd.read_csv(preparation.RANDOM_ROOT/TASK/'matching_qc.tsv',sep='\t')
    if len(qc)!=23672 or not qc.matching_passed.all(): raise ValueError('Incomplete structure-matched random control')


def signature_qc_stage():
    import numpy as np
    import pandas as pd
    folder=preparation.GRAPH_ROOT/TASK
    cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False)
    labels=pd.read_csv(preparation.METADATA_ROOT/TASK/'counts_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
    if not cells.cell_id.is_unique or set(cells.cell_id)!=set(labels.index): raise ValueError('Signature/metadata identity mismatch')
    with np.load(folder/'signatures.npz') as z: sizes=np.diff(z['ptr'])
    if len(sizes)!=len(cells) or (sizes<=0).any(): raise ValueError('Empty or mismatched graph signatures')
    cells['drug']=cells.cell_id.map(labels.drug);cells['n_nodes']=sizes
    k=json.loads((folder/'parameters.json').read_text())['top_k']
    records=[]
    for drug,part in cells.groupby('drug',sort=True):
        records.append({'dataset':TASK,'drug':drug,'n_cells':len(part),'source_k':k,
            'min_nodes':int(part.n_nodes.min()),'p10_nodes':float(part.n_nodes.quantile(.1)),
            'p25_nodes':float(part.n_nodes.quantile(.25)),'median_nodes':float(part.n_nodes.median()),
            'max_nodes':int(part.n_nodes.max()),'n_short_signatures':int(part.n_nodes.lt(k).sum()),
            'fraction_short_signatures':float(part.n_nodes.lt(k).mean()),'n_empty_signatures':0})
    pd.DataFrame(records).to_csv(BENCH/'output/summary/alk_target_signature_sizes.tsv',sep='\t',index=False)
    print(pd.DataFrame(records).to_string(index=False),flush=True)


def cohort_stage():
    import numpy as np
    import pandas as pd
    folder=BENCH/'data/cohorts'/TASK;folder.mkdir(parents=True,exist_ok=True)
    graph=preparation.GRAPH_ROOT/TASK
    cells=pd.read_csv(graph/'cells.tsv',sep='\t',keep_default_na=False)
    metadata=pd.read_csv(preparation.METADATA_ROOT/TASK/'counts_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
    if not cells.cell_id.is_unique or set(cells.cell_id)!=set(metadata.index): raise ValueError('Count/graph cell identity mismatch')
    for column in metadata.columns: cells[column]=cells.cell_id.map(metadata[column])
    cells['graph_row']=np.arange(len(cells));cells['response']=cells.label
    cells['y']=cells.response.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)
    cells['split']='external';cells['label_status']='author_organoid_phenotype'
    state=preparation.SE_ROOT/TASK
    ids=pd.read_csv(state/'cell_metadata.tsv',sep='\t').cell_id
    index=pd.Index(ids).get_indexer(cells.cell_id)
    if (index<0).any(): raise ValueError('SE cell identity mismatch')
    emb=np.load(state/'cell_embeddings.npy')[index,:2048]
    if emb.shape!=(len(cells),2048) or not np.isfinite(emb).all(): raise ValueError('Invalid native SE representation')
    np.save(folder/'state_cell_embeddings.npy',emb.astype(np.float32))
    cells.to_csv(folder/'cells.tsv',sep='\t',index=False)
    for package in [preparation.GRAPH_ROOT,preparation.RANDOM_ROOT]:
        existing=pd.read_csv(package/TASK/'cells.tsv',sep='\t')
        if existing.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Random/reference graph cell order mismatch')
        cells.to_csv(package/TASK/'cells.tsv',sep='\t',index=False)
    (folder/'data_sources.json').write_text(json.dumps({'graph_source':str(graph),'state_source':str(state),
        'source':'GSE223779 FA34 OS/OC','target':'TS485T OS/OC/OA','top_k':1024,
        'k_selection':'frozen original FA34 source policy; not an additional fixed-K control or target-tuned K',
        'target_training':False,'same_drug_primary':'untreated sensitive versus crizotinib-resistant',
        'cross_drug_secondary':'untreated sensitive versus alectinib-resistant'},indent=2)+'\n',encoding='utf-8')


def prediction_stage():
    import numpy as np
    import pandas as pd
    import torch
    import train_response_exp as core
    from cell_graph_dataset import CellGraphDataset
    from edge_null_graphs import remove_edges
    torch.set_num_threads(4);core.seed_all(42);summary=[]
    configurations=[('state_15b',BENCH/'output','gat_gine'),
        *[(r,BENCH/'output/transfer'/r,'gat_gine') for r in ['esm2_650m','qwen','ppi']],
        ('no_edges',BENCH/'output/transfer/edge_ablation/no_edges/shared','gat_gine'),
        ('random_signature',BENCH/'output/transfer/signature_ablation/random_genes/state_15b','gat_gine'),
        ('native_state_embedding',BENCH/'output','state_mlp')]
    for relation,base,kind in configurations:
        checkpoint=base/'within/GSE223779'/kind/'checkpoint.pt'
        model,saved=core.restore(checkpoint,'cuda')
        if relation=='random_signature':
            raw=CellGraphDataset(preparation.RANDOM_ROOT,TASK,'state_15b')
            cells=load_cells(BENCH/'data/cohorts'/TASK/'cells.tsv',TASK,mode='infer')
            if raw.cells.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Random graph order mismatch')
            data=[]
            for i,row in enumerate(cells.itertuples()):
                graph=raw.to_pyg(i);graph.x=graph.x[:,:1].clone();graph.y=torch.tensor([row.y]);data.append(graph)
            genes=raw.genes.tolist()
        else:
            graph_relation=relation if relation in ['esm2_650m','qwen','ppi'] else 'state_15b'
            data,cells,genes=core.load_cohort(BENCH,TASK,kind,graph_relation)
        if kind=='gat_gine':
            if genes!=saved['gene_symbols']: raise ValueError('Frozen gene identity vocabulary mismatch')
            if relation=='no_edges':
                for graph in data: remove_edges(graph)
        if set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids']):
            raise ValueError('Target cells leaked into source fitting')
        frame=core.predict(model,data,cells,kind,'cuda',32 if kind=='gat_gine' else 256)
        for col in ['patient_id','sample_id','drug']: frame[col]=frame.cell_id.map(cells.set_index('cell_id')[col])
        dest=base/'transfer'/('GSE223779_to_'+TASK)/kind;dest.mkdir(parents=True,exist_ok=True)
        frame.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
        scopes={
            'same_drug_crizotinib':apply_splits(frame,'GSE223779_TS485T',mode='infer',scope='crizotinib'),
            'cross_drug_alectinib':apply_splits(frame,'GSE223779_TS485T',mode='infer',scope='alectinib')}
        results={}
        for scope,part in scopes.items():
            result=core.metrics(part)
            cm=np.asarray(result['confusion_matrix_true_rows_sensitive_resistant'])
            for i,label in enumerate(['sensitive','resistant']): result['recall_'+label]=float(cm[i,i]/cm[i].sum())
            results[scope]=result
            summary.append(dict(dataset=TASK,model=relation,scope=scope,**{k:v for k,v in result.items() if not isinstance(v,list)}))
        core.write_json(dest/'metrics.json',dict(source='FA34',target='TS485T',checkpoint=str(checkpoint),
            target_training=False,target_threshold_tuning=False,threshold=.5,scopes=results,
            limitation='one held-out patient-derived organoid background; cells are not independent patients'))
        print('FROZEN_ALK',relation,{k:v['auroc'] for k,v in results.items()},flush=True)
        del data,model;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(BENCH/'output/summary/alk_frozen_transfer_metrics.tsv',sep='\t',index=False,float_format='%.17g')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    parser.add_argument('--wait_pid',type=int,default=0,help='Wait for the identified preparation batch to release the GPU')
    parser.add_argument('--stage',choices=['counts','prepare','graphs','graphs_worker','random','signature_qc','cohorts','predict','all'],default='all')
    args=parser.parse_args();root=Path(args.open_path)
    if args.wait_pid:
        import psutil
        try:
            process=psutil.Process(args.wait_pid)
            if not any(str(x).endswith('mechanism_batch_exp.py') for x in process.cmdline()):
                raise ValueError('Wait process is not the expected preparation batch')
            print('WAIT_PREPARATION_BATCH',args.wait_pid,flush=True)
            process.wait()
        except psutil.NoSuchProcess: pass
    if args.stage=='counts': counts_stage(root);return
    if args.stage=='signature_qc': signature_qc_stage();return
    if args.stage=='graphs_worker': graph_stage(root);return
    if args.stage in ['prepare','all']:
        counts_stage(root);preparation.attribution_stage(root,TASK);preparation.index_stage(root,TASK);preparation.state_stage(root,TASK)
        if args.stage=='prepare': return
    if args.stage in ['graphs','all']:
        import subprocess
        subprocess.run([preparation.PYTHON['attribution'],str(Path(__file__)), '--open_path',str(root),'--stage','graphs_worker'],check=True)
    if args.stage in ['random','all']: random_stage()
    if args.stage in ['cohorts','all']: cohort_stage()
    if args.stage in ['predict','all']: prediction_stage()


if __name__=='__main__': main()

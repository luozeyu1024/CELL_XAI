"""Frozen GSE223779 transfer to verified cell-line response datasets."""

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
import subprocess
import sys
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = PROJECT_ROOT
BENCH = ROOT/'model/response_benchmark'
TARGETS = ['GSE202234', 'GSE163836']
GRAPH_ROOT = Path(os.environ.get('CELL_XAI_GRAPH_ROOT', ROOT/'work/cell_graphs'))
METADATA_ROOT = Path(os.environ.get('CELL_XAI_COHORT_METADATA', ROOT/'work/cohort_metadata'))
UTILITY = ROOT/'assets/SIGnature/src/SIGnature/utils.py'
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'signature_attr_run/.cache/numba'))
cache = ROOT/'signature_attr_run/.cache'
for name in ['numba','tmp']:
    (cache/name).mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=str(cache/'tmp')
os.environ['TMP']=str(cache/'tmp')

from split_registry import load_cells


from split_registry import apply_splits

def module(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def graph_stage():
    import numpy as np
    import pandas as pd
    build = module('external_graph_builder', ROOT/'gene_sim_pathway/code/emb_KNN_graph_G/build_cell_signature_graphs_exp.py')
    utility = module('external_signature_utility', UTILITY)
    package = GRAPH_ROOT
    genes = pd.read_csv(package/'gene_vocab.tsv', sep='\t').gene_symbol.tolist()
    with np.load(package/'global_graphs/coverage.npz') as z:
        masks = {k:z[k] for k in z.files}
    mapper = build.GeneMapper(ROOT/'assets/hgnc_complete_set.txt')
    args = SimpleNamespace(save_path=str(package), dataset_top_k=1024)
    for ds in TARGETS:
        if not (package/ds/'signatures.npz').exists():
            build.run_dataset(ROOT/'work/attributions'/ds, args, mapper, genes, masks, utility)


def random_stage(pilot_cells=0):
    import pandas as pd
    from scipy import sparse
    build = module('external_random_builder', ROOT/'gene_sim_pathway/code/state_signature_ablation/build_random_state_signatures_exp.py')
    reference = GRAPH_ROOT
    out = ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation'
    args = SimpleNamespace(reference_path=str(reference), save_path=str(out),
        graph_path=str(ROOT/'gene_sim_pathway/data/emb_KNN_graph/state_ablation'),
        benchmark_path=str(BENCH), signature_utils=str(UTILITY), seed=42,
        count_jitter=.05, density_tolerance=.02, degree_cdf_tolerance=.05,
        isolation_tolerance=.02, max_steps=2000, restarts=12, pilot_cells=pilot_cells, rebuild=False)
    graph = sparse.load_npz(reference/'global_graphs/state_15b.npz').tocsr()
    genes = pd.read_csv(reference/'gene_vocab.tsv', sep='\t')
    for ds in TARGETS:
        if not (out/ds/'signatures.npz').exists():
            build.build_dataset(args, ds, graph, genes)


def cohort_stage():
    import anndata as ad
    import numpy as np
    import pandas as pd
    for ds in TARGETS:
        folder = BENCH/'data/cohorts'/ds
        folder.mkdir(parents=True, exist_ok=True)
        package = GRAPH_ROOT/ds
        cells = pd.read_csv(package/'cells.tsv', sep='\t', keep_default_na=False)
        labels = pd.read_csv(METADATA_ROOT/ds/'counts_labels.tsv', sep='\t').set_index('Cell_ID')
        cells['graph_row'] = np.arange(len(cells))
        assert cells.cell_id.is_unique and set(cells.cell_id)==set(labels.index)
        for column in labels.columns:
            cells[column] = cells.cell_id.map(labels[column])
        cells['response'] = cells['label']
        cells['y'] = cells.response.map({'sensitive':0,'resistant':1}).fillna(-1).astype(int)
        cells['split'] = np.where(cells.y.ge(0), 'external', 'excluded_unknown_response')
        cells['label_status']=np.where(cells.y.ge(0),'source_evidence_verified','unknown_mixed_identity')
        cells.to_csv(folder/'cells.tsv', sep='\t', index=False)
        cells.to_csv(package/'cells.tsv',sep='\t',index=False)
        graph_parameters=json.loads((package/'parameters.json').read_text())
        graph_parameters['n_verified_response_labels']=int(cells.y.ge(0).sum())
        (package/'parameters.json').write_text(json.dumps(graph_parameters,indent=2)+'\n')
        random_folder=ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation'/ds
        if (random_folder/'cells.tsv').exists():
            random_cells=pd.read_csv(random_folder/'cells.tsv',sep='\t')
            assert random_cells.cell_id.tolist()==cells.cell_id.tolist()
            cells.to_csv(random_folder/'cells.tsv',sep='\t',index=False)
        state = ad.read_h5ad(BENCH/'data/state_embeddings'/ds/'state_embeddings.h5ad')
        index = state.obs_names.get_indexer(cells.cell_id)
        assert (index>=0).all()
        emb = np.asarray(state.obsm['X_state'])[index, :2048]
        assert emb.shape==(len(cells),2048) and np.isfinite(emb).all()
        np.save(folder/'state_cell_embeddings.npy', emb.astype(np.float32))
        (folder/'data_sources.json').write_text(json.dumps(dict(graph_source=str(package),
            label_source=str(METADATA_ROOT/ds/'counts_labels.tsv'),
            state_source=str(BENCH/'data/state_embeddings'/ds/'state_embeddings.h5ad'),
            top_k=1024, target_training=False, target_label_selection=False), indent=2)+'\n')
    sync_summaries()


def sync_summaries():
    for package in [GRAPH_ROOT, ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation']:
        path=package/'summary.json'
        info=json.loads(path.read_text())
        previous={item['dataset']:item for item in info['datasets']}
        for ds in TARGETS:
            parameter=package/ds/'parameters.json'
            if parameter.exists():previous[ds]=json.loads(parameter.read_text())
        info['datasets']=list(previous.values())
        path.write_text(json.dumps(info,indent=2)+'\n')


def validate_graphs_stage():
    sync_summaries()
    subprocess.run([sys.executable,str(ROOT/'gene_sim_pathway/code/state_signature_ablation/verify_random_state_signatures_exp.py'),
        '--open_path',str(ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation'),
        '--reference_path',str(GRAPH_ROOT),
        '--graph_path',str(ROOT/'gene_sim_pathway/data/emb_KNN_graph/state_ablation')],check=True)


def prediction_stage():
    import numpy as np
    import pandas as pd
    import torch
    import train_response_exp as core
    from cell_graph_dataset import CellGraphDataset
    from edge_null_graphs import remove_edges
    torch.set_num_threads(4)
    core.seed_all(42)
    summary=[]
    configurations = [('state_15b', BENCH/'output', 'gat_gine'),
        *[(r, BENCH/'output/transfer'/r, 'gat_gine') for r in ['esm2_650m','qwen','ppi']],
        ('no_edges', BENCH/'output/transfer/edge_ablation/no_edges/shared', 'gat_gine'),
        ('random_signature', BENCH/'output/transfer/signature_ablation/random_genes/state_15b', 'gat_gine'),
        ('native_state_embedding', BENCH/'output', 'state_mlp')]
    for ds in TARGETS:
        for relation, base, kind in configurations:
            checkpoint = base/'within/GSE223779'/kind/'checkpoint.pt'
            model, saved = core.restore(checkpoint, 'cuda')
            if relation=='random_signature':
                raw=CellGraphDataset(ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation', ds, 'state_15b')
                cells=load_cells(BENCH/'data/cohorts'/ds/'cells.tsv',ds,mode='infer')
                assert raw.cells.cell_id.tolist()==cells.cell_id.tolist()
                data=[]
                for i,row in enumerate(cells.itertuples()):
                    g=raw.to_pyg(i);g.x=g.x[:,:1].clone();g.y=torch.tensor([row.y]);data.append(g)
                genes=raw.genes.tolist()
            else:
                r=relation if relation in ['esm2_650m','qwen','ppi'] else 'state_15b'
                data,cells,genes=core.load_cohort(BENCH,ds,kind,r)
            if kind=='gat_gine':
                assert genes==saved['gene_symbols']
                if relation=='no_edges':
                    for g in data:remove_edges(g)
            assert not set(cells.cell_id)&set(saved['training_cell_ids']+saved['validation_cell_ids'])
            frame=core.predict(model,data,cells,kind,'cuda',32 if kind=='gat_gine' else 256)
            meta=cells.set_index('cell_id')
            for col in ['sample_id','cell_line','original_group','treatment','batch']:
                if col in meta:frame[col]=frame.cell_id.map(meta[col])
            dest=base/'transfer'/f'GSE223779_to_{ds}'/kind
            dest.mkdir(parents=True, exist_ok=True)
            frame.to_csv(dest/'predictions.tsv',sep='\t',index=False,float_format='%.17g')
            scopes={'all_labeled':frame[frame.y.ge(0)]}
            if ds=='GSE163836':
                scopes['primary_FCIBC02_pair']=apply_splits(frame,'GSE163836',mode='infer',scope='FCIBC02')
                scopes['additional_SUM149PR']=apply_splits(frame,'GSE163836',mode='infer',scope='SUM149')
            else:
                for treatment,sub in frame.groupby('treatment'):scopes['treatment_'+str(treatment)]=sub[sub.y.ge(0)]
            results={}
            for scope,sub in scopes.items():
                m=core.metrics(sub)
                if sub.y[sub.y.ge(0)].nunique()<2:
                    for key in ['balanced_accuracy','mcc','macro_f1']:m[key]=None
                if m['n_cells']:
                    cm=np.asarray(m['confusion_matrix_true_rows_sensitive_resistant'])
                    for i,label in enumerate(['sensitive','resistant']):
                        m['recall_'+label]=float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None
                results[scope]=m
                summary.append(dict(dataset=ds,model=relation,scope=scope,**{k:v for k,v in m.items() if not isinstance(v,list)}))
            core.write_json(dest/'metrics.json',dict(source='GSE223779',target=ds,relation=relation,
                checkpoint=str(checkpoint),target_training=False,target_threshold_tuning=False,
                threshold=.5,scopes=results))
            print(ds,relation,{k:v.get('auroc') for k,v in results.items()},flush=True)
            del data,model;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(BENCH/'output/summary/external_transfer_metrics.tsv',sep='\t',index=False,float_format='%.17g')


def report_stage():
    """Recompute scope metrics from saved predictions without neural inference."""
    import numpy as np
    import pandas as pd
    import train_response_exp as core
    rows=[]
    for ds in TARGETS:
        folders=[BENCH/'output/transfer'/f'GSE223779_to_{ds}'/kind for kind in ['gat_gine','state_mlp']]
        folders += [BENCH/'output/transfer'/relation/'transfer'/f'GSE223779_to_{ds}'/'gat_gine' for relation in ['esm2_650m','qwen','ppi','edge_ablation/no_edges/shared','signature_ablation/random_genes/state_15b']]
        for folder in folders:
            f=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
            report=json.loads((folder/'metrics.json').read_text())
            for scope in report['scopes']:
                if scope=='all_labeled':sub=f[f.y.ge(0)]
                elif scope=='primary_FCIBC02_pair':sub=f[f.cell_line.eq('FCIBC02')]
                elif scope=='additional_SUM149PR':sub=f[f.original_group.eq('SUM149PR')]
                elif scope.startswith('treatment_'):sub=f[f.treatment.eq(scope[len('treatment_'):])&f.y.ge(0)]
                else:raise ValueError(scope)
                m=core.metrics(sub)
                if sub.y[sub.y.ge(0)].nunique()<2:
                    for key in ['balanced_accuracy','mcc','macro_f1']:m[key]=None
                cm=np.asarray(m['confusion_matrix_true_rows_sensitive_resistant'])
                for i,label in enumerate(['sensitive','resistant']):
                    m['recall_'+label]=float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None
                report['scopes'][scope]=m
                rows.append(dict(dataset=ds,model=report['relation'],scope=scope,**{k:v for k,v in m.items() if not isinstance(v,list)}))
            core.write_json(folder/'metrics.json',report)
    pd.DataFrame(rows).to_csv(BENCH/'output/summary/external_transfer_metrics.tsv',sep='\t',index=False,float_format='%.17g')


def validation_stage():
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from validate_results_exp import check_metrics as check_binary_metrics
    from sklearn.metrics import confusion_matrix
    import torch
    import train_response_exp as core
    from cell_graph_dataset import CellGraphDataset
    from edge_null_graphs import remove_edges
    torch.set_num_threads(4)
    core.seed_all(42)
    checkpoint=torch.load(BENCH/'output/within/GSE223779/gat_gine/checkpoint.pt',map_location='cpu',weights_only=False)
    seen=np.zeros(len(checkpoint['gene_symbols']),bool)
    seen[checkpoint['training_observed_gene_ids']]=True
    input_qc=[]
    def check_metrics(frame, saved):
        frame=frame[frame.y.ge(0)]
        cm=confusion_matrix(frame.y,frame.probability_resistant.ge(.5),labels=[0,1])
        for i,label in enumerate(['sensitive','resistant']):
            expected=float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None
            if expected is None:assert saved['recall_'+label] is None
            else:assert np.isclose(expected,saved['recall_'+label])
        if frame.y.nunique()==2:
            check_binary_metrics(frame,saved)
        else:
            assert saved['n_cells']==len(frame)
            assert all(saved[k] is None for k in ['auroc','auprc_sensitive','auprc_resistant'])
            assert all(saved[k] is None for k in ['balanced_accuracy','mcc','macro_f1'])
            assert confusion_matrix(frame.y,frame.probability_resistant.ge(.5),labels=[0,1]).tolist()==saved['confusion_matrix_true_rows_sensitive_resistant']
    checks=[]
    for ds in TARGETS:
        cohort=pd.read_csv(BENCH/'data/cohorts'/ds/'cells.tsv',sep='\t')
        assert cohort.cell_id.is_unique
        graph=GRAPH_ROOT/ds
        with np.load(graph/'signatures.npz') as z:
            assert len(z['ptr'])==len(cohort)+1 and np.diff(z['ptr']).max()<=1024
            assert np.isfinite(z['features']).all()
            n=np.diff(z['ptr'])
            assert (n>0).all()
            unseen=np.add.reduceat((~seen[z['gene_id']]).astype(float),z['ptr'][:-1])/n
            q=pd.read_csv(graph/'signature_qc.tsv',sep='\t')
            row=dict(dataset=ds,n_cells=len(n),n_eligible_genes=len(pd.read_csv(graph/'eligible_genes.tsv',sep='\t')))
            for prefix,values in [('nodes',n),('coverage',q.selected_importance_fraction),('source_unseen_nodes',unseen)]:
                for label,quantile in [('q25',.25),('median',.5),('q75',.75)]:row[prefix+'_'+label]=float(np.quantile(values,quantile))
            input_qc.append(row)
        atts=sparse.load_npz(ROOT/'work/attributions'/ds/'attributions/scimilarity_ig.npz')
        assert atts.shape[0]==len(cohort) and np.isfinite(atts.data).all() and (atts.data>=0).all()
        sums=np.asarray(atts.sum(axis=1)).ravel()
        assert np.all((sums==0)|np.isclose(sums,1000,atol=.01))
        qc=pd.read_csv(ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation'/ds/'matching_qc.tsv',sep='\t')
        assert qc.cell_id.tolist()==cohort.cell_id.tolist() and qc.matching_passed.all()
        for prefix,values in [('random_overlap_reference_fraction',qc.original_signature_overlap/qc.reference_nodes),
            ('random_jaccard',qc.original_signature_overlap/(qc.reference_nodes+qc.selected_nodes-qc.original_signature_overlap)),
            ('expressed_candidate_pool',qc.detected_candidate_genes)]:
            for label,quantile in [('q25',.25),('median',.5),('q75',.75)]:input_qc[-1][prefix+'_'+label]=float(np.quantile(values,quantile))
        paths=[BENCH/'output/transfer'/f'GSE223779_to_{ds}'/kind for kind in ['gat_gine','state_mlp']]
        paths += [BENCH/'output/transfer'/relation/'transfer'/f'GSE223779_to_{ds}'/'gat_gine' for relation in ['esm2_650m','qwen','ppi']]
        paths += [BENCH/'output/transfer'/suffix/'transfer'/f'GSE223779_to_{ds}'/'gat_gine' for suffix in ['edge_ablation/no_edges/shared','signature_ablation/random_genes/state_15b']]
        for folder in paths:
            frame=pd.read_csv(folder/'predictions.tsv',sep='\t',float_precision='round_trip')
            report=json.loads((folder/'metrics.json').read_text())
            assert frame.cell_id.tolist()==cohort.cell_id.tolist() and frame.y.tolist()==cohort.y.tolist()
            assert not report['target_training'] and not report['target_threshold_tuning']
            check_metrics(frame[frame.y.ge(0)],report['scopes']['all_labeled'])
            if ds=='GSE163836':
                check_metrics(apply_splits(frame,'GSE163836',mode='infer',scope='FCIBC02'),report['scopes']['primary_FCIBC02_pair'])
                check_metrics(apply_splits(frame,'GSE163836',mode='infer',scope='SUM149'),report['scopes']['additional_SUM149PR'])
            else:
                for treatment,sub in frame.groupby('treatment'):
                    check_metrics(sub[sub.y.ge(0)],report['scopes']['treatment_'+str(treatment)])
            idx=np.unique(np.linspace(0,len(cohort)-1,min(16,len(cohort)),dtype=int))
            model,saved=core.restore(Path(report['checkpoint']),'cuda')
            kind=saved['model_type'];relation=report['relation']
            chosen=cohort.iloc[idx].reset_index(drop=True)
            if kind=='state_mlp':
                x=np.load(BENCH/'data/cohorts'/ds/'state_cell_embeddings.npy')
                batch=torch.from_numpy(x[idx])
            else:
                root=ROOT/'gene_sim_pathway/data/cell_signature_graphs_state_ablation' if relation=='random_signature' else GRAPH_ROOT
                raw=CellGraphDataset(root,ds,relation if relation in ['esm2_650m','qwen','ppi'] else 'state_15b')
                assert raw.genes.tolist()==saved['gene_symbols']
                batch=[]
                for i in idx:
                    g=raw.to_pyg(int(i));g.x=g.x[:,:1].clone();g.y=torch.tensor([cohort.y.iloc[i]])
                    if relation=='no_edges':remove_edges(g)
                    batch.append(g)
            fresh=core.predict(model,batch,chosen,kind,'cuda',16)
            error=float(np.max(np.abs(fresh.margin_resistant_minus_sensitive.to_numpy()-frame.margin_resistant_minus_sensitive.iloc[idx].to_numpy())))
            assert np.allclose(fresh.margin_resistant_minus_sensitive,frame.margin_resistant_minus_sensitive.iloc[idx],atol=2e-5,rtol=2e-5)
            assert np.array_equal(fresh.predicted.to_numpy(),frame.predicted.iloc[idx].to_numpy())
            checks.append(dict(dataset=ds,relation=relation,n_predictions=len(frame),n_reloaded_predictions=len(idx),max_logit_margin_error=error,status='passed'))
            del model,batch;gc.collect();torch.cuda.empty_cache()
    (BENCH/'output/summary/external_transfer_validation.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2)+'\n')
    pd.DataFrame(input_qc).to_csv(BENCH/'output/summary/external_graph_input_qc.tsv',sep='\t',index=False)
    print('External transfer validation passed',flush=True)


def main():
    global TARGETS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['graphs','random','cohorts','predict','report','validate','validate_graphs','all'])
    p.add_argument('--open_path',type=Path,default=ROOT/'work/counts',help='Prepared count matrices by dataset')
    p.add_argument('--wait_pid',type=int,default=0)
    p.add_argument('--pilot_cells',type=int,default=0,help='Read-only random-signature feasibility check')
    p.add_argument('--datasets',nargs='+',choices=TARGETS,default=TARGETS)
    a=p.parse_args()
    TARGETS=a.datasets
    if a.stage=='all':
        import psutil
        if a.wait_pid:
            try: psutil.Process(a.wait_pid).wait()
            except psutil.NoSuchProcess: pass
        for ds in TARGETS:
            out=ROOT/'work/attributions'/ds
            if not (out/'manifests/scimilarity_ig_manifest.json').exists():
                subprocess.run([os.environ.get('CELL_XAI_ATTR_PYTHON', sys.executable),
                    str(ROOT/'signature_attr_run/code/run_attribution.py'), '--model','scimilarity',
                    '--h5ad',str(a.open_path/ds/'counts.h5ad'),
                    '--out-dir',str(out),'--model-dir',str(ROOT/'assets/scimilarity'),
                    '--method','ig','--batch-size','100','--device','cuda','--counts-policy','auto'],check=True)
        subprocess.run([os.environ.get('CELL_XAI_ATTR_PYTHON', sys.executable),str(Path(__file__)), '--stage','graphs','--datasets',*TARGETS],check=True)
        for ds in TARGETS:
            out=BENCH/'data/state_embeddings'/ds
            if not (out/'validation.json').exists():
                subprocess.run([os.environ.get('CELL_XAI_SE_PYTHON', sys.executable),str(ROOT/'env/state/test/run_se_test.py'),
                    '--open_path',str(a.open_path/ds/'counts.h5ad'),
                    '--model_path',str(ROOT/'assets/SE-600M'),'--save_path',str(out),
                    '--batch_size','8','--max_cells','0','--seed','42'],check=True)
        cohort_stage()
        random_stage()
        validate_graphs_stage()
        prediction_stage()
        report_stage()
        validation_stage()
        return
    {'graphs':graph_stage,'random':lambda:random_stage(a.pilot_cells),'cohorts':cohort_stage,'predict':prediction_stage,'report':report_stage,'validate':validation_stage,'validate_graphs':validate_graphs_stage}[a.stage]()


if __name__=='__main__':main()

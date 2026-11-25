"""Prepare common-vocabulary SCimilarity-IG cell graphs with shared CSR priors."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gene_target_neighbors_exp import GeneMapper, MODELS
from cell_graph_dataset import RELATIONS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def max_csr(src, dst, weight, n):
    valid = (src >= 0) & (dst >= 0) & (src != dst) & np.isfinite(weight)
    src, dst, weight = src[valid], dst[valid], weight[valid]
    keys = src.astype(np.int64)*n + dst
    order = np.argsort(keys, kind='stable')
    keys, weight = keys[order], weight[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    unique, values = keys[starts], np.maximum.reduceat(weight, starts)
    result = sparse.csr_matrix((values.astype(np.float32), (unique//n, unique%n)), shape=(n,n))
    result.sort_indices()
    return result


def build_globals(args, mapper):
    root = Path(args.open_path); out = Path(args.save_path)
    global_dir = out/'global_graphs'; global_dir.mkdir(parents=True, exist_ok=True)
    mappings, known = {}, {}
    for model, folder in MODELS.items():
        with np.load(root/folder/'human_swissprot_aligned.npz', allow_pickle=False) as z:
            ids, names = z['ids'].astype(str), z['gene_names'].astype(str)
        explicit = {}
        if model == 'state_15b':
            table = pd.read_csv(root/folder/'protein_gene_mapping.tsv', sep='\t', keep_default_na=False)
            explicit = dict(zip(table.accession, [mapper.canonical(g) for g in table.hgnc_symbol]))
        mapping = {pid: explicit.get(pid) if model == 'state_15b' else mapper.protein_gene(pid,name)
                   for pid,name in zip(ids,names)}
        mappings[model] = mapping
        pd.DataFrame([{'uniprot_id':p,'gene_symbol':g or '',
                       'status':'mapped' if g else 'unmapped_or_ambiguous'} for p,g in mapping.items()]).to_csv(
            global_dir/f'{model}_protein_mapping.tsv',sep='\t',index=False)
    raw = pd.read_csv(args.ppi_path, usecols=['g1_symbol','g2_symbol','conn'])
    mapping = {g:mapper.canonical(g) for g in set(raw.g1_symbol.astype(str))|set(raw.g2_symbol.astype(str))}
    raw['source'] = raw.g1_symbol.map(mapping); raw['target'] = raw.g2_symbol.map(mapping)
    known['ppi'] = set(raw.source.dropna()) | set(raw.target.dropna())
    raw.loc[raw.source.isna() | raw.target.isna()].to_csv(global_dir/'ppi_unmapped.tsv', sep='\t', index=False)
    for model in MODELS:
        with np.load(root/'emb_KNN_graph'/f'protein_knn_{model}_k30.npz',allow_pickle=True) as z:
            node_ids=z['node_ids'].astype(str)
        known[model] = {mappings[model].get(p) for p in node_ids} - {None}
    genes = sorted(set.union(*known.values())); index = {g:i for i,g in enumerate(genes)}
    pd.DataFrame({'gene_id':np.arange(len(genes)), 'gene_symbol':genes}).to_csv(out/'gene_vocab.tsv',sep='\t',index=False)
    masks = {r:np.array([g in known[r] for g in genes]) for r in RELATIONS}
    np.savez_compressed(global_dir/'coverage.npz',**masks)
    summary = {}
    for model in MODELS:
        with np.load(root/'emb_KNN_graph'/f'protein_knn_{model}_k30.npz',allow_pickle=True) as z:
            ids=z['node_ids'].astype(str); edges=z['edge_index']; weights=z['edge_weight']
        mapped=np.array([index.get(mappings[model].get(p),-1) for p in ids],dtype=np.int64)
        graph=max_csr(mapped[edges[0]], mapped[edges[1]], weights,len(genes))
        if (graph!=graph.T).nnz:
            raise ValueError('Asymmetric mapped graph: '+model)
        sparse.save_npz(global_dir/f'{model}.npz',graph)
        summary[model]={'known_genes':len(known[model]),'undirected_edges':graph.nnz//2,
                        'source':str((root/'emb_KNN_graph'/f'protein_knn_{model}_k30.npz').resolve())}
    keep=raw.source.notna() & raw.target.notna() & np.isfinite(raw.conn) & (raw.conn>=args.ppi_threshold)
    table=raw.loc[keep]; s=table.source.map(index).to_numpy(); t=table.target.map(index).to_numpy(); w=table.conn.to_numpy()
    graph=max_csr(np.r_[s,t],np.r_[t,s],np.r_[w,w],len(genes))
    sparse.save_npz(global_dir/'ppi.npz',graph)
    summary['ppi']={'known_genes':len(known['ppi']), 'undirected_edges':graph.nnz//2,
                    'source':str(Path(args.ppi_path).resolve()), 'threshold_inclusive':args.ppi_threshold}
    write_json(global_dir/'parameters.json',{'mapping':'HGNC approved; ambiguous excluded',
               'aggregation':'max weight over mapped protein edges; self edges removed',
               'node_order':'gene_vocab.tsv', 'relations':summary})
    print('Global graphs:',summary,flush=True)
    return genes, masks


def unique_canonical(names, mapper):
    mapped=[mapper.canonical(g) for g in names]
    counts=Counter(g for g in mapped if g)
    return {g:i for i,g in enumerate(mapped) if g and counts[g]==1}, [g for g,c in counts.items() if c>1]


def run_dataset(folder,args,mapper,genes,masks,utils):
    import anndata as ad
    manifest_path=folder/'manifests/scimilarity_ig_manifest.json'
    if not manifest_path.exists():
        return None
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    info=manifest['input']; name=folder.name
    print('Preparing',name,flush=True)
    obs=ad.read_h5ad(info['h5ad'])
    indices=np.asarray(info['cell_indices_used'],dtype=int)
    obs=obs[indices].copy()
    if obs.obs_names.astype(str).tolist()!=info['cell_ids_used'] or not obs.obs_names.is_unique:
        raise ValueError('Cell identity mismatch: '+name)
    obs.var_names_make_unique()
    raw_map,raw_duplicates=unique_canonical(obs.var_names,mapper)
    model_names=json.loads((folder/'attributions/scimilarity_ig_var_names.json').read_text(encoding='utf-8'))
    model_map,model_duplicates=unique_canonical(model_names,mapper)
    counts=sparse.csr_matrix(obs.layers['counts'] if 'counts' in obs.layers else obs.X,dtype=np.float32)
    if not np.isfinite(counts.data).all() or np.any(counts.data<0) or not np.allclose(counts.data,np.rint(counts.data),rtol=0,atol=1e-6):
        raise ValueError('Counts validation failed: '+name)
    # Match official preprocessing: exact source feature names, alignment first,
    # normalize to 10k over the model gene space, then log1p.
    expression=ad.AnnData(X=counts,obs=obs.obs.copy(),var=obs.var.copy())
    expression.layers['counts']=counts
    expression=utils.align_dataset(expression, model_names, keep_obsm=False, gene_overlap_threshold=500)
    expression=utils.lognorm_counts(expression)
    if expression.obs_names.astype(str).tolist()!=info['cell_ids_used']:
        raise ValueError('Preprocessing reordered cells')
    x=sparse.csr_matrix(expression.X)
    attributions=sparse.load_npz(folder/'attributions/scimilarity_ig.npz').tocsr()
    if attributions.shape!=(len(obs),len(model_names)):
        raise ValueError('Attribution shape mismatch')
    if not np.isfinite(attributions.data).all() or np.any(attributions.data<0):
        raise ValueError('Invalid attribution values')
    sums=np.asarray(attributions.sum(axis=1)).ravel()
    if not np.all((sums==0)|np.isclose(sums,1000,rtol=1e-5,atol=.01)):
        raise ValueError('Unexpected attribution normalization')
    common=np.logical_and.reduce([masks[r] for r in RELATIONS])
    eligible=[]; columns=[]
    for gid,g in enumerate(genes):
        # Attribution preprocessing matched source names exactly, not merely aliases.
        if common[gid] and g in raw_map and g in model_map and model_names[model_map[g]] in obs.var_names:
            eligible.append(gid); columns.append(model_map[g])
    eligible=np.asarray(eligible,dtype=np.int32); columns=np.asarray(columns,dtype=np.int32)
    out=Path(args.save_path)/name; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'gene_id':eligible,'gene_symbol':np.asarray(genes)[eligible],
                  'attribution_column':columns}).to_csv(out/'eligible_genes.tsv',sep='\t',index=False)
    write_json(out/'excluded_gene_collisions.json',{'source_canonical_collisions':raw_duplicates,
                                                 'model_canonical_collisions':model_duplicates,
                                                 'policy':'exclude all members of ambiguous canonical collisions'})
    metadata=obs.obs.copy(); metadata.insert(0,'cell_id',obs.obs_names.astype(str))
    metadata.to_csv(out/'source_metadata.tsv',sep='\t',index=False)
    graphs={r:sparse.load_npz(Path(args.save_path)/'global_graphs'/f'{r}.npz').tocsr() for r in RELATIONS}
    ptr=[0]; node_ids=[]; features=[]; qc=[]; cells=[]
    for row,cell_id in enumerate(info['cell_ids_used']):
        scores=attributions[row,columns].toarray().ravel()
        order=np.argsort(-scores,kind='stable')
        order=order[scores[order]>0][:args.dataset_top_k]
        ids=eligible[order]; selected=columns[order]; value=scores[order]
        n=len(ids)
        values=x[row,selected].toarray().ravel()
        rank=(np.arange(n,dtype=np.float32)/max(n-1,1))
        feat=np.column_stack([values,value/1000,rank]).astype(np.float32)
        node_ids.append(ids); features.append(feat); ptr.append(ptr[-1]+n)
        total=float(sums[row])
        record={'cell_id':cell_id,'n_nodes':n,'total_importance':total,
                'eligible_importance_fraction':float(scores.sum()/total) if total else np.nan,
                'selected_importance_fraction':float(value.sum()/total) if total else np.nan}
        for relation,graph in graphs.items():
            sub=graph[ids][:,ids]
            degree=np.diff(sub.indptr)
            if n:
                nc,labels=connected_components(sub,directed=False)
                largest=np.bincount(labels).max()/n
            else:
                nc=0; largest=np.nan
            record.update({f'{relation}_undirected_edges':sub.nnz//2,
                           f'{relation}_isolated_fraction':float((degree==0).mean()) if n else np.nan,
                           f'{relation}_largest_component_fraction':float(largest),
                           f'{relation}_n_components':int(nc)})
        qc.append(record)
        cells.append({'cell_id':cell_id,'source_row':int(indices[row]),'y':-1,
                      'label_status':'response_label_not_verified','has_signature':bool(n)})
        if (row+1)%5000==0: print(name,row+1,'cells',flush=True)
    np.savez_compressed(out/'signatures.npz',ptr=np.asarray(ptr,dtype=np.int64),
                        gene_id=np.concatenate(node_ids),features=np.concatenate(features))
    pd.DataFrame(cells).to_csv(out/'cells.tsv',sep='\t',index=False)
    frame=pd.DataFrame(qc); frame.to_csv(out/'signature_qc.tsv',sep='\t',index=False)
    summary={'dataset':name,'n_cells':len(cells),'n_eligible_genes':len(eligible),
             'n_signature_nodes':int(ptr[-1]),'n_cells_with_signature':int(frame.n_nodes.gt(0).sum()),
             'median_selected_importance_fraction':float(frame.selected_importance_fraction.median()),
             'n_verified_response_labels':0,'source_h5ad':info['h5ad'],
             'attribution_manifest':str(manifest_path.resolve()),'top_k':args.dataset_top_k,
             'vocabulary':'common_four_graphs_measured_unique_HGNC_positive_attribution',
             'feature_columns':['lognorm_expression','importance_per_1000','rank_percentile_0_best'],
             'expression_preprocessing':'official SIGnature align_dataset then lognorm_counts; no HVG',
             'rank_ties':'gene_symbol_ascending','runtime':{'python':sys.version.split()[0],
              'numpy':np.__version__,'pandas':pd.__version__}}
    write_json(out/'parameters.json',summary)
    print('Finished',name,summary['n_cells'],'cells',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path',required=True); p.add_argument('--save_path',required=True)
    p.add_argument('--attribution_path',required=True); p.add_argument('--hgnc_path',required=True)
    p.add_argument('--ppi_path',required=True); p.add_argument('--signature_utils',required=True)
    p.add_argument('--top_k',type=int,default=512); p.add_argument('--ppi_threshold',type=float,default=.5)
    p.add_argument('--policy_path',help='Dataset-specific K from a reviewed policy JSON')
    p.add_argument('--datasets',nargs='+',help='Limit construction to these attribution cohort directories')
    args=p.parse_args()
    if args.top_k<1: p.error('top_k must be positive')
    policy_document=json.loads(Path(args.policy_path).read_text(encoding='utf-8')) if args.policy_path else None
    policy=policy_document['datasets'] if policy_document else None
    attribution_root=Path(args.attribution_path)
    cohorts=[attribution_root/name for name in args.datasets] if args.datasets else sorted(attribution_root.iterdir())
    if args.datasets and any(not (folder/'manifests/scimilarity_ig_manifest.json').is_file() for folder in cohorts):
        p.error('Every selected cohort requires a SCimilarity IG manifest')
    if policy and any(not isinstance(v['k'],int) or v['k']<1 for v in policy.values()):
        p.error('Every policy K must be a positive integer')
    mapper=GeneMapper(args.hgnc_path)
    genes,masks=build_globals(args,mapper)
    if policy_document:
        policy_document['use']='descriptive_applied_to_current_graph_package'
        write_json(Path(args.save_path)/'dataset_policy.json',policy_document)
    spec=importlib.util.spec_from_file_location('signature_official_utils',args.signature_utils)
    utils=importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)
    summaries=[]; excluded=[]
    for folder in cohorts:
        if not folder.is_dir(): continue
        if policy is not None and (folder/'manifests/scimilarity_ig_manifest.json').exists() and folder.name not in policy:
            raise ValueError('Missing policy for usable dataset: '+folder.name)
        args.dataset_top_k=policy[folder.name]['k'] if policy and folder.name in policy else args.top_k
        result=run_dataset(folder,args,mapper,genes,masks,utils)
        if result: summaries.append(result)
        else:
            out=Path(args.save_path)/folder.name;out.mkdir(parents=True,exist_ok=True)
            reason={'dataset':folder.name,'status':'skipped_no_scimilarity_attribution',
                    'source_status':str((folder/'dataset_status.json').resolve())}
            write_json(out/'dataset_status.json',reason); excluded.append(reason)
    write_json(Path(args.save_path)/'summary.json',{'datasets':summaries,'excluded':excluded})


if __name__=='__main__': main()

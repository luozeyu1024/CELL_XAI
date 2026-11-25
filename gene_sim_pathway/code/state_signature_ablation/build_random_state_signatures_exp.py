"""Build structure-matched random-gene STATE signatures without using attribution ranks."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zlib
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from matched_random_genes import graph_degrees,match_subset,random_initial

DATASETS=['GSE115978','GSE140440','GSE143363','GSE149383','GSE186960','GSE223779']


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def expression_matrix(info,utility_path):
    manifest=json.loads(Path(info['attribution_manifest']).read_text(encoding='utf-8'))['input']
    obs=ad.read_h5ad(info['source_h5ad']);obs=obs[np.asarray(manifest['cell_indices_used'],int)].copy()
    assert obs.obs_names.astype(str).tolist()==manifest['cell_ids_used']
    obs.var_names_make_unique()
    counts=sparse.csr_matrix(obs.layers['counts'] if 'counts' in obs.layers else obs.X,dtype=np.float32)
    assert np.isfinite(counts.data).all() and (counts.data>=0).all() and np.allclose(counts.data,np.rint(counts.data),rtol=0,atol=1e-6)
    names=json.loads((Path(info['attribution_manifest']).parents[1]/'attributions/scimilarity_ig_var_names.json').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('official_signature_utils',utility_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expression=ad.AnnData(X=counts,obs=obs.obs.copy(),var=obs.var.copy());expression.layers['counts']=counts
    expression=module.align_dataset(expression,names,keep_obsm=False,gene_overlap_threshold=500)
    # Exact same operation as official lognorm_counts: normalize over the
    # aligned model genes to 10,000 counts, followed by elementwise log1p.
    # SciPy implementation avoids unrelated Scanpy import/cache initialization.
    normalized=sparse.csr_matrix(expression.layers['counts'],dtype=np.float32).copy()
    totals=np.asarray(normalized.sum(axis=1)).ravel().astype(np.float32)
    factors=totals/np.float32(10000.)
    factors[factors==0]=1
    normalized.data/=np.repeat(factors,np.diff(normalized.indptr))
    np.log1p(normalized.data,out=normalized.data)
    return normalized,manifest['cell_ids_used']


def build_dataset(a,ds,graph,genes):
    print('PREPARE',ds,flush=True)
    reference=Path(a.reference_path)/ds
    info=json.loads((reference/'parameters.json').read_text(encoding='utf-8'))
    eligible=pd.read_csv(reference/'eligible_genes.tsv',sep='\t')
    gid=eligible.gene_id.to_numpy(np.int32);columns=eligible.attribution_column.to_numpy(int)
    global_degree=np.diff(graph.indptr).astype(np.int32)
    inverse=np.full(graph.shape[0],-1,np.int32);inverse[gid]=np.arange(len(gid),dtype=np.int32)
    expression,cell_ids=expression_matrix(info,a.signature_utils)
    expression=expression[:,columns].tocsr();expression.eliminate_zeros()
    cells=pd.read_csv(reference/'cells.tsv',sep='\t',keep_default_na=False)
    assert cells.cell_id.tolist()==cell_ids
    cohort=Path(a.benchmark_path)/'data/cohorts'/ds/'cells.tsv'
    if cohort.exists():
        known=pd.read_csv(cohort,sep='\t',keep_default_na=False).set_index('cell_id')
        for col in ['y','response','split']:cells[col]=cells.cell_id.map(known[col])
    with np.load(reference/'signatures.npz') as data:
        old_ptr=data['ptr'];old_nodes=data['gene_id'];old_features=data['features']
    if a.pilot_cells:
        row_indices=np.unique(np.linspace(0,len(cells)-1,min(a.pilot_cells,len(cells)),dtype=int))
    else:row_indices=np.arange(len(cells))
    out=Path(a.save_path)/ds
    if not a.pilot_cells and not a.rebuild and (out/'signatures.npz').exists() and (out/'parameters.json').exists():
        old=json.loads((out/'parameters.json').read_text())
        for key in ['seed','count_jitter','density_tolerance','degree_cdf_tolerance','isolation_tolerance','max_steps','restarts']:
            assert old[key]==getattr(a,key),(ds,key)
        print('Completed package exists',ds,flush=True);return old
    all_nodes=[];all_features=[];ptr=[0];records=[];plans=[];start=time.perf_counter()
    for position,row in enumerate(row_indices):
        cell_id=cells.cell_id.iloc[row]
        dense=expression[row].toarray().ravel()
        pool=gid[dense>0]
        ref=old_nodes[old_ptr[row]:old_ptr[row+1]].astype(np.int32)
        ref_x=dense[inverse[ref]]
        assert (inverse[ref]>=0).all() and (ref_x>0).all()
        assert np.allclose(ref_x,old_features[old_ptr[row]:old_ptr[row+1],0],rtol=2e-6,atol=1e-6)
        degrees=graph_degrees(graph.indptr,graph.indices,ref,graph.shape[0])
        cell_seed=(zlib.crc32((ds+'|'+cell_id).encode())^a.seed)&0x7fffffff
        rng=np.random.default_rng(cell_seed)
        low=max(1,int(np.floor(len(ref)*(1-a.count_jitter))))
        high=min(len(pool),int(np.ceil(len(ref)*(1+a.count_jitter))))
        low=min(low,high);requested=int(rng.integers(low,high+1))
        best=None;best_score=np.inf;passed=False
        for attempt in range(a.restarts):
            if attempt>=3:requested=int(rng.integers(low,high+1))
            initial=random_initial(pool,ref,global_degree,requested,rng)
            selected,errors,steps,accepted,ok=match_subset(graph.indptr,graph.indices,pool,initial,degrees,len(ref),
                int(rng.integers(0,2**31-1)),a.max_steps*max(1,attempt-1),a.density_tolerance,a.degree_cdf_tolerance,a.isolation_tolerance)
            score=float(errors[0]/a.density_tolerance+errors[1]/a.degree_cdf_tolerance+errors[2]/a.isolation_tolerance)
            if score<best_score:best=(selected.copy(),errors.copy(),steps,accepted,attempt+1);best_score=score
            if ok:
                best=(selected.copy(),errors.copy(),steps,accepted,attempt+1)
                passed=True;break
        selected,errors,steps,accepted,attempt=best
        requested=len(selected)
        selected=np.sort(selected).astype(np.int32)
        assert len(selected)==requested and len(np.unique(selected))==len(selected)
        assert np.isin(selected,pool).all()
        # This control contains measured expression only, never fabricated IG scores.
        feat=dense[inverse[selected]][:,None].astype(np.float32)
        all_nodes.append(selected);all_features.append(feat);ptr.append(ptr[-1]+len(selected))
        records.append(dict(cell_id=cell_id,reference_nodes=len(ref),requested_nodes=requested,selected_nodes=len(selected),
            detected_candidate_genes=len(pool),density_relative_error=float(errors[0]),degree_cdf_distance=float(errors[1]),
            isolation_absolute_error=float(errors[2]),matching_passed=bool(passed),restarts_used=int(attempt),proposals=int(steps),
            original_signature_overlap=int(np.intersect1d(selected,ref,assume_unique=True).size)))
        plans.append((cell_seed,requested,len(ref)))
        if (position+1)%1000==0:print(ds,position+1,'/',len(row_indices),'matched',sum(r['matching_passed'] for r in records),flush=True)
    qc=pd.DataFrame(records)
    print(ds,'matched',int(qc.matching_passed.sum()),'/',len(qc),'seconds',round(time.perf_counter()-start,2),
          'median errors',qc[['density_relative_error','degree_cdf_distance','isolation_absolute_error']].median().to_dict(),flush=True)
    if a.pilot_cells:return None
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'signatures.npz',ptr=np.asarray(ptr,np.int64),gene_id=np.concatenate(all_nodes),features=np.concatenate(all_features))
    cells.to_csv(out/'cells.tsv',sep='\t',index=False)
    eligible.to_csv(out/'eligible_genes.tsv',sep='\t',index=False)
    qc.to_csv(out/'matching_qc.tsv',sep='\t',index=False,float_format='%.17g')
    summary=dict(dataset=ds,n_cells=len(cells),n_signature_nodes=ptr[-1],feature_columns=['lognorm_expression'],
        reference_path=str(reference.resolve()),source_h5ad=info['source_h5ad'],attribution_manifest=info['attribution_manifest'],
        selection='random expressed genes in the same measured, mapped, SCimilarity and common-four-graph vocabulary; no attribution scores or ranks',
        graph_relation='state_15b',n_matching_passed=int(qc.matching_passed.sum()),
        hard_cell_search='after three attempts redraw count within the same bounds; step budget increases deterministically; never initialize from original signature',
        **{key:getattr(a,key) for key in ['seed','count_jitter','density_tolerance','degree_cdf_tolerance','isolation_tolerance','max_steps','restarts']})
    write_json(out/'parameters.json',summary)
    plan=Path(a.graph_path)/ds;plan.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(plan/'sampling_plan.npz',cell_seed=np.asarray(plans,np.int64)[:,0],
        requested_nodes=np.asarray(plans,np.int64)[:,1],reference_nodes=np.asarray(plans,np.int64)[:,2])
    write_json(plan/'graph.json',dict(adjacency=str((Path(a.graph_path)/'state_15b.npz').resolve()),
        node_order=str((Path(a.graph_path)/'gene_vocab.tsv').resolve()),
        cell_nodes=str((out/'signatures.npz').resolve()),cell_ids=str((out/'cells.tsv').resolve()),
        representation='one fixed global weighted STATE graph; induce each cell graph on its filtered random nodes when loading; edges are not rewired'))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference_path','graph_path','save_path','benchmark_path','signature_utils']:p.add_argument('--'+name,required=True)
    p.add_argument('--datasets',nargs='+',default=DATASETS,help='Cohort folders in the reference graph package')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--count_jitter',type=float,default=.05)
    p.add_argument('--density_tolerance',type=float,default=.02);p.add_argument('--degree_cdf_tolerance',type=float,default=.05)
    p.add_argument('--isolation_tolerance',type=float,default=.02);p.add_argument('--max_steps',type=int,default=2000)
    p.add_argument('--rebuild',action='store_true',help='Regenerate this ablation package only; original data stay read-only')
    p.add_argument('--restarts',type=int,default=12);p.add_argument('--pilot_cells',type=int,default=0)
    a=p.parse_args();reference=Path(a.reference_path);out=Path(a.save_path);graphs=Path(a.graph_path)
    matrix=sparse.load_npz(reference/'global_graphs/state_15b.npz').tocsr()
    genes=pd.read_csv(reference/'gene_vocab.tsv',sep='\t')
    if not a.pilot_cells:
        graphs.mkdir(parents=True,exist_ok=True);(out/'global_graphs').mkdir(parents=True,exist_ok=True)
        for dest in [graphs/'state_15b.npz',out/'global_graphs/state_15b.npz']:
            shutil.copyfile(reference/'global_graphs/state_15b.npz',dest)
        for dest in [graphs/'gene_vocab.tsv',out/'gene_vocab.tsv']:shutil.copyfile(reference/'gene_vocab.tsv',dest)
    summaries=[]
    for ds in a.datasets:
        result=build_dataset(a,ds,matrix,genes)
        if result:summaries.append(result)
    if not a.pilot_cells:
        write_json(out/'summary.json',dict(datasets=summaries,graph_path=str(graphs.resolve()),
          scientific_scope='cell-level structure-matched random-signature control; no response labels used for sampling or matching; reference graph summaries are inherited from original signatures'))


if __name__=='__main__':main()

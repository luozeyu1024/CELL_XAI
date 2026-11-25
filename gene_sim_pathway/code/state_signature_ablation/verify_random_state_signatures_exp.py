"""Verify all filtered random STATE cell graphs and independently check expression."""
import argparse,json,zlib
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path',required=True);p.add_argument('--reference_path',required=True);p.add_argument('--graph_path',required=True)
    a=p.parse_args();root=Path(a.open_path);reference=Path(a.reference_path);graphs=Path(a.graph_path)
    graph=sparse.load_npz(root/'global_graphs/state_15b.npz').tocsr()
    original=sparse.load_npz(reference/'global_graphs/state_15b.npz').tocsr()
    assert (graph-original).nnz==0 and (graph-sparse.load_npz(graphs/'state_15b.npz')).nnz==0
    assert (graph-graph.T).nnz==0 and np.isfinite(graph.data).all()
    vocab=pd.read_csv(root/'gene_vocab.tsv',sep='\t');checks=[]
    assert vocab.equals(pd.read_csv(reference/'gene_vocab.tsv',sep='\t'))
    for ds in json.loads((root/'summary.json').read_text())['datasets']:
        name=ds['dataset'];folder=root/name;old=reference/name
        cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False)
        assert cells.cell_id.tolist()==pd.read_csv(old/'cells.tsv',sep='\t').cell_id.tolist()
        eligible=pd.read_csv(folder/'eligible_genes.tsv',sep='\t')
        assert eligible.equals(pd.read_csv(old/'eligible_genes.tsv',sep='\t'))
        allowed=np.zeros(graph.shape[0],bool);allowed[eligible.gene_id]=True
        with np.load(folder/'signatures.npz') as z:ptr=z['ptr'];ids=z['gene_id'];features=z['features']
        with np.load(old/'signatures.npz') as z:oldptr=z['ptr'];oldids=z['gene_id']
        assert features.shape==(len(ids),1) and np.isfinite(features).all() and (features>0).all()
        assert allowed[ids].all() and len(ptr)==len(cells)+1 and ptr[0]==0 and ptr[-1]==len(ids)
        qc=pd.read_csv(folder/'matching_qc.tsv',sep='\t',float_precision='round_trip')
        assert qc.cell_id.tolist()==cells.cell_id.tolist()
        with np.load(graphs/name/'sampling_plan.npz') as z:
            assert np.array_equal(z['requested_nodes'],np.diff(ptr)) and np.array_equal(z['reference_nodes'],np.diff(oldptr))
            assert z['cell_seed'].tolist()==[(zlib.crc32((name+'|'+c).encode())^ds['seed'])&0x7fffffff for c in cells.cell_id]
        n_passed=0
        for row in range(len(cells)):
            selected=ids[ptr[row]:ptr[row+1]];ref=oldids[oldptr[row]:oldptr[row+1]]
            assert len(selected)==len(np.unique(selected)) and len(selected)>0
            assert max(1,int(np.floor(len(ref)*(1-ds['count_jitter']))))<=len(selected)<=int(np.ceil(len(ref)*(1+ds['count_jitter'])))
            degree=np.diff(graph[selected][:,selected].indptr);refdegree=np.diff(graph[ref][:,ref].indptr)
            denom=len(degree)*(len(degree)-1);refdenom=len(refdegree)*(len(refdegree)-1)
            expected_edges=refdegree.sum()/2*denom/max(refdenom,1)
            density_error=abs(degree.sum()/2-expected_edges)/max(expected_edges,1)
            length=max(degree.max(initial=0),refdegree.max(initial=0))+1
            cdf=np.cumsum(np.bincount(degree,minlength=length))/len(degree)
            target=np.cumsum(np.bincount(refdegree,minlength=length))/len(refdegree)
            distance=np.abs(cdf-target).max();isolation=abs((degree==0).mean()-(refdegree==0).mean())
            actual=np.array([density_error,distance,isolation]);saved=qc.iloc[row]
            assert np.allclose(actual,[saved.density_relative_error,saved.degree_cdf_distance,saved.isolation_absolute_error],atol=1e-10,rtol=1e-10)
            passed=density_error<=ds['density_tolerance']+1e-12 and distance<=ds['degree_cdf_tolerance']+1e-12 and isolation<=ds['isolation_tolerance']+1e-12
            assert passed==saved.matching_passed
            n_passed+=passed
        info=json.loads((old/'parameters.json').read_text())
        manifest=json.loads(Path(info['attribution_manifest']).read_text())['input']
        names=json.loads((Path(info['attribution_manifest']).parents[1]/'attributions/scimilarity_ig_var_names.json').read_text())
        raw=ad.read_h5ad(info['source_h5ad']);raw.var_names_make_unique()
        lookup=pd.Index(raw.var_names).get_indexer(names);model_present=lookup[lookup>=0]
        source_rows=np.asarray(manifest['cell_indices_used'],int)
        counts=sparse.csr_matrix(raw.layers['counts'] if 'counts' in raw.layers else raw.X,dtype=np.float32)
        column_map=np.full(graph.shape[0],-1,int);column_map[eligible.gene_id]=lookup[eligible.attribution_column]
        chosen=np.unique(np.linspace(0,len(cells)-1,min(16,len(cells)),dtype=int))
        for row in chosen:
            selected=ids[ptr[row]:ptr[row+1]];columns=column_map[selected]
            assert (columns>=0).all()
            total=float(counts[source_rows[row],model_present].sum())
            values=counts[source_rows[row],columns].toarray().ravel().astype(np.float64)
            assert (values>0).all()
            expected=np.log1p(values*10000/total)
            assert np.allclose(expected,features[ptr[row]:ptr[row+1],0],rtol=3e-6,atol=2e-6)
        del raw,counts
        checks.append(dict(dataset=name,n_cells=len(cells),n_matching_passed=int(n_passed),n_expression_cells_checked=len(chosen),status='passed' if n_passed==len(cells) else 'unmatched_cells'))
        print('VERIFY',checks[-1],flush=True)
    result=dict(status='passed' if all(c['status']=='passed' for c in checks) else 'incomplete_matching',checks=checks)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['status']=='passed',result


if __name__=='__main__':main()

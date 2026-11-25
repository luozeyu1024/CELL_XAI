"""Check every cell's signature and four induced graphs against saved sources."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from cell_graph_dataset import CellGraphDataset, RELATIONS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path',required=True)
    args=p.parse_args(); root=Path(args.open_path)
    genes=pd.read_csv(root/'gene_vocab.tsv',sep='\t').gene_symbol.to_numpy()
    n=len(genes)
    graphs={r:sparse.load_npz(root/'global_graphs'/f'{r}.npz').tocsr() for r in RELATIONS}
    with np.load(root/'global_graphs/coverage.npz') as z:
        common=np.logical_and.reduce([z[r] for r in RELATIONS])
    for r,g in graphs.items():
        assert g.shape==(n,n) and np.isfinite(g.data).all()
        assert not np.any(g.diagonal()) and (g!=g.T).nnz==0
        assert np.all(g.data<=1) and np.all(g.data>=(-1 if r!='ppi' else .5))
    report=[]
    for folder in sorted(root.iterdir()):
        if not (folder/'signatures.npz').exists(): continue
        params=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
        manifest=json.loads(Path(params['attribution_manifest']).read_text(encoding='utf-8'))
        cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False)
        assert cells.cell_id.tolist()==manifest['input']['cell_ids_used']
        assert cells.source_row.tolist()==manifest['input']['cell_indices_used']
        assert cells.y.eq(-1).all()
        with np.load(folder/'signatures.npz') as z:
            ptr=z['ptr']; ids=z['gene_id']; feat=z['features']
        assert ptr[0]==0 and ptr[-1]==len(ids)==len(feat) and len(ptr)==len(cells)+1
        assert ids.dtype==np.int32 and feat.dtype==np.float32 and feat.shape[1]==3
        assert np.isfinite(feat).all() and (feat>=0).all()
        assert np.all(np.diff(ptr)>=0) and np.all(np.diff(ptr)<=params['top_k'])
        assert common[ids].all()
        eligible=pd.read_csv(folder/'eligible_genes.tsv',sep='\t')
        qc=pd.read_csv(folder/'signature_qc.tsv',sep='\t')
        assert qc.cell_id.tolist()==cells.cell_id.tolist()
        attr_dir=Path(params['attribution_manifest']).parents[1]/'attributions'
        attr=sparse.load_npz(attr_dir/'scimilarity_ig.npz').tocsr()
        feature_names=json.loads((attr_dir/'scimilarity_ig_var_names.json').read_text(encoding='utf-8'))
        gid_to_col=dict(zip(eligible.gene_id,eligible.attribution_column))
        # Direct expression reconstruction for evenly spaced cells; no official
        # preprocessing helper is called here, giving an independent formula check.
        import anndata as ad
        raw=ad.read_h5ad(params['source_h5ad'])
        raw.var_names_make_unique()
        raw_index={s:i for i,s in enumerate(raw.var_names)}
        model_raw=[raw_index[s] for s in feature_names if s in raw_index]
        sample=set(np.linspace(0,len(cells)-1,min(16,len(cells)),dtype=int).tolist())
        for c in range(len(cells)):
            lo,hi=ptr[c:c+2]; selected=ids[lo:hi]; features=feat[lo:hi]; k=hi-lo
            assert len(set(selected))==k
            scores=attr[c,eligible.attribution_column.to_numpy()].toarray().ravel()
            # Independent lexsort ranking rather than the builder's stable argsort.
            ranking=np.lexsort((eligible.gene_symbol.to_numpy(),-scores))
            ranking=ranking[scores[ranking]>0][:params['top_k']]
            assert np.array_equal(selected,eligible.gene_id.to_numpy()[ranking])
            assert np.allclose(features[:,1],scores[ranking]/1000,atol=1e-7)
            assert np.allclose(features[:,2],np.arange(k)/max(k-1,1),atol=1e-7)
            total=float(attr[c].sum())
            record=qc.iloc[c]
            assert record.n_nodes==k
            if total:
                assert np.isclose(record.selected_importance_fraction,scores[ranking].sum()/total,atol=1e-6)
                assert np.isclose(record.eligible_importance_fraction,scores.sum()/total,atol=1e-6)
            for relation,g in graphs.items():
                sub=g[selected][:,selected]
                assert record[f'{relation}_undirected_edges']==sub.nnz//2
                if k:
                    _,labels=connected_components(sub,directed=False)
                    assert np.isclose(record[f'{relation}_isolated_fraction'],np.mean(np.diff(sub.indptr)==0))
                    assert np.isclose(record[f'{relation}_largest_component_fraction'],np.bincount(labels).max()/k)
            if c in sample and k:
                row=raw[int(cells.iloc[c].source_row):int(cells.iloc[c].source_row)+1,:].copy()
                counts=sparse.csr_matrix(row.layers['counts'] if 'counts' in row.layers else row.X)
                denominator=float(counts[:,model_raw].sum())
                columns=[raw_index[feature_names[gid_to_col[v]]] for v in selected]
                expected=np.log1p(counts[:,columns].toarray().ravel()*10000/denominator)
                assert np.allclose(features[:,0],expected,atol=2e-5),folder.name
        raw.file.close()
        for relation in RELATIONS:
            loader=CellGraphDataset(root,folder.name,relation)
            for c in sample:
                item=loader[c]; lo,hi=ptr[c:c+2]
                assert item['cell_id']==cells.iloc[c].cell_id
                assert np.array_equal(item['x'],feat[lo:hi])
                coo=graphs[relation][ids[lo:hi]][:,ids[lo:hi]].tocoo()
                assert np.array_equal(item['edge_index'],np.stack([coo.row,coo.col]))
                assert np.array_equal(item['edge_attr'],coo.data)
        result={'dataset':folder.name,'cells_checked':len(cells),
                'graph_views_checked':len(cells)*4,'expression_cells_checked':len(sample),
                'status':'passed'}
        report.append(result); print(result,flush=True)
    (root/'validation.json').write_text(json.dumps({'datasets':report,'status':'passed'},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()

"""Count validation and HGNC mapping shared by the cohort converters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

METADATA_ROOT=Path("results/intermediate")


def metadata_folder(folder):
    return METADATA_ROOT/Path(folder).name


def write_counts(data, folder, source):
    folder.mkdir(parents=True,exist_ok=True)
    metadata=metadata_folder(folder);metadata.mkdir(parents=True,exist_ok=True)
    data.obs.index.name=None;data.var.index.name=None
    if not data.obs_names.is_unique:
        raise ValueError('Duplicate cell identifiers')
    data.X=sparse.csr_matrix(data.X,dtype=np.float32)
    if not np.isfinite(data.X.data).all() or np.any(data.X.data<0) or np.any(data.X.data!=np.rint(data.X.data)):
        raise ValueError('Input is not nonnegative integer counts')
    mapping_info={}
    if any(str(name).startswith('ENSG') for name in data.var_names):
        hgnc=Path(__file__).resolve().parents[2]/'resources/hgnc_gene_mapping.tsv'
        table=pd.read_csv(hgnc,sep='\t',dtype=str,keep_default_na=False)
        table=table[(table.status=='Approved')&table.ensembl_gene_id.ne('')]
        groups=table.groupby('ensembl_gene_id').symbol.agg(lambda s:sorted(set(s)))
        lookup={key:value[0] for key,value in groups.items() if len(value)==1}
        original=data.var_names.astype(str).tolist()
        mapped=[lookup.get(name.split('.')[0],name) if name.startswith('ENSG') else name for name in original]
        pd.DataFrame({'source_gene_id':original,'analysis_gene_symbol':mapped}).to_csv(metadata/'gene_mapping.tsv',sep='\t',index=False)
        data.var_names=pd.Index(mapped)
        mapping_info={'hgnc_path':str(hgnc),'n_ensembl_mapped':sum(a!=b for a,b in zip(original,mapped)),
                      'unmapped_policy':'retain original identifier; no ambiguous symbol assignment'}
    if not data.var_names.is_unique:
        names=pd.Index(data.var_names)
        unique=pd.Index(names.unique())
        mapping=sparse.csr_matrix((np.ones(len(names)),(np.arange(len(names)),unique.get_indexer(names))),shape=(len(names),len(unique)))
        data=ad.AnnData(X=(data.X@mapping).astype(np.float32),obs=data.obs.copy(),var=pd.DataFrame(index=unique))
    data.obs['Cell_ID']=data.obs_names.astype(str)
    temporary=folder/'counts.pending.h5ad'
    data.write_h5ad(temporary,compression='gzip')
    temporary.replace(folder/'counts.h5ad')
    data.obs.to_csv(metadata/'counts_labels.tsv',sep='\t',index=False)
    record=dict(n_cells=data.n_obs,n_genes=data.n_vars,all_nonnegative_integer=True,
                unique_cells=True,unique_genes=True,source=str(source),counts_h5ad=str(folder/'counts.h5ad'),
                original_gene_duplicates='sum counts within identical analysis gene symbol',**mapping_info)
    (metadata/'counts_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('COUNTS_VALIDATED',folder.name,record,flush=True)
    return data



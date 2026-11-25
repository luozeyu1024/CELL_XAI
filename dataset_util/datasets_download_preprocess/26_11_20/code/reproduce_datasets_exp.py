"""Download, reconstruct and verify the frozen 14-design dataset collection."""
import argparse
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile
import sys
import shutil
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.io import mmread

BUNDLE=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
CONFIG=json.loads((BUNDLE/'datasets.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=1)
def cohort_axes():
    with gzip.open(BUNDLE/'resources/cohort_axes.json.gz','rt',encoding='utf-8') as stream:
        return json.load(stream)


def load(name):
    spec=importlib.util.spec_from_file_location(name,BUNDLE/'code/reference'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def cell_table():
    return pd.read_csv(BUNDLE/'resources/cohort_cells.tsv.gz',sep='\t',keep_default_na=False,dtype=str)


def reference_cells(dataset):
    frame=cell_table()
    return frame[frame.dataset_id.eq(dataset)].set_index('cell_id',drop=False)


def numeric_counts(matrix):
    matrix=sparse.csr_matrix(matrix)
    x=matrix.data
    if not np.isfinite(x).all() or (x<0).any() or (x!=np.rint(x)).any():
        raise ValueError('Expected finite nonnegative integer counts')
    return matrix


def tabular_handle(handle,sep='\t',source='stream'):
    blocks=[];genes=[];cells=None
    for block in pd.read_csv(handle,sep=sep,index_col=0,chunksize=256,keep_default_na=False):
        if cells is None:cells=block.columns.astype(str)
        blocks.append(numeric_counts(sparse.csr_matrix(block.to_numpy(dtype=np.float32))))
        genes.extend(block.index.astype(str))
    return ad.AnnData(sparse.vstack(blocks,format='csr').T.tocsr(),obs=pd.DataFrame(index=cells),var=pd.DataFrame(index=genes))


def tabular_genes(path,sep='\t'):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:return tabular_handle(stream,sep,str(path))


def finish(data,dataset,args):
    """Enforce the historical cohort, without padding missing cells or genes."""
    expected=cohort_axes()[dataset]
    if not data.obs_names.is_unique or not data.var_names.is_unique:raise ValueError('Duplicate IDs: '+dataset)
    if set(data.obs_names)!=set(expected['cell_ids']):raise ValueError('Cell selection differs: '+dataset)
    if set(data.var_names)!=set(expected['gene_ids']):raise ValueError('Gene mapping differs: '+dataset)
    data=data[expected['cell_ids'],expected['gene_ids']].copy()
    data.obs.index.name=None;data.var.index.name=None
    data.X=numeric_counts(data.X).astype(np.float32)
    reference=reference_cells(dataset).loc[data.obs_names]
    # Compare derived biological labels before attaching the frozen evaluation fields.
    if 'label' in data.obs and 'label' in reference:
        expected_label=reference.label.where(reference.label.ne(''),reference.response)
        mask=expected_label.ne('')
        if not np.array_equal(data.obs.loc[mask.to_numpy(),'label'].astype(str).to_numpy(),expected_label[mask].astype(str).to_numpy()):
            raise ValueError('Derived labels differ: '+dataset)
    if 'state_y' in data.obs and 'state_y' in reference:
        if not np.array_equal(pd.to_numeric(data.obs.state_y).astype(int),pd.to_numeric(reference.state_y).astype(int)):
            raise ValueError('Derived treatment-state labels differ: '+dataset)
    for key in reference:
        if key not in ['dataset_id','cell_id','graph_row','source_row']:data.obs[key]=reference[key].to_numpy()
    data.obs['Cell_ID']=data.obs_names.astype(str)
    if args.check_only:
        if args.reference_path is not None:
            original=ad.read_h5ad(args.reference_path/dataset/'counts.h5ad',backed='r')
            if original.obs_names.tolist()!=data.obs_names.tolist() or original.var_names.tolist()!=data.var_names.tolist():
                raise ValueError('Reference identifiers differ: '+dataset)
            for start in range(0,data.n_obs,256):
                difference=data.X[start:start+256]-sparse.csr_matrix(original.X[start:start+256]);difference.eliminate_zeros()
                if difference.nnz:raise ValueError('Reconstructed count values differ: '+dataset)
            original.file.close()
        args.checked.append(dict(dataset_id=dataset,n_cells=data.n_obs,n_genes=data.n_vars,
                                 complete_count_matrix_equal=True if args.reference_path is not None else None,
                                 ordered_identifiers_equal=True,derived_labels_equal=True))
        args.metadata_path.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(args.checked).to_csv(args.metadata_path/'raw_reconstruction_audit.tsv',sep='\t',index=False)
        print('RECONSTRUCTION_VALIDATED',dataset,data.shape,flush=True)
        return
    folder=args.save_path/dataset;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'counts.h5ad'
    if path.exists():raise FileExistsError('Use a new --save_path; existing counts are protected: '+str(path))
    pending=folder/'counts.pending.h5ad'
    if pending.exists():raise FileExistsError('Incomplete output exists; inspect before rerunning: '+str(pending))
    data.write_h5ad(pending,compression='gzip')
    pending.rename(path)
    meta=args.metadata_path/dataset;meta.mkdir(parents=True,exist_ok=True)
    data.obs.to_csv(meta/'counts_labels.tsv',sep='\t',index=False)
    (meta/'counts_validation.json').write_text(json.dumps(dict(dataset_id=dataset,n_cells=data.n_obs,n_genes=data.n_vars,
        all_nonnegative_integer=True,cell_and_gene_order_match_reference=True,counts_h5ad=str(path)),indent=2)+'\n',encoding='utf-8')
    print('RECONSTRUCTED',dataset,data.shape,flush=True)


def tenx_member(archive,names,prefix,feature_suffix):
    def table(suffix):
        return pd.read_csv(gzip.GzipFile(fileobj=archive.extractfile(names[prefix+suffix])),sep='\t',header=None,dtype=str)
    genes=table(feature_suffix);barcodes=table('_barcodes.tsv.gz')[0].astype(str)
    with gzip.GzipFile(fileobj=archive.extractfile(names[prefix+'_matrix.mtx.gz'])) as stream:x=mmread(stream).T.tocsr()
    symbols=genes.iloc[:,1 if genes.shape[1]>1 else 0].astype(str)
    return x,barcodes,symbols


def prostate(args):
    rows=[];ids=[];gene_names=None;source_ids=None
    with tarfile.open(args.open_path/'GSE140440/GSE140440_RAW.tar') as archive:
        for member in sorted((m for m in archive.getmembers() if m.isfile()),key=lambda m:m.name):
            with gzip.GzipFile(fileobj=archive.extractfile(member)) as stream:
                table=pd.read_csv(stream,sep='\t',header=None,dtype={0:str})
            names=table[0].str.split('|',n=1).str[-1].tolist()
            if gene_names is None:gene_names=names;source_ids=table[0].astype(str).tolist()
            if names!=gene_names:raise ValueError('Different raw feature order across prostate cells')
            rows.append(sparse.csr_matrix(table[1].to_numpy(dtype=np.float32)[None,:]));ids.append(Path(member.name).name.split('_')[0])
    data=ad.AnnData(sparse.vstack(rows,format='csr'),obs=pd.DataFrame(index=ids),
                   var=pd.DataFrame({'source_feature_id':source_ids,'full_gene_symbol':gene_names},index=[g.split('-')[0] for g in gene_names]))
    # Historical scATD feature names truncate at a hyphen, then reserve existing suffixes.
    data.var_names_make_unique(join='_')
    meta=args.metadata_path/'GSE140440';meta.mkdir(parents=True,exist_ok=True)
    data.var.assign(analysis_feature=data.var_names).to_csv(meta/'feature_name_map.tsv.gz',sep='\t',index=False,compression='gzip')
    conditions=pd.read_csv(BUNDLE/'resources/sample_conditions.tsv',sep='\t',keep_default_na=False).set_index('sample_id')
    data.obs['label']=conditions.loc[data.obs_names,'docetaxel'].str.lower().to_numpy()
    finish(data,'GSE140440',args)


def alk(dataset,args,utility):
    samples=['fa34os','fa34oc'] if dataset=='GSE223779' else ['ts485tos','ts485toc','ts485toa']
    parts=[]
    with tarfile.open(args.open_path/'GSE223779_expand/GSE223779_RAW.tar') as archive:
        names={Path(m.name).name:m for m in archive.getmembers() if m.isfile()}
        for sid in samples:
            matches=[n[:-len('_matrix.mtx.gz')] for n in names if n.endswith('_matrix.mtx.gz') and n.split('_')[1].lower()==sid]
            if len(matches)!=1:raise ValueError('Ambiguous organoid sample: '+sid)
            x,barcodes,genes=tenx_member(archive,names,matches[0],'_features.tsv.gz')
            obs=pd.DataFrame(index=pd.Index(barcodes+'-'+sid))
            obs['label']='sensitive' if sid.endswith('os') else 'resistant'
            part=ad.AnnData(x,obs=obs,var=pd.DataFrame(index=genes.to_numpy()))
            if dataset=='GSE223779':
                # Historical scATD input preserves all 33,538 features with unique suffixes.
                part.var_names_make_unique()
            else:part=utility.write_counts(part,args.save_path/'intermediate'/('GSE223779_'+sid),'GSE223779_RAW.tar')
            parts.append(part)
    finish(ad.concat(parts,join='outer',fill_value=0,index_unique=None),dataset,args)


def ovarian_source(args,utility):
    folder=args.open_path/'GSE165897'
    labels=pd.read_csv(folder/'GSE165897_cellInfo_HGSOC.tsv.gz',sep='\t',keep_default_na=False).set_index('cell')
    selected=set(labels.index[labels.cell_type.eq('EOC')&labels.treatment_phase.eq('treatment-naive')])
    source=folder/'GSE165897_UMIcounts_HGSOC.tsv.gz'
    header=pd.read_csv(source,sep='\t',nrows=0).columns
    columns=[0]+[i for i,name in enumerate(header) if name in selected]
    blocks=[];genes=[];cells=None
    # Select cells before numeric parsing; preserve every original gene row, including zeros.
    for block in pd.read_csv(source,sep='\t',usecols=columns,index_col=0,chunksize=256,dtype={name:'float32' for name in selected}):
        if cells is None:cells=block.columns.astype(str)
        blocks.append(sparse.csr_matrix(block.to_numpy(dtype=np.float32)));genes.extend(block.index.astype(str))
    data=ad.AnnData(sparse.vstack(blocks,format='csr').T.tocsr(),obs=pd.DataFrame(index=cells),var=pd.DataFrame(index=genes))
    data.obs=labels.loc[data.obs_names].copy()
    root=ET.parse(BUNDLE/'resources/clinical_table1.xml').getroot()
    table=root if root.tag=='table-wrap' and root.get('id')=='T1' else root.find('.//table-wrap[@id="T1"]')
    if table is None:raise ValueError('Clinical follow-up records are missing')
    patients=[]
    for row in table.findall('.//tbody/tr'):
        values=[''.join(cell.itertext()).strip() for cell in row.findall('td')]
        if len(values)!=13:raise ValueError('Unexpected clinical table layout')
        patients.append(dict(patient_id=values[0],pfi_days=int(re.match(r'\d+',values[4]).group())))
    clinical=pd.DataFrame(patients).set_index('patient_id')
    saved=pd.read_csv(BUNDLE/'resources/patient_clinical.tsv',sep='\t').set_index('patient_id')
    if clinical.loc[saved.index,'pfi_days'].tolist()!=saved.pfi_days.tolist():raise ValueError('PFI table/snapshot conflict')
    if data.obs.patient_id.map(clinical.pfi_days).isna().any():raise ValueError('Missing original PFI evidence')
    data.obs['label']=np.where(data.obs.patient_id.map(clinical.pfi_days).lt(180),'resistant','sensitive')
    data=data[data.obs.cell_type.eq('EOC')&data.obs.treatment_phase.eq('treatment-naive')].copy()
    finish(data,'GSE165897_pre_EOC',args)


def resistrace(dataset,args,utility):
    drug={'carboplatin':'Carboplatin','olaparib':'Olaparib','nk':'NK'}[dataset.split('_')[1]]
    parts=[]
    with tarfile.open(args.open_path/'GSE223003/GSE223003_RAW.tar') as archive:
        names={Path(m.name).name:m for m in archive.getmembers() if m.isfile()}
        samples=sorted(n[:-len('_UMIcounts.tsv.gz')] for n in names if n.endswith(f'_ReSisTrace_{drug}_1_pre_treatment_UMIcounts.tsv.gz') or n.endswith(f'_ReSisTrace_{drug}_2_pre_treatment_UMIcounts.tsv.gz'))
        if len(samples)!=2:raise ValueError('Expected both pretreatment replicates')
        for sample in samples:
            with gzip.GzipFile(fileobj=archive.extractfile(names[sample+'_UMIcounts.tsv.gz'])) as stream:
                part=utility.tabular_handle(io.TextIOWrapper(stream,encoding='utf-8'),source=sample)
            with gzip.GzipFile(fileobj=archive.extractfile(names[sample+'_cell_info.tsv.gz'])) as stream:
                labels=pd.read_csv(stream,sep='\t',index_col=0,keep_default_na=False)
            part.obs=labels.loc[part.obs_names].copy()
            part.obs['label']=part.obs.drugSens.map({'pre-sensitive':'sensitive','pre-resistant':'resistant'}).fillna('unknown')
            part.obs_names=pd.Index(sample+'|'+part.obs_names.astype(str))
            part=utility.write_counts(part,args.save_path/'intermediate'/('GSE223003_'+sample.split('_ReSisTrace_')[1]),sample)
            parts.append(part)
    data=ad.concat(parts,join='outer',fill_value=0,index_unique=None)
    finish(data[data.obs.label.isin(['sensitive','resistant'])].copy(),dataset,args)


def pc9(args,utility):
    task='GSE149383_PC9_DTEP_sample_grouped';parts=[]
    resources=CONFIG['download_resources']['GSE149383']
    for sid in ['GSM3972669','GSM4494347','GSM4494348','GSM4494349']:
        path=args.open_path/next(r['path'] for r in resources if Path(r['path']).name.startswith(sid+'_'))
        part=utility.tabular_genes(path)
        part.obs_names=pd.Index(sid+'|'+part.obs_names.astype(str))
        # Identical HGNC mapping and duplicate summation precede QC in the original workflow.
        part=utility.write_counts(part,args.save_path/'intermediate'/('GSE149383_expand_'+sid),path)
        x=part.X.tocsr();totals=np.asarray(x.sum(axis=1)).ravel();ngenes=np.asarray((x>0).sum(axis=1)).ravel()
        mt=np.asarray(x[:,part.var_names.str.upper().str.startswith('MT-')].sum(axis=1)).ravel()
        pct=np.divide(100*mt,totals,out=np.zeros(len(totals)),where=totals>0)
        part=part[(ngenes>=200)&(pct<=20)&(totals>0)].copy()
        part.obs['label']='unknown';part.obs['state_y']=int(sid in ['GSM4494348','GSM4494349'])
        parts.append(part)
    finish(ad.concat(parts,join='outer',fill_value=0,index_unique=None),task,args)


def ovarian_external(args,utility):
    parts=[]
    lookup={'GSM4675273':('59','resistant'),'GSM4675274':('76','unknown'),
            'GSM4675275':('77','resistant'),'GSM4675276':('89','sensitive'),'GSM4675277':('90','sensitive')}
    with tarfile.open(args.open_path/'GSE154600/GSE154600_RAW.tar') as archive:
        names={Path(m.name).name:m for m in archive.getmembers() if m.isfile()}
        for prefix in sorted(n[:-len('_matrix.mtx.gz')] for n in names if n.endswith('_matrix.mtx.gz')):
            sid=prefix.split('_')[0];author,response=lookup[sid]
            annotation=BUNDLE/'resources'/f'author_T{author}.tsv'
            if args.refresh_author_annotations:
                annotation=args.metadata_path/f'author_T{author}.tsv';annotation.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run([args.r_path,str(BUNDLE/'code/reference/export_ovarian_annotation_exp.R'),
                    str(args.open_path/f'GSE154600/author_annotations/sample{author}_sce.rds'),str(annotation)],check=True)
            labels=pd.read_csv(annotation,sep='\t',keep_default_na=False)
            x,barcodes,genes=tenx_member(archive,names,prefix,'_genes.tsv.gz')
            part=ad.AnnData(x,obs=pd.DataFrame(index=pd.Index(sid+'|'+barcodes)),var=pd.DataFrame(index=genes.to_numpy()))
            part.obs['label']=response
            part=utility.write_counts(part,args.save_path/'intermediate'/('GSE154600_'+sid),prefix)
            selected=sid+'|'+labels.loc[labels.author_celltype.eq('EPI'),'Barcode'].astype(str)
            parts.append(part[selected].copy())
    finish(ad.concat(parts,join='outer',fill_value=0,index_unique=None),'GSE154600_EPI_external',args)


def external(dataset,args):
    if dataset=='GSE163836':
        scratch=args.save_path/'intermediate/GSE163836';scratch.mkdir(parents=True,exist_ok=True)
        meta=args.metadata_path/'GSE163836';meta.mkdir(parents=True,exist_ok=True)
        source=args.open_path/'GSE163836/GSE163836_IBC_sc10x_raw_counts.RData.gz'
        subprocess.run([args.r_path,str(BUNDLE/'code/reference/export_gse163836_labels.R'),str(source),str(meta)],check=True)
        subprocess.run([args.r_path,str(BUNDLE/'code/reference/export_external_counts.R'),str(source),str(scratch)],check=True)
        labels=pd.read_csv(meta/'cell_labels.tsv',sep='\t',keep_default_na=False).set_index('Cell_ID')
        ids=pd.read_csv(scratch/'cells.tsv',header=None)[0].astype(str)
        genes=pd.read_csv(scratch/'genes.tsv',header=None)[0].astype(str)
        data=ad.AnnData(mmread(scratch/'counts.mtx').T.tocsr(),obs=labels.loc[ids].copy(),var=pd.DataFrame(index=genes))
    else:
        parts=[]
        with tarfile.open(args.open_path/'GSE202234/GSE202234_RAW.tar') as archive:
            for member in archive.getmembers():
                if not member.isfile():continue
                sid=Path(member.name).name.split('_')[0]
                with zipfile.ZipFile(io.BytesIO(archive.extractfile(member).read())) as inner:
                    def content(suffix):return inner.read(next(n for n in inner.namelist() if n.endswith(suffix)))
                    x=mmread(io.BytesIO(content('matrix.mtx'))).T.tocsr()
                    ids=content('barcodes.tsv').decode().splitlines()
                    genes=pd.read_csv(io.BytesIO(content('genes.tsv')),sep='\t',header=None,dtype=str).iloc[:,-1]
                    unique=pd.Index(genes).drop_duplicates()
                    mapping=sparse.csr_matrix((np.ones(len(genes)),(np.arange(len(genes)),unique.get_indexer(genes))),shape=(len(genes),len(unique)))
                    obs=pd.DataFrame(index=[sid+'_'+c for c in ids])
                    obs['label']='resistant' if sid in ['GSM6106784','GSM6106785'] else ('sensitive' if sid in ['GSM6106786','GSM6106787'] else 'unknown')
                    parts.append(ad.AnnData((x@mapping).tocsr(),obs=obs,var=pd.DataFrame(index=unique)))
        data=ad.concat(parts,join='inner',merge='same')
    finish(data,dataset,args)


def verify(dataset,args):
    meta=CONFIG['datasets'][dataset]
    path=args.save_path/dataset/'counts.h5ad'
    data=ad.read_h5ad(path,backed='r')
    if data.shape!=(meta['n_cells'],meta['n_genes']):raise ValueError('Shape mismatch: '+dataset)
    axes=cohort_axes()[dataset]
    if data.obs_names.tolist()!=axes['cell_ids'] or data.var_names.tolist()!=axes['gene_ids']:raise ValueError('Ordered identifiers differ: '+dataset)
    original=ad.read_h5ad(args.reference_path/dataset/'counts.h5ad',backed='r') if args.reference_path is not None else None
    if original is not None and (original.obs_names.tolist()!=data.obs_names.tolist() or original.var_names.tolist()!=data.var_names.tolist()):
        raise ValueError('Reference identifiers differ: '+dataset)
    for start in range(0,data.n_obs,256):
        block=numeric_counts(data.X[start:start+256])
        if original is not None:
            difference=block-sparse.csr_matrix(original.X[start:start+256]);difference.eliminate_zeros()
            if difference.nnz:raise ValueError('Counts differ from supplied reference: '+dataset)
    if original is not None:original.file.close()
    data.file.close()
    return dict(dataset_id=dataset,counts_path=str(path),n_cells=meta['n_cells'],n_genes=meta['n_genes'],
        integer_counts_verified=True,ordered_cell_and_gene_ids_verified=True,
        all_count_values_equal_reference=True if args.reference_path is not None else None)


def splits(args):
    table=pd.read_csv(BUNDLE/'resources/source_splits.tsv.gz',sep='\t',keep_default_na=False)
    def write(frame,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        records=[dict(cell_id=str(row.cell_id),label=int(row.y)) for row in frame.itertuples()]
        path.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    count=0
    for (dataset,fold),part in table.groupby(['dataset_id','fold'],sort=False):
        if dataset not in args.datasets:continue
        cells=reference_cells(dataset).loc[part.cell_id].copy()
        cells['split']=part.split.to_numpy()
        if len(cells)!=CONFIG['datasets'][dataset]['n_cells']:raise ValueError('Incomplete fold')
        folder=args.metadata_path/dataset
        if fold!='single':folder=folder/fold
        for role,filename in [('train','train'),('validation','val'),('test','test')]:
            write(cells[cells.split.eq(role)],folder/f'{filename}.json')
        count+=1
    for dataset in ['GSE163836','GSE202234','GSE223779_TS485T','GSE154600_EPI_external']:
        if dataset not in args.datasets:continue
        cells=reference_cells(dataset).copy()
        known=pd.to_numeric(cells.y).ge(0)
        folder=args.metadata_path/dataset
        write(cells[known],folder/'infer.json')
        if (~known).any():write(cells[~known],folder/'unknown.json')
        if dataset=='GSE163836':
            write(cells[known & cells.cell_line.eq('FCIBC02')],folder/'FCIBC02/infer.json')
            write(cells[known & cells.cell_line.eq('SUM149')],folder/'SUM149/infer.json')
        if dataset=='GSE223779_TS485T':
            for drug in ['crizotinib','alectinib']:
                write(cells[known & cells.drug.isin(['untreated',drug])],folder/drug/'infer.json')
    print('Exported',count,'source folds; JSON files contain cell_id and label')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['download','preprocess','splits','verify'],required=True)
    parser.add_argument('--datasets',nargs='+',choices=list(CONFIG['datasets']),default=list(CONFIG['datasets']))
    parser.add_argument('--open_path',type=Path,default=Path('data/raw'),help='Downloaded GEO files')
    parser.add_argument('--save_path',type=Path,default=Path('data/processed'),help='Output counts directory')
    parser.add_argument('--metadata_path',type=Path,default=Path('results'),help='Output labels, splits and validation tables')
    parser.add_argument('--r_path',default='Rscript',help='Rscript executable, required for GSE163836 or fresh author annotations')
    parser.add_argument('--reference_path',type=Path,help='Optional directory containing DATASET/counts.h5ad for exact comparison')
    parser.add_argument('--candidate',action='store_true',help='Compatibility option: requires --reference_path')
    parser.add_argument('--refresh_author_annotations',action='store_true',help='Download author RDS objects or regenerate GSE154600 annotations; otherwise use bundled annotations')
    parser.add_argument('--check_only',action='store_true',help='Validate reconstruction in memory without saving counts; optional --reference_path enables exact comparison')
    args=parser.parse_args()
    if args.candidate and args.reference_path is None:parser.error('--candidate requires --reference_path; omit both for standard validation')
    needs_r=args.stage=='preprocess' and ('GSE163836' in args.datasets or args.refresh_author_annotations)
    if needs_r and shutil.which(args.r_path) is None:
        parser.error('Rscript not found. Install R with Matrix and add Rscript to PATH, or set --r_path to its executable.')
    if args.stage=='splits':splits(args);return
    if args.stage=='download':
        fetch=load('download_mechanism_exp').fetch
        for gse in sorted({CONFIG['datasets'][d]['gse'] for d in args.datasets}):
            for resource in CONFIG['download_resources'][gse]:
                if '/author_annotations/' in resource['path'] and not args.refresh_author_annotations:continue
                path=args.open_path/resource['path'];path.parent.mkdir(parents=True,exist_ok=True)
                fetch(resource['url'],path)
        return
    if args.stage=='verify':
        records=[verify(dataset,args) for dataset in args.datasets]
        args.metadata_path.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(records).to_csv(args.metadata_path/'counts_validation.tsv',sep='\t',index=False)
        print(pd.DataFrame(records).to_string(index=False));return
    utility=load('prepare_mechanism_counts_exp');utility.METADATA_ROOT=args.metadata_path/'intermediate'
    utility.tabular_handle=tabular_handle;utility.tabular_genes=tabular_genes
    args.checked=[]
    if args.check_only:
        hgnc=pd.read_csv(BUNDLE/'resources/hgnc_gene_mapping.tsv',sep='\t',dtype=str,keep_default_na=False)
        table=hgnc[(hgnc.status=='Approved')&hgnc.ensembl_gene_id.ne('')]
        groups=table.groupby('ensembl_gene_id').symbol.agg(lambda s:sorted(set(s)))
        lookup={key:value[0] for key,value in groups.items() if len(value)==1}
        def canonicalize(data,folder,source):
            data.X=numeric_counts(data.X).astype(np.float32)
            data.var_names=pd.Index([lookup.get(n.split('.')[0],n) if n.startswith('ENSG') else n for n in data.var_names.astype(str)])
            if not data.var_names.is_unique:
                names=data.var_names;unique=pd.Index(names.unique())
                mapping=sparse.csr_matrix((np.ones(len(names)),(np.arange(len(names)),unique.get_indexer(names))),shape=(len(names),len(unique)))
                data=ad.AnnData((data.X@mapping).astype(np.float32),obs=data.obs.copy(),var=pd.DataFrame(index=unique))
            return data
        utility.write_counts=canonicalize
    for dataset in args.datasets:
        if dataset=='GSE140440':prostate(args)
        elif dataset in ['GSE223779','GSE223779_TS485T']:alk(dataset,args,utility)
        elif dataset=='GSE165897_pre_EOC':ovarian_source(args,utility)
        elif dataset.startswith('GSE223003_'):resistrace(dataset,args,utility)
        elif dataset.startswith('GSE149383_'):pc9(args,utility)
        elif dataset=='GSE154600_EPI_external':ovarian_external(args,utility)
        else:external(dataset,args)
    if args.check_only:
        args.metadata_path.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(args.checked).to_csv(args.metadata_path/'raw_reconstruction_audit.tsv',sep='\t',index=False)


if __name__=='__main__':main()

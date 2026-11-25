"""Profile full IG coverage on outer-training cells before reviewing a topK policy."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=PROJECT_ROOT
K_GRID=[128,256,512,768,1024,1536,2048,3072,4096]


from split_registry import load_cells

def profile(task,graph_root,save_path):
    folder=graph_root/task
    eligible=pd.read_csv(folder/'eligible_genes.tsv',sep='\t')
    cells=pd.read_csv(folder/'cells.tsv',sep='\t',keep_default_na=False)
    attribution=ROOT/'work/attributions'/task
    manifest=json.loads((attribution/'manifests/scimilarity_ig_manifest.json').read_text())
    if cells.cell_id.tolist()!=manifest['input']['cell_ids_used']: raise ValueError('Attribution cell order differs')
    full=sparse.load_npz(attribution/'attributions/scimilarity_ig.npz').tocsr()
    if not np.isfinite(full.data).all() or np.any(full.data<0): raise ValueError('Invalid full attribution')
    values=full[:,eligible.attribution_column.to_numpy(int)].tocsr()
    total=np.asarray(full.sum(axis=1)).ravel()
    ceiling=np.asarray(values.sum(axis=1)).ravel()/np.where(total>0,total,np.nan)
    mass=np.zeros((len(cells),len(K_GRID)))
    counts=np.zeros(len(cells),dtype=int)
    for row in range(len(cells)):
        scores=values.data[values.indptr[row]:values.indptr[row+1]]
        scores=np.sort(scores[scores>0])[::-1];counts[row]=len(scores)
        cumulative=np.cumsum(scores,dtype=np.float64)
        for column,k in enumerate(K_GRID):
            mass[row,column]=cumulative[min(k,len(scores))-1]/total[row] if len(scores) and total[row]>0 else np.nan
    save_path.mkdir(parents=True,exist_ok=True)
    diagnostic=pd.DataFrame({'cell_id':cells.cell_id,'positive_eligible_genes':counts,'eligible_mass_ceiling':ceiling,'total_importance':total})
    for column,k in enumerate(K_GRID): diagnostic[f'mass_k{k}']=mass[:,column]
    diagnostic.to_csv(save_path/'cell_coverage_diagnostics.tsv',sep='\t',index=False)
    all_records=[]
    folds=sorted((ROOT/'model/response_benchmark/data/grouped'/task).glob('fold_*'))
    if not folds: raise ValueError('Create independent-group folds before predictive K selection')
    for fold in folds:
        annotations=load_cells(fold/'data/cohorts'/task/'cells.tsv',task,mode='source')
        if annotations.cell_id.tolist()!=cells.cell_id.tolist(): raise ValueError('Fold cell order mismatch')
        selected=annotations.split.eq('train').to_numpy()
        if not selected.any(): raise ValueError('No training calibration cells')
        out=save_path/fold.name;out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'dataset':task,'cell_id':cells.loc[selected,'cell_id']}).to_csv(out/'calibration_cells.tsv',sep='\t',index=False)
        for column,k in enumerate(K_GRID):
            v=mass[selected,column];node_count=np.minimum(k,counts[selected])
            record={'dataset':task,'fold':fold.name,'k':k,'scope':'outer_training_cells_only',
                'n_calibration_cells':int(selected.sum()),'mass_p10':float(np.nanquantile(v,.1)),
                'mass_p25':float(np.nanquantile(v,.25)),'mass_median':float(np.nanmedian(v)),
                'mass_p75':float(np.nanquantile(v,.75)),'mass_p90':float(np.nanquantile(v,.9)),
                'eligible_ceiling_median':float(np.nanmedian(ceiling[selected])),
                'median_nodes':float(np.median(node_count)),'fraction_shorter_than_k':float((counts[selected]<k).mean())}
            for target in [.5,.6,.7,.8,.9]: record[f'fraction_mass_ge_{target}']=float((v>=target).mean())
            if 'patient_id' in annotations:
                per_patient=pd.DataFrame({'patient':annotations.loc[selected,'patient_id'].to_numpy(),'mass':v}).groupby('patient').mass.median()
                record['n_training_patients']=len(per_patient)
                record['minimum_patient_median_mass']=float(per_patient.min())
                record['patient_median_mass_median']=float(per_patient.median())
            all_records.append(record)
    pd.DataFrame(all_records).to_csv(save_path/'training_size_coverage_tradeoff.tsv',sep='\t',index=False,float_format='%.17g')
    print(pd.DataFrame(all_records).query('k in [512,768,1024,1536,2048]')[['fold','k','mass_p10','mass_median','eligible_ceiling_median']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    parser.add_argument('--graph_root',default=str(ROOT/'gene_sim_pathway/data/cell_signature_graphs_adaptive'))
    parser.add_argument('--save_path',required=True)
    args=parser.parse_args();profile(args.task,Path(args.graph_root),Path(args.save_path))

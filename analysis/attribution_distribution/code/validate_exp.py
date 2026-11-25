"""Audit full-cohort coverage, frozen IDs, IG arithmetic and figure statistics."""
import argparse,json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch_geometric.data import Batch
from base_common import *
from train_response_exp import restore,seed_all


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=["base","all"],default="all")
    p.add_argument("--datasets",nargs="+");p.add_argument("--device",default="cuda");a=p.parse_args();torch.set_num_threads(4);seed_all(42)
    records=[]
    for job in experiments(a.datasets):
        if a.datasets and job["dataset"] not in a.datasets:continue
        selected_domains=domains(job)
        for domain in ((next(selected_domains),) if a.stage=="base" else selected_domains):
            inputs=Inputs(input_folder(domain["dataset"],domain["cap"]));expected=domain_selection(job,domain,inputs)
            for model_name in MODELS:
                folder=attr_folder(job,domain,model_name)
                if not (folder/"complete.json").is_file():raise ValueError("IG incomplete: "+str(folder))
                cells=pd.read_csv(folder/"cells.tsv",sep="\t",keep_default_na=False)
                if cells.cell_id.tolist()!=expected.cell_id.tolist() or cells.y.tolist()!=expected.y.tolist():raise ValueError("JSON IDs/labels differ")
                margins=[];errors=[];rows=[];converged=[];steps=[]
                for file in sorted((folder/"blocks").glob("*.npz")):
                    with np.load(file,allow_pickle=False) as z:
                        count=len(z["raw_row"]);assert len(z["ptr"])==count+1
                        assert len(z["signed_ig"])==z["ptr"][-1] and np.isfinite(z["signed_ig"]).all()
                        assert np.array_equal(np.diff(z["ptr"]),inputs.ptr[z["raw_row"]+1]-inputs.ptr[z["raw_row"]])
                        sums=np.add.reduceat(z["signed_ig"].astype(np.float64),z["ptr"][:-1])
                        if not np.allclose(sums,z["margin"]-z["baseline_margin"]+z["delta"],atol=.001,rtol=.0001):raise ValueError("IG sum/delta mismatch")
                        recomputed=np.abs(z["delta"])/(1+np.abs(z["margin"]-z["baseline_margin"]))
                        if not np.allclose(recomputed,z["error"],atol=1e-7):raise ValueError("Convergence arithmetic differs")
                        rows.extend(z["raw_row"]);margins.extend(z["margin"]);errors.extend(z["error"])
                        converged.extend(z["converged"]);steps.extend(z["steps"])
                if rows!=expected.raw_row.tolist():raise ValueError("Incomplete or repeated cells")
                model,saved=restore(checkpoint(job,model_name),a.device)
                if model.identity.weight.shape[1]!=16 or model.config["gene_dim"]!=16:raise ValueError("Unexpected classifier family")
                if inputs.genes.tolist()!=saved["gene_symbols"]:raise ValueError("Gene lookup table differs")
                source=cohort_cells(job)
                for role,key in [("train","training_cell_ids"),("validation","validation_cell_ids")]:
                    if source.loc[source.split.eq(role),"cell_id"].tolist()!=saved[key]:raise ValueError("Fitting IDs changed")
                test=sorted({0,len(cells)//2,len(cells)-1});adj=sparse.load_npz(Path(inputs.info["package"])/"global_graphs"/(model_name+".npz")).tocsr()
                graphs=Batch.from_data_list([inputs.graph(int(cells.raw_row.iloc[i]),adj) for i in test]).to(a.device)
                with torch.no_grad():
                    logits=model(graphs.x,graphs.edge_index,graphs.edge_attr,graphs.batch,graphs.gene_id)
                    fresh=(logits[:,1]-logits[:,0]).cpu().numpy()
                difference=float(np.max(np.abs(fresh-np.array(margins)[test])))
                if difference>1e-4:raise ValueError("Batched IG endpoint does not match classifier")
                old_error=np.nan
                if domain["role"]=="base":
                    old=pd.read_csv(checkpoint(job,model_name).parent/"predictions.tsv",sep="\t",float_precision="round_trip").set_index("cell_id")
                    old_error=float(np.max(np.abs(old.loc[cells.cell_id,"margin_resistant_minus_sensitive"].to_numpy()-margins)))
                    if old_error>1e-4:raise ValueError("Historical full-base prediction mismatch")
                records.append(dict(experiment=job["experiment"],domain=domain["name"],model=model_name,n_cells=len(cells),
                    n_converged=int(np.sum(converged)),max_error=float(max(errors)),max_steps=int(max(steps)),
                    endpoint_reload_error=difference,historical_prediction_error=old_error,status="pass"))
                del model,graphs;torch.cuda.empty_cache()
                print("VERIFIED",job["experiment"],domain["name"],model_name,len(cells),flush=True)
    summary=pd.read_csv(OUT/"source_data/ridge_summary.tsv",sep="\t")
    values=pd.read_csv(OUT/"source_data/ridge_gene_values.tsv.gz",sep="\t")
    for row in summary.itertuples():
        if a.datasets and row.dataset not in a.datasets:continue
        if a.stage=="base" and row.domain!="base":continue
        part=values[(values.experiment==row.experiment)&(values.domain==row.domain)&(values.population==row.population)&
                    (values.stratum==row.stratum)&(values.set_kind==row.set_kind)&(values.method==row.method)]
        if len(part)!=row.n_selected or part.gene_id.duplicated().any():raise ValueError("Top gene count differs")
        if not np.isclose(row.gene_percent,100*row.n_selected/row.n_base_genes):raise ValueError("Sector percentage differs")
        if row.comparable:
            raw=part.value.to_numpy();assert np.isfinite(raw).all()
            q=np.quantile(raw,[.05,.25,.75,.95])
            if not np.allclose(q,[row.q05,row.q25,row.q75,row.q95]) or not np.isclose(raw.mean(),row.mean):raise ValueError("Ridge summary differs")
        elif part.value.notna().any():raise ValueError("Missing values filled incorrectly")
    pd.DataFrame(records).to_csv(OUT/"source_data/validation.tsv",sep="\t",index=False)
    print("VALIDATED",len(records),"model/domain jobs",sum(r["n_cells"] for r in records),"model-cells",flush=True)


if __name__=="__main__":main()

"""Summarize complete base IG, freeze gene sets, and export matched ridge statistics."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from base_common import *


def populations(cells,dataset):
    result={"cohort":np.ones(len(cells),bool)}
    if dataset=="GSE223779_TS485T":
        for drug in ["crizotinib","alectinib"]:result[drug]=cells.drug.isin(["untreated",drug]).to_numpy()
    if dataset=="GSE163836":
        for line in ["FCIBC02","SUM149"]:result[line]=cells.cell_line.eq(line).to_numpy()
    if dataset=="GSE140440":
        meta=pd.read_csv(SPLITS/"resources/sample_conditions.tsv",sep="\t",keep_default_na=False)
        meta=meta.loc[meta.dataset.eq(dataset)].set_index("sample_id")
        lines=cells.cell_id.map(meta["cell line"])
        if lines.isna().any():raise ValueError("Cell-line annotation missing")
        for line in ["DU145","PC3"]:result[line]=lines.eq(line).to_numpy()
    if dataset=="GSE202234":
        for treatment in sorted(cells.treatment.unique()):result[str(treatment)]=cells.treatment.eq(treatment).to_numpy()
    return result


def block_files(folder):
    return sorted((folder/"blocks").glob("*.npz"))


def ready(job,domain):
    return all((attr_folder(job,domain,m)/"complete.json").is_file() for m in MODELS)


def aggregate(job,domain):
    inputs=Inputs(prepare_input(domain));cells=domain_selection(job,domain,inputs)
    n=len(cells);positions={int(row):i for i,row in enumerate(cells.raw_row)}
    reference_mass=np.add.reduceat(np.asarray(inputs.signature,dtype=np.float64),inputs.ptr[:-1])
    valid=np.isfinite(reference_mass[cells.raw_row]) & (reference_mass[cells.raw_row]>0)
    checks=[];qc_tables={}
    for model in MODELS:
        folder=attr_folder(job,domain,model);saved=pd.read_csv(folder/"cells.tsv",sep="\t",keep_default_na=False)
        if saved.cell_id.tolist()!=cells.cell_id.tolist() or saved.y.tolist()!=cells.y.tolist():raise ValueError("Cell IDs or labels changed")
        seen_rows=[];qc=[]
        for file in block_files(folder):
            with np.load(file,allow_pickle=False) as z:
                rows=z["raw_row"];index=np.array([positions[int(r)] for r in rows]);size=np.diff(z["ptr"])
                expected=inputs.ptr[rows+1]-inputs.ptr[rows]
                if not np.array_equal(size,expected):raise ValueError("IG/node order changed")
                mass=np.add.reduceat(np.abs(z["signed_ig"].astype(np.float64)),z["ptr"][:-1])
                good=z["converged"] & np.isfinite(mass) & (mass>0)
                valid[index]&=good;seen_rows.extend(rows.tolist())
                part=pd.DataFrame({key:z[key] for key in ["margin","baseline_margin","delta","error","steps","converged"]})
                part["cell_id"]=cells.cell_id.iloc[index].to_numpy();part["label"]=cells.y.iloc[index].to_numpy()
                part["absolute_mass"]=mass;part["raw_row"]=rows;qc.append(part)
        if seen_rows!=cells.raw_row.tolist():raise ValueError("Incomplete or repeated attribution rows")
        qc_tables[model]=pd.concat(qc,ignore_index=True)
    scopes=[];masks=[]
    for pop,mask in populations(cells,domain["dataset"]).items():
        for stratum,label in [("All",None),("label_0",0),("label_1",1)]:
            selected=mask.copy()
            if label is not None:selected&=cells.y.eq(label).to_numpy()
            scopes.append((pop,stratum));masks.append(selected)
    masks=np.array(masks);counts=(masks&valid[None]).sum(1);total=masks.sum(1)
    means=np.zeros((len(METHODS),len(scopes),len(inputs.genes)),dtype=np.float64)
    for method_index,method in enumerate(METHODS):
        if method=="signature":
            chunks=[cells.raw_row.to_numpy()[i:i+512] for i in range(0,n,512)]
            blocks=[None]*len(chunks)
        else:
            blocks=block_files(attr_folder(job,domain,method));chunks=[None]*len(blocks)
        for chunk,file in zip(chunks,blocks):
            if file is None:
                rows=chunk;lengths=inputs.ptr[rows+1]-inputs.ptr[rows]
                indices=np.concatenate([np.arange(inputs.ptr[r],inputs.ptr[r+1]) for r in rows])
                ids=np.array(inputs.gene_id[indices],int);values=np.array(inputs.signature[indices],np.float64)
                ptr=np.r_[0,np.cumsum(lengths)]
            else:
                with np.load(file,allow_pickle=False) as z:
                    rows=z["raw_row"];ptr=z["ptr"].copy();lengths=np.diff(ptr)
                    indices=np.concatenate([np.arange(inputs.ptr[r],inputs.ptr[r+1]) for r in rows])
                    ids=np.array(inputs.gene_id[indices],int);values=np.abs(z["signed_ig"].astype(np.float64))
            local=np.array([positions[int(r)] for r in rows]);mass=np.add.reduceat(values,ptr[:-1])
            normalized=np.divide(values,np.repeat(mass,lengths),out=np.zeros_like(values),where=np.repeat(mass>0,lengths))*1000
            for s in range(len(scopes)):
                keep=np.repeat(masks[s,local]&valid[local],lengths)
                means[method_index,s]+=np.bincount(ids[keep],weights=normalized[keep],minlength=len(inputs.genes))
        means[method_index]=np.divide(means[method_index],counts[:,None],out=np.full_like(means[method_index],np.nan),where=counts[:,None]>0)
    dest=OUT/"source_data/gene_means"/job["experiment"];dest.mkdir(parents=True,exist_ok=True)
    file=dest/(domain["name"]+".npz")
    np.savez_compressed(file,means=means,population=np.array([p for p,s in scopes]),stratum=np.array([s for p,s in scopes]),n_cells=counts,n_total=total)
    cells.assign(common_usable=valid,signature_mass_in_graph=reference_mass[cells.raw_row]).to_csv(dest/(domain["name"]+"_cells.tsv.gz"),sep="\t",index=False)
    for model,table in qc_tables.items():
        checks.append(dict(experiment=job["experiment"],domain=domain["name"],model=model,n_total=n,n_common_usable=int(valid.sum()),
                           n_converged=int(table.converged.sum()),max_relative_error=float(table.error.max()),
                           mean_signature_coverage=float(reference_mass[cells.raw_row].mean())))
    import torch
    for model in MODELS:
        saved=torch.load(checkpoint(job,model),map_location="cpu",weights_only=False)
        observed=np.zeros(len(inputs.genes),bool);observed[saved["training_observed_gene_ids"]]=True
        node_count=unseen=0;unique=[]
        for row in cells.raw_row.to_numpy():
            ids=inputs.gene_id[inputs.ptr[row]:inputs.ptr[row+1]]
            node_count+=len(ids);unseen+=int((~observed[ids]).sum());unique.append(ids)
        unique=np.unique(np.concatenate(unique))
        for record in checks:
            if record["model"]==model:
                record["n_unique_genes_unseen_training"]=int((~observed[unique]).sum())
                record["fraction_nodes_unseen_training"]=unseen/node_count
    return dict(means=means,scopes=scopes,n_cells=counts,n_total=total,inputs=inputs,cells=cells,checks=checks)


def gene_sets(job,base):
    inputs=base["inputs"];eligible=inputs.eligible;all_index=base["scopes"].index(("cohort","All"))
    result={}
    for mi,method in enumerate(METHODS):
        scores=base["means"][mi,all_index,eligible]
        if not np.isfinite(scores).all():raise ValueError("No usable full-base scores")
        order=np.lexsort((inputs.genes[eligible],-scores))
        selected=eligible[order[:min(100,len(eligible))]]
        result[method]=selected
    write_json(OUT/"source_data/top100"/(job["experiment"]+".json"),dict(experiment=job["experiment"],
        n_base_genes=len(eligible),universe="common graph-supported measured unique HGNC candidates before per-cell K",
        methods={m:dict(gene_id=g.tolist(),gene_symbol=inputs.genes[g].tolist()) for m,g in result.items()}))
    return result


def export_ridges(job,domain,stats,selected,base_universe,records,summaries):
    inputs=stats["inputs"];measured=set(inputs.eligible.tolist())
    for set_kind in ["own_top100","signature_anchor"]:
        for mi,method in enumerate(METHODS):
            ids=selected[method] if set_kind=="own_top100" else selected["signature"]
            present=np.array([int(g) in measured for g in ids]);available=int(present.sum())
            for si,(population,stratum) in enumerate(stats["scopes"]):
                values=stats["means"][mi,si,ids]
                comparable=available==len(ids) and stats["n_cells"][si]>0 and np.isfinite(values).all()
                row=dict(experiment=job["experiment"],dataset=job["dataset"],fold=job["fold"],domain=domain["name"],
                         target_dataset=domain["dataset"],population=population,stratum=stratum,set_kind=set_kind,method=method,
                         n_total_cells=int(stats["n_total"][si]),n_cells=int(stats["n_cells"][si]),n_base_genes=base_universe,
                         n_selected=len(ids),n_measured=available,gene_percent=100*len(ids)/base_universe,comparable=comparable)
                if comparable:
                    q=np.quantile(values,[.05,.25,.75,.95]);row.update(mean=float(values.mean()),q05=float(q[0]),q25=float(q[1]),q75=float(q[2]),q95=float(q[3]),zero_genes=int((values==0).sum()))
                else:row.update(mean=np.nan,q05=np.nan,q25=np.nan,q75=np.nan,q95=np.nan,zero_genes=0)
                summaries.append(row)
                for gid,value,has in zip(ids,values,present):
                    records.append(dict(**{k:row[k] for k in ["experiment","domain","population","stratum","set_kind","method"]},
                                        gene_id=int(gid),gene_symbol=inputs.genes[gid],value=float(value) if has and comparable else np.nan,measured=bool(has)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--datasets",nargs="+")
    p.add_argument("--stage",choices=["base","all"],default="all");p.add_argument("--allow_partial",action="store_true")
    args=p.parse_args();records=[];summaries=[];checks=[];catalog=[]
    for job in experiments(args.datasets):
        if args.datasets and job["dataset"] not in args.datasets:continue
        base_domain=next(domains(job))
        if not ready(job,base_domain):
            if args.allow_partial:continue
            raise RuntimeError("Full-base IG is incomplete: "+job["experiment"])
        base=aggregate(job,base_domain);sets=gene_sets(job,base);n_base=len(base["inputs"].eligible)
        export_ridges(job,base_domain,base,sets,n_base,records,summaries);checks.extend(base["checks"])
        catalog.append(dict(experiment=job["experiment"],domain="base",n_cells=len(base["cells"])))
        if args.stage=="all":
            for domain in list(domains(job))[1:]:
                if not ready(job,domain):
                    if args.allow_partial:continue
                    raise RuntimeError("Target IG incomplete: "+job["experiment"]+" / "+domain["name"])
                stats=aggregate(job,domain);export_ridges(job,domain,stats,sets,n_base,records,summaries)
                checks.extend(stats["checks"]);catalog.append(dict(experiment=job["experiment"],domain=domain["name"],n_cells=len(stats["cells"])))
                print("SUMMARIZED",job["experiment"],domain["name"],flush=True)
        print("SUMMARIZED",job["experiment"],"base",flush=True)
    if not records:raise ValueError("No complete experiments available")
    dest=OUT/"source_data";dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(dest/"ridge_gene_values.tsv.gz",sep="\t",index=False)
    pd.DataFrame(summaries).to_csv(dest/"ridge_summary.tsv",sep="\t",index=False)
    pd.DataFrame(checks).to_csv(dest/"attribution_qc.tsv",sep="\t",index=False)
    pd.DataFrame(catalog).to_csv(dest/"completed_domains.tsv",sep="\t",index=False)
    print("EXPORTED",len(catalog),"experiment/domains",len(summaries),"ridge summaries",flush=True)


if __name__=="__main__":main()

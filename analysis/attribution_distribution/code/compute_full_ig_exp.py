"""Compute zero-baseline Captum IG for complete JSON-defined base cohorts."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from captum.attr import IntegratedGradients
from base_common import *
from train_response_exp import restore,seed_all


def batch_ig(model,graphs,steps,internal_batch,device):
    count=len(graphs);sizes=[g.num_nodes for g in graphs];width=max(sizes)
    expression=torch.zeros((count,width),dtype=torch.float32,device=device)
    mask=torch.zeros((count,width),dtype=torch.bool,device=device)
    edges=[];attrs=[];identities=[];batches=[];offset=0
    for i,g in enumerate(graphs):
        n=g.num_nodes;expression[i,:n]=g.x[:,0].to(device);mask[i,:n]=True
        edges.append(g.edge_index.to(device)+offset);attrs.append(g.edge_attr.to(device));identities.append(g.gene_id.to(device))
        batches.append(torch.full((n,),i,dtype=torch.long,device=device));offset+=n
    edge_index=torch.cat(edges,1);edge_attr=torch.cat(attrs);identity=torch.cat(identities);batch=torch.cat(batches);nodes=offset
    def forward(values):
        copies=len(values)//count
        if len(values)%count:raise ValueError("Captum interpolation batch does not preserve graph groups")
        copies_index=torch.arange(copies,device=device)
        x=values.reshape(copies,count,width)[:,mask].reshape(copies*nodes,1)
        edge=(edge_index[None]+copies_index[:,None,None]*nodes).permute(1,0,2).reshape(2,-1)
        group=(batch[None]+copies_index[:,None]*count).reshape(-1)
        z=model(x,edge,edge_attr.repeat(copies,1),group,identity.repeat(copies))
        return z[:,1]-z[:,0]
    internal=max(count,(internal_batch//count)*count)
    with torch.no_grad():original=forward(expression);reference=forward(torch.zeros_like(expression))
    values,delta=IntegratedGradients(forward).attribute(expression,baselines=torch.zeros_like(expression),
              method="riemann_middle",n_steps=steps,internal_batch_size=internal,return_convergence_delta=True)
    raw=values.detach().cpu().numpy();original=original.cpu().numpy();reference=reference.cpu().numpy();delta=delta.cpu().numpy()
    attrs=[raw[i,:n].copy() for i,n in enumerate(sizes)]
    error=np.abs(delta)/(1+np.abs(original-reference))
    return attrs,original,reference,delta,error


def safe_batch_ig(model,graphs,steps,args):
    internal=args.internal_batch
    while True:
        failed=False
        try:
            return (*batch_ig(model,graphs,steps,internal,args.device),internal)
        except torch.cuda.OutOfMemoryError:
            failed=True
        if failed:
            gc.collect();torch.cuda.empty_cache()
            if internal<=len(graphs):raise RuntimeError("One interpolation graph batch exceeds GPU memory; reduce cell_batch")
            internal=max(len(graphs),(internal//2//len(graphs))*len(graphs))


def compute_batch(model,graphs,args):
    values,original,reference,delta,error,internal=safe_batch_ig(model,graphs,args.steps,args)
    steps=np.full(len(graphs),args.steps,dtype=int);effective=np.full(len(graphs),internal,dtype=int)
    more=args.steps*2
    while more<=args.max_steps:
        bad=np.flatnonzero(error>args.tolerance)
        if not len(bad):break
        a,o,b,d,e,internal=safe_batch_ig(model,[graphs[i] for i in bad],more,args)
        for j,i in enumerate(bad):values[i]=a[j]
        original[bad]=o;reference[bad]=b;delta[bad]=d;error[bad]=e;steps[bad]=more;effective[bad]=internal
        more*=2
    if not all(np.isfinite(value).all() for value in values):raise ValueError("Nonfinite IG")
    sums=np.array([v.sum(dtype=np.float64) for v in values])
    if not np.allclose(sums,original-reference+delta,atol=.001,rtol=.0001):raise ValueError("Completeness sum mismatch")
    return values,dict(margin=original,baseline_margin=reference,delta=delta,error=error,steps=steps,
                       internal_batch=effective,converged=error<=args.tolerance)


def run(job,domain,model_name,args):
    input_path=prepare_input(domain);inputs=Inputs(input_path);cells=domain_selection(job,domain,inputs)
    folder=attr_folder(job,domain,model_name);folder.mkdir(parents=True,exist_ok=True)
    params=dict(experiment=job["experiment"],domain=domain["name"],input_path=str(input_path),checkpoint=str(checkpoint(job,model_name)),
                baseline="zero_expression_fixed_nodes_identity_and_graph",method="riemann_middle",steps=args.steps,max_steps=args.max_steps,
                tolerance=args.tolerance,cell_batch=args.cell_batch,internal_batch=args.internal_batch,block_cells=args.block_cells,
                source_K=job["cap"],n_cells=len(cells),relation=model_name,seed=42,device=args.device,threads=args.threads)
    param=folder/"parameters.json"
    if param.exists() and json.loads(param.read_text(encoding="utf-8"))!=params:raise ValueError("Saved job parameters differ")
    write_json(param,params);cells.to_csv(folder/"cells.tsv",sep="\t",index=False)
    blocks=folder/"blocks";blocks.mkdir(exist_ok=True)
    model,saved=restore(checkpoint(job,model_name),args.device)
    source_cells=cohort_cells(job)
    for role,key in [("train","training_cell_ids"),("validation","validation_cell_ids")]:
        if saved[key]!=source_cells.loc[source_cells.split.eq(role),"cell_id"].tolist():raise ValueError("Checkpoint split differs from JSON")
    if inputs.genes.tolist()!=saved["gene_symbols"]:raise ValueError("Model gene vocabulary differs")
    for parameter in model.parameters():parameter.requires_grad_(False)
    adjacency=sparse.load_npz(Path(inputs.info["package"])/"global_graphs"/(model_name+".npz")).tocsr()
    start=time.perf_counter()
    for lo in range(0,len(cells),args.block_cells):
        hi=min(lo+args.block_cells,len(cells));path=blocks/f"{lo:07d}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z["raw_row"],cells.raw_row.iloc[lo:hi]):raise ValueError("Cached cell IDs differ")
            continue
        arrays=[];sizes=[];diagnostics=[]
        for first in range(lo,hi,args.cell_batch):
            rows=cells.raw_row.iloc[first:min(first+args.cell_batch,hi)].to_numpy(int)
            graphs=[inputs.graph(int(row),adjacency) for row in rows]
            values,qc=compute_batch(model,graphs,args);arrays.extend(values);sizes.extend(map(len,values));diagnostics.append(qc)
        data=dict(raw_row=cells.raw_row.iloc[lo:hi].to_numpy(np.int64),ptr=np.r_[0,np.cumsum(sizes)].astype(np.int64),signed_ig=np.concatenate(arrays).astype(np.float32))
        for key in diagnostics[0]:data[key]=np.concatenate([q[key] for q in diagnostics])
        pending=path.with_suffix(".pending.npz");np.savez_compressed(pending,**data);pending.replace(path)
        done=hi;elapsed=time.perf_counter()-start
        if lo==0 or hi==len(cells) or hi%1024==0:print("BLOCK",job["experiment"],domain["name"],model_name,done,"/",len(cells),"elapsed_s",round(elapsed,1),"failures",int((~data["converged"]).sum()),flush=True)
    write_json(folder/"complete.json",dict(n_cells=len(cells),n_blocks=len(list(blocks.glob("*.npz"))),status="complete"))
    print("COMPLETE",job["experiment"],domain["name"],model_name,len(cells),flush=True)
    del model,inputs,adjacency;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["prepare","base","target","all"],default="base")
    p.add_argument("--datasets",nargs="+");p.add_argument("--fold");p.add_argument("--models",nargs="+",choices=MODELS,default=MODELS)
    p.add_argument("--cell_batch",type=int,default=8);p.add_argument("--internal_batch",type=int,default=128)
    p.add_argument("--block_cells",type=int,default=256);p.add_argument("--steps",type=int,default=64)
    p.add_argument("--max_steps",type=int,default=1024);p.add_argument("--tolerance",type=float,default=.005)
    p.add_argument("--device",default="cuda");p.add_argument("--threads",type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads);seed_all(42)
    jobs=sorted(experiments(a.datasets, a.fold),key=lambda j:(j["n_cells"],j["dataset"],j["fold"]))
    write_json(OUT/"source_data/experiments.json",jobs)
    for job in jobs:
        if a.datasets and job["dataset"] not in a.datasets:continue
        if a.fold and job["fold"]!=a.fold:continue
        selected_domains=domains(job)
        for domain in ((next(selected_domains),) if a.stage=="base" else selected_domains):
            if a.stage=="target" and domain["role"]!="target":continue
            if a.stage=="prepare":prepare_input(domain);continue
            for model in a.models:run(job,domain,model,a)


if __name__=="__main__":main()

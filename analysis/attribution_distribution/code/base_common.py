"""Shared experiment definitions and frozen graph inputs for attribution ridgelines."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
BENCH = Path(os.environ.get("CELL_XAI_BENCHMARK_ROOT", ROOT / "model/response_benchmark")).resolve()
ATTRIBUTIONS = Path(os.environ.get("CELL_XAI_ATTRIBUTION_ROOT", ROOT / "work/attributions")).resolve()
GROUPED_GRAPHS = Path(os.environ.get("CELL_XAI_GROUPED_GRAPHS_ROOT",
                                      ROOT / "gene_sim_pathway/data/cell_signature_graphs_grouped")).resolve()
sys.path.insert(0, str(ROOT / "model/response_benchmark/code"))
import benchmark_paths
from split_registry import load_cells, apply_splits

MODELS = ["state_15b", "esm2_650m", "qwen"]
METHODS = ["signature", *MODELS]
NAMES = {"signature": "SIGnature", "state_15b": "STATE-15B", "esm2_650m": "ESM2-650M", "qwen": "Qwen"}
COLORS = {"signature": "#67A8AF", "state_15b": "#CA5596", "esm2_650m": "#DB9735", "qwen": "#007580"}
SPLITS = ROOT / "dataset_util/datasets_download_preprocess/26_11_20"


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def experiments(datasets=None, selected_fold=None):
    config = json.loads((SPLITS / "datasets.json").read_text(encoding="utf-8"))
    jobs = []
    for design in config["source_designs"]:
        dataset, fold = design["dataset"], design["fold"]
        if (datasets and dataset not in datasets) or (selected_fold and fold != selected_fold):
            continue
        cohort = BENCH if fold == "single" else BENCH / "data/grouped" / dataset / fold
        info = json.loads((cohort / "data/cohorts" / dataset / "data_sources.json").read_text(encoding="utf-8"))
        cap = int(info.get("selected_k", info.get("top_k")))
        jobs.append(dict(dataset=dataset, fold=fold, experiment=dataset + "__" + fold,
                         cohort=str(cohort), package=str(Path(info["graph_source"]).parent),
                         cap=cap, n_cells=config["datasets"][dataset]["n_cells"]))
    return jobs


def checkpoint(job, model):
    if job["fold"] == "single":
        parent = BENCH / "output" if model == "state_15b" else BENCH / "output/transfer" / model
    else:
        parent = BENCH / "output/grouped_adaptive" / job["dataset"] / job["fold"] / model
    return parent / "within" / job["dataset"] / "gat_gine/checkpoint.pt"


def input_folder(dataset, cap):
    return OUT / "data/inputs" / (dataset + "__K" + str(cap))


def cohort_cells(job, dataset=None):
    dataset = dataset or job["dataset"]
    if dataset == job["dataset"]:
        return load_cells(Path(job["cohort"]) / "data/cohorts" / dataset / "cells.tsv", dataset, mode="source")
    if dataset == "GSE154600_EPI_external":
        graph = GROUPED_GRAPHS / f"K{job['cap']}" / dataset
        cells = pd.read_csv(graph / "cells.tsv", sep="\t", keep_default_na=False)
        labels = pd.read_csv(Path(os.environ.get("CELL_XAI_COHORT_METADATA", ROOT / "work/cohort_metadata")) / dataset / "counts_labels.tsv",
                             sep="\t", keep_default_na=False).set_index("Cell_ID")
        cells["y"] = cells.cell_id.map(labels.label).map({"sensitive": 0, "resistant": 1}).fillna(-1).astype(int)
        for key in ["patient_id", "sample_id", "label"]:
            if key in labels:cells[key] = cells.cell_id.map(labels[key])
        return apply_splits(cells, dataset, mode="infer")
    return load_cells(BENCH / "data/cohorts" / dataset / "cells.tsv", dataset, mode="infer")


def domains(job):
    yield dict(name="base", dataset=job["dataset"], package=job["package"], cap=job["cap"], role="base")
    if job["dataset"] == "GSE223779":
        for dataset in ["GSE223779_TS485T", "GSE140440", "GSE163836", "GSE202234"]:
            info = json.loads((BENCH / "data/cohorts" / dataset / "data_sources.json").read_text(encoding="utf-8"))
            yield dict(name="target__" + dataset, dataset=dataset, package=str(Path(info["graph_source"]).parent), cap=job["cap"], role="target")
    if job["dataset"] == "GSE165897_pre_EOC":
        yield dict(name="target__GSE154600_EPI_external", dataset="GSE154600_EPI_external",
                   package=str(GROUPED_GRAPHS / f"K{job['cap']}"), cap=job["cap"], role="target")


def prepare_input(domain):
    dataset, cap = domain["dataset"], domain["cap"]
    dest = input_folder(dataset, cap)
    if (dest / "input.json").exists():
        saved = json.loads((dest / "input.json").read_text(encoding="utf-8"))
        if saved["package"] != domain["package"]:raise ValueError("Input package changed")
        return dest
    package = Path(domain["package"]);dest.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(package / dataset / "cells.tsv", sep="\t", keep_default_na=False)
    if not cells.cell_id.is_unique:raise ValueError("Duplicate graph cells")
    genes = pd.read_csv(package / "gene_vocab.tsv", sep="\t")
    eligible = pd.read_csv(package / dataset / "eligible_genes.tsv", sep="\t")
    if eligible.gene_id.duplicated().any() or eligible.gene_symbol.duplicated().any():raise ValueError("Ambiguous candidate genes")
    if genes.gene_symbol.iloc[eligible.gene_id].tolist() != eligible.gene_symbol.tolist():raise ValueError("Gene order differs")
    with np.load(package / dataset / "signatures.npz", allow_pickle=False) as source:
        oldptr, ids, features = source["ptr"], source["gene_id"], source["features"]
        ptr = np.r_[0, np.cumsum(np.minimum(np.diff(oldptr), cap))].astype(np.int64)
        if (np.diff(ptr) < 1).any():raise ValueError("Cell with no graph nodes")
        gene_id = np.empty(ptr[-1], dtype=np.int32)
        expression = np.empty(ptr[-1], dtype=np.float32)
        signature = np.empty(ptr[-1], dtype=np.float32)
        for i in range(len(cells)):
            count = ptr[i+1]-ptr[i];lo, hi = ptr[i:i+2];start=oldptr[i]
            gene_id[lo:hi]=ids[start:start+count];expression[lo:hi]=features[start:start+count,0]
            signature[lo:hi]=features[start:start+count,1]
        if not np.isfinite(expression).all() or not np.isfinite(signature).all() or (signature<0).any():raise ValueError("Invalid input values")
    for key, value in [("ptr",ptr),("gene_id",gene_id),("expression",expression),("signature",signature)]:
        np.save(dest / (key+".npy"),value,allow_pickle=False)
    cells.to_csv(dest / "cells.tsv",sep="\t",index=False)
    genes.to_csv(dest / "gene_vocab.tsv",sep="\t",index=False)
    eligible.to_csv(dest / "eligible_genes.tsv",sep="\t",index=False)
    manifest = json.loads((ATTRIBUTIONS / dataset / "manifests/scimilarity_ig_manifest.json").read_text(encoding="utf-8"))
    if not set(cells.cell_id)<=set(manifest["input"]["cell_ids_used"]):raise ValueError("SIGnature cells missing")
    write_json(dest/"input.json",dict(dataset=dataset,cap=cap,package=domain["package"],n_cells=len(cells),
                                    n_nodes=int(ptr[-1]),signature_feature="original IG divided by 1000",node_policy="source_K_frozen"))
    return dest


class Inputs:
    def __init__(self, folder):
        self.folder=Path(folder)
        self.info=json.loads((self.folder/"input.json").read_text(encoding="utf-8"))
        self.cells=pd.read_csv(self.folder/"cells.tsv",sep="\t",keep_default_na=False)
        self.genes=pd.read_csv(self.folder/"gene_vocab.tsv",sep="\t").gene_symbol.astype(str).to_numpy()
        self.eligible=pd.read_csv(self.folder/"eligible_genes.tsv",sep="\t").gene_id.to_numpy(int)
        for name in ["ptr","gene_id","expression","signature"]:setattr(self,name,np.load(self.folder/(name+".npy"),mmap_mode="r"))

    def graph(self,row,adjacency):
        import torch
        from torch_geometric.data import Data
        lo,hi=self.ptr[row:row+2];ids=np.array(self.gene_id[lo:hi],dtype=np.int64)
        sub=adjacency[ids][:,ids].tocoo()
        return Data(x=torch.from_numpy(np.array(self.expression[lo:hi],dtype=np.float32)[:,None]),
                    gene_id=torch.from_numpy(ids),edge_index=torch.from_numpy(np.stack([sub.row,sub.col]).astype(np.int64)),
                    edge_attr=torch.from_numpy(sub.data.astype(np.float32)[:,None]),num_nodes=len(ids))


def domain_selection(job,domain,inputs):
    cells=cohort_cells(job,domain["dataset"])
    if domain["role"]=="target":cells=cells[cells.y.ge(0)].copy()
    index=pd.Index(inputs.cells.cell_id).get_indexer(cells.cell_id)
    if (index<0).any():raise ValueError("Unmatched graph cell ID")
    cells=cells.reset_index(drop=True);cells["raw_row"]=index
    return cells


def attr_folder(job,domain,model):
    return OUT/"data/attributions"/job["experiment"]/domain["name"]/model

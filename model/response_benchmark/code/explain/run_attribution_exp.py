"""Explain three complete graph models on predetermined held-out biological units."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from benchmark_paths import BENCH_ROOT, PROJECT_ROOT
from cell_graph_dataset import CellGraphDataset
from split_registry import load_cells, apply_splits, split_root
from train_response_exp import restore, seed_all
from attribution_methods import adaptive_ig, edge_explanation

RELATIONS = ["state_15b", "esm2_650m", "qwen"]
TASKS = ["GSE149383_PC9_DTEP_sample_grouped", "GSE223003_carboplatin_pre", "GSE223003_olaparib_pre", "GSE223779_TS485T"]


def jobs(root=BENCH_ROOT):
    for dataset in TASKS:
        folds = ["infer"] if dataset.endswith("TS485T") else ["fold_1", "fold_2"]
        for fold in folds:
            external = fold == "infer"
            source = "GSE223779" if external else dataset
            cohort = root if external else root / "data/grouped" / dataset / fold
            source_cohort = root if external else cohort
            yield dict(dataset=dataset, fold=fold, source=source, cohort=cohort,
                       root=root,
                       source_cohort=source_cohort, group="drug" if external else
                       ("sample_id" if "PC9" in dataset else "biological_replicate"))


def cohort_inputs(root, dataset):
    folder = root / "data/cohorts" / dataset
    cells = load_cells(folder / "cells.tsv", dataset)
    info = json.loads((folder / "data_sources.json").read_text(encoding="utf-8"))
    return cells, Path(info["graph_source"]).parent


def checkpoint_path(job, relation):
    benchmark = job["root"]
    if job["fold"] == "infer":
        root = benchmark / "output" if relation == "state_15b" else benchmark / "output/transfer" / relation
    else:
        root = benchmark / "output/grouped_adaptive" / job["source"] / job["fold"] / relation
    return root / "within" / job["source"] / "gat_gine/checkpoint.pt"


def select_cells(cells, job, cap, seed):
    candidates = cells[cells.y.ge(0)] if job["fold"] == "infer" else cells[cells.split.eq("test")]
    rng = np.random.default_rng(seed)
    result = []
    for _, group in candidates.groupby([job["group"], "y"], sort=True):
        group = group.sort_values("cell_id")
        if cap and len(group) > cap:
            group = group.iloc[np.sort(rng.choice(len(group), cap, replace=False))]
        result.append(group)
    return pd.concat(result).sort_values("cell_id").reset_index(drop=True)


def sensitive_reference(cells, raw, graph_ids):
    index = pd.Index(raw.cells.cell_id).get_indexer(cells.cell_id)
    if (index < 0).any():
        raise ValueError("Source graph cell missing")
    sensitive = cells.split.eq("train") & cells.y.eq(0)
    totals = np.zeros(len(raw.genes), dtype=np.float64)
    counts = np.zeros(len(raw.genes), dtype=np.int64)
    for i in index[sensitive.to_numpy()]:
        lo, hi = raw.ptr[i:i+2]
        ids = raw.gene_id[lo:hi]
        np.add.at(totals, ids, raw.features[lo:hi, 0])
        np.add.at(counts, ids, 1)
    if not sensitive.any():
        raise ValueError("No training-sensitive reference cells")
    reference = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    return reference.astype(np.float32), counts, cells.loc[sensitive, "cell_id"].tolist()


def run_job(job, relation, args):
    folder = args.save_path / job["dataset"] / job["fold"] / relation
    folder.mkdir(parents=True, exist_ok=True)
    cells, package = cohort_inputs(job["cohort"], job["dataset"])
    selected = select_cells(cells, job, args.cells_per_group, args.seed)
    selected.to_csv(folder / "cells.tsv", sep="\t", index=False)
    checkpoint = checkpoint_path(job, relation)
    parameters = dict(dataset=job["dataset"], fold=job["fold"], relation=relation,
                      checkpoint=str(checkpoint), splits_path=str(split_root()),
                      target="resistant_minus_sensitive_logit", node_policy="existing_fixed_K",
                      cell_selection="seeded sampling within held-out biological group and true label",
                      cells_per_group=args.cells_per_group, seed=args.seed, initial_steps=args.steps,
                      maximum_steps=args.max_steps, tolerance=args.tolerance, internal_batch=args.internal_batch,
                      baselines=["zero", "training_sensitive_selected_node_mean"], gnn_epochs=args.gnn_epochs)
    param_path = folder / "parameters.json"
    if param_path.exists() and json.loads(param_path.read_text()) != parameters:
        raise ValueError("Existing attribution parameters differ; choose a new save_path")
    param_path.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    raw = CellGraphDataset(package, job["dataset"], relation)
    indices = pd.Index(raw.cells.cell_id).get_indexer(selected.cell_id)
    if (indices < 0).any():
        raise ValueError("Selected graph ID absent")
    model, saved = restore(checkpoint, args.device)
    if raw.genes.tolist() != saved["gene_symbols"]:
        raise ValueError("Checkpoint gene vocabulary mismatch")
    source_cells, source_package = cohort_inputs(job["source_cohort"], job["source"])
    for role, key in [("train", "training_cell_ids"), ("validation", "validation_cell_ids")]:
        if saved[key] != source_cells.loc[source_cells.split.eq(role), "cell_id"].tolist():
            raise ValueError("Checkpoint fitting IDs differ from authoritative JSON")
    if set(selected.cell_id) & set(saved["training_cell_ids"] + saved["validation_cell_ids"]):
        raise ValueError("Explanation cells overlap fitting cells")
    source_raw = raw if job["source"] == job["dataset"] else CellGraphDataset(source_package, job["source"], relation)
    reference, reference_counts, sensitive_ids = sensitive_reference(source_cells, source_raw, raw.genes)
    pd.DataFrame(dict(gene_symbol=raw.genes, sensitive_mean=reference, n_sensitive_selected_occurrences=reference_counts)).to_csv(folder / "sensitive_reference.tsv", sep="\t", index=False)
    (folder / "reference_cells.json").write_text(json.dumps(sensitive_ids, indent=2) + "\n", encoding="utf-8")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    cache = folder / "cells"
    cache.mkdir(exist_ok=True)
    started = time.perf_counter()
    for count, (row, raw_index) in enumerate(zip(selected.itertuples(), indices)):
        target = cache / f"{count:05d}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as z:
                if str(z["cell_id"]) != row.cell_id:
                    raise ValueError("Cached cell identity changed")
            continue
        graph = raw.to_pyg(int(raw_index)).to(args.device)
        graph.x = graph.x[:, :1].clone()
        graph.y = torch.tensor([int(row.y)], device=args.device)
        graph_ids = graph.gene_id.cpu().numpy()
        if len(np.unique(graph_ids)) != len(graph_ids):
            raise ValueError("Duplicate nodes")
        arrays = dict(cell_id=np.array(row.cell_id), gene_id=graph_ids, expression=graph.x[:, 0].cpu().numpy(),
                      reference_missing=reference_counts[graph_ids] == 0)
        diagnostic = []
        for name, baseline in [("zero", torch.zeros_like(graph.x)),
                               ("sensitive", torch.as_tensor(reference[graph_ids, None], device=args.device))]:
            values, info = adaptive_ig(model, graph, baseline, args.steps, args.max_steps,
                                       args.internal_batch, args.tolerance)
            arrays[name + "_signed_ig"] = values
            diagnostic.append(dict(cell_id=row.cell_id, baseline=name, **info))
        arrays["diagnostics"] = np.array(json.dumps(diagnostic))
        np.savez_compressed(target, **arrays)
        if (count + 1) % 8 == 0 or count == 0:
            print("IG", job["dataset"], job["fold"], relation, count + 1, "/", len(selected),
                  "seconds", round(time.perf_counter() - started, 1), flush=True)
    if args.gnn_epochs:
        cases = selected.groupby("y", sort=True).head(1)
        for row in cases.itertuples():
            target = folder / f"gnn_class_{row.y}.npz"
            if target.exists():continue
            i = int(pd.Index(raw.cells.cell_id).get_loc(row.cell_id))
            graph = raw.to_pyg(i).to(args.device);graph.x = graph.x[:, :1].clone()
            seed_all(args.seed + int(row.y))
            masks, nodes, predicted = edge_explanation(model, graph, args.gnn_epochs)
            np.savez_compressed(target, cell_id=np.array(row.cell_id), true_label=int(row.y),
                                target_class=predicted, gene_id=graph.gene_id.cpu().numpy(),
                                edge_index=graph.edge_index.cpu().numpy(), prior_weight=graph.edge_attr[:, 0].cpu().numpy(),
                                edge_mask=masks, node_mask=nodes)
    print("DONE", job["dataset"], job["fold"], relation, len(selected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open_path", type=Path, default=BENCH_ROOT)
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--models", nargs="+", choices=RELATIONS, default=RELATIONS)
    parser.add_argument("--cells_per_group", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--internal_batch", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=.005)
    parser.add_argument("--gnn_epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = args.open_path / "output/attribution"
    torch.set_num_threads(4);seed_all(args.seed)
    for job in jobs(args.open_path):
        if job["dataset"] not in args.datasets:continue
        for relation in args.models:
            run_job(job, relation, args)
            if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

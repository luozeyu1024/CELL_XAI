"""Aggregate matched-cell IG and compare model gene rankings on explicit backgrounds."""
import argparse
import json
from pathlib import Path
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from benchmark_paths import BENCH_ROOT

MODELS = ["state_15b", "esm2_650m", "qwen"]


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def read_cell(folder, index):
    with np.load(folder / "cells" / f"{index:05d}.npz", allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def scope_indices(cells):
    result = {"all": np.arange(len(cells))}
    for label in [0, 1]:
        result["label_" + str(label)] = np.flatnonzero(cells.y.eq(label))
    if "drug" in cells and set(cells.drug) & {"alectinib", "crizotinib"}:
        for drug in ["crizotinib", "alectinib"]:
            result[drug] = np.flatnonzero(cells.drug.isin(["untreated", drug]))
            for label in [0, 1]:
                result[drug + "_label_" + str(label)] = np.flatnonzero(cells.drug.isin(["untreated", drug]) & cells.y.eq(label))
    return {key: value for key, value in result.items() if len(value)}


def report(args):
    summary_rows, pair_rows, cell_rows, diagnostics, selection = [], [], [], [], []
    folders = sorted(args.open_path.glob("*/*"))
    for trial in folders:
        if not trial.is_dir() or not (trial / MODELS[0] / "cells.tsv").is_file():continue
        dataset, fold = trial.parent.name, trial.name
        if args.datasets and dataset not in args.datasets:continue
        cells = pd.read_csv(trial / MODELS[0] / "cells.tsv", sep="\t", keep_default_na=False)
        genes = pd.read_csv(trial / MODELS[0] / "sensitive_reference.tsv", sep="\t").gene_symbol.astype(str).to_numpy()
        tables = {}
        for model in MODELS:
            folder = trial / model
            other = pd.read_csv(folder / "cells.tsv", sep="\t", keep_default_na=False)
            if cells.cell_id.tolist() != other.cell_id.tolist() or cells.y.tolist() != other.y.tolist():
                raise ValueError("Model cells differ: " + str(trial))
            if len(list((folder / "cells").glob("*.npz"))) != len(cells):
                raise ValueError("Attribution job incomplete: " + str(folder))
            matrices = {baseline: np.zeros((len(cells), len(genes)), dtype=np.float64)
                        for baseline in ["zero", "sensitive"]}
            presence = np.zeros((len(cells), len(genes)), dtype=bool)
            for i, row in enumerate(cells.itertuples()):
                data = read_cell(folder, i)
                if str(data["cell_id"]) != row.cell_id:raise ValueError("Cell order mismatch")
                ids = data["gene_id"]
                presence[i, ids] = True
                for baseline in matrices:matrices[baseline][i, ids] = data[baseline + "_signed_ig"]
                for item in json.loads(str(data["diagnostics"])):
                    diagnostics.append(dict(dataset=dataset, fold=fold, model=model,
                                            reference_missing_fraction=float(data["reference_missing"].mean()), **item))
            tables[model] = (matrices, presence)
        for a, b in combinations(MODELS, 2):
            if not np.array_equal(tables[a][1], tables[b][1]):raise ValueError("Node backgrounds differ")
            for baseline in ["zero", "sensitive"]:
                for i, row in enumerate(cells.itertuples()):
                    ids = tables[a][1][i] & tables[b][1][i]
                    x, y = tables[a][0][baseline][i, ids], tables[b][0][baseline][i, ids]
                    cell_rows.append(dict(dataset=dataset, fold=fold, cell_id=row.cell_id, label=row.y,
                                          baseline=baseline, model_a=a, model_b=b, n_genes=int(ids.sum()),
                                          signed_rho=correlation(x, y), absolute_rho=correlation(abs(x), abs(y))))
        for scope, index in scope_indices(cells).items():
            means = {}
            for model in MODELS:
                matrices, presence = tables[model]
                n_present = presence[index].sum(0)
                for baseline in ["zero", "sensitive"]:
                    values = matrices[baseline][index]
                    mass = abs(values).sum(1)
                    relative = np.divide(values, mass[:, None], out=np.zeros_like(values), where=mass[:, None] > 0) * 1000
                    fields = dict(mean_signed_ig=values.mean(0), mean_absolute_ig=abs(values).mean(0),
                                  mean_relative_absolute_ig=abs(relative).mean(0), mean_relative_signed_ig=relative.mean(0))
                    means[(model, baseline)] = fields
                    for gene in np.flatnonzero(n_present):
                        summary_rows.append(dict(dataset=dataset, fold=fold, scope=scope, model=model, baseline=baseline,
                            gene_symbol=genes[gene], n_cells=len(index), n_present=int(n_present[gene]),
                            fraction_present=float(n_present[gene] / len(index)),
                            **{key: float(value[gene]) for key, value in fields.items()}))
            fraction = tables[MODELS[0]][1][index].mean(0)
            for background, keep in [("all_observed", fraction > 0), ("presence_10pct", fraction >= .1)]:
                for baseline in ["zero", "sensitive"]:
                    for a, b in combinations(MODELS, 2):
                        for metric in ["mean_relative_absolute_ig", "mean_signed_ig"]:
                            x, y = means[(a, baseline)][metric][keep], means[(b, baseline)][metric][keep]
                            pair_rows.append(dict(dataset=dataset, fold=fold, scope=scope, baseline=baseline,
                                background=background, metric=metric, model_a=a, model_b=b, n_cells=len(index),
                                n_genes=int(keep.sum()), rho=correlation(x, y)))
        selection.append(cells.assign(dataset=dataset, fold=fold))
        print("SUMMARIZED", dataset, fold, len(cells), flush=True)
    args.save_path.mkdir(parents=True, exist_ok=True)
    for filename, records in [("gene_importance", summary_rows), ("spearman_gene_mean", pair_rows),
                              ("spearman_per_cell", cell_rows), ("ig_diagnostics", diagnostics)]:
        pd.DataFrame(records).to_csv(args.save_path / (filename + ".tsv"), sep="\t", index=False)
    pd.concat(selection).to_csv(args.save_path / "selected_cells.tsv", sep="\t", index=False)
    gene_table = pd.DataFrame(summary_rows)
    contrasts = []
    group_keys = ["dataset", "fold", "model", "gene_symbol", "scope"]
    sensitive = gene_table[gene_table.baseline.eq("sensitive") & gene_table.scope.str.endswith("label_0")].set_index(group_keys)
    resistant = gene_table[gene_table.baseline.eq("sensitive") & gene_table.scope.str.endswith("label_1")]
    for row in resistant.itertuples():
        if row.dataset == "GSE223779_TS485T" and row.scope == "label_1":continue
        key = (row.dataset, row.fold, row.model, row.gene_symbol, row.scope[:-1] + "0")
        lower = float(sensitive.loc[key, "mean_signed_ig"]) if key in sensitive.index else 0.
        contrasts.append(dict(dataset=row.dataset, scope=row.scope, fold=row.fold, model=row.model,
                              gene_symbol=row.gene_symbol, positive_mean=row.mean_signed_ig,
                              negative_mean=lower, contrast=row.mean_signed_ig-lower,
                              fraction_present_positive=row.fraction_present,
                              supports_positive=row.mean_signed_ig>0 and row.mean_signed_ig>lower and row.fraction_present>=.2))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(args.save_path / "directional_gene_contrasts.tsv", sep="\t", index=False)
    support = contrasts.groupby(["dataset", "scope", "gene_symbol"]).agg(
        n_model_fold_support=("supports_positive", "sum"), n_model_fold_observed=("supports_positive", "size"),
        mean_positive_ig=("positive_mean", "mean"), mean_label_contrast=("contrast", "mean")).reset_index()
    support["n_model_fold_expected"] = support.dataset.map({dataset:3*len(set(gene_table.loc[gene_table.dataset.eq(dataset), "fold"])) for dataset in support.dataset.unique()})
    support["consistent_across_models_and_folds"] = support.n_model_fold_support.eq(support.n_model_fold_expected)
    support.sort_values(["dataset", "consistent_across_models_and_folds", "n_model_fold_support", "mean_label_contrast"], ascending=[True, False, False, False]).to_csv(args.save_path / "directional_gene_consensus.tsv", sep="\t", index=False)

    pd.DataFrame(diagnostics).groupby(["dataset", "fold", "model", "baseline"]).agg(
        n_cells=("cell_id", "size"), n_converged=("converged", "sum"),
        median_relative_error=("relative_completeness_error", "median"), max_relative_error=("relative_completeness_error", "max"),
        mean_reference_missing_fraction=("reference_missing_fraction", "mean")).to_csv(args.save_path / "convergence_summary.tsv", sep="\t")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--open_path", type=Path, default=BENCH_ROOT / "output/attribution")
    p.add_argument("--save_path", type=Path, default=BENCH_ROOT / "output/attribution/summary")
    p.add_argument("--datasets", nargs="+")
    report(p.parse_args())


if __name__ == "__main__":main()

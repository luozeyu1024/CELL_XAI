"""Render PDF attribution comparisons from saved tables and optimized edge masks."""
import argparse
from pathlib import Path
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from benchmark_paths import BENCH_ROOT

MODELS = ["state_15b", "esm2_650m", "qwen"]
NAMES = {"state_15b": "STATE-15B", "esm2_650m": "ESM2-650M", "qwen": "Qwen"}
COLORS = {"state_15b": "#CA5596", "esm2_650m": "#DB9735", "qwen": "#007580"}
PAIRS = list(combinations(MODELS, 2))
CMAP = LinearSegmentedColormap.from_list("rho", ["#CA5596", "#FFFFFF", "#67A8AF"])
CMAP.set_bad("#E5E5E5")
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})


def short(dataset):
    if "PC9" in dataset:return "PC9 DTEP"
    if "carboplatin" in dataset:return "ReSis carboplatin"
    if "olaparib" in dataset:return "ReSis olaparib"
    if "crizotinib" in dataset:return "ALK TS485T / crizotinib"
    if "alectinib" in dataset:return "ALK TS485T / alectinib"
    return "ALK TS485T"


def gene_heatmaps(table, out):
    selected = table[table.background.eq("presence_10pct") & table.scope.isin(["all", "crizotinib", "alectinib"])]
    selected = selected[~(selected.dataset.eq("GSE223779_TS485T") & selected.scope.eq("all"))]
    rows = sorted(set(zip(selected.dataset, selected.fold, selected.scope)))
    for baseline in ["zero", "sensitive"]:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, max(3.8, .48 * len(rows))), layout="constrained")
        for ax, metric, title in zip(axes, ["mean_relative_absolute_ig", "mean_signed_ig"],
                                    ["Mean relative |IG|", "Mean signed IG"]):
            matrix = np.full((len(rows), 3), np.nan)
            for r, (dataset, fold, scope) in enumerate(rows):
                for c, (a, b) in enumerate(PAIRS):
                    part = selected[(selected.dataset == dataset) & (selected.fold == fold) & (selected.scope == scope) &
                                    (selected.baseline == baseline) & (selected.metric == metric) &
                                    (selected.model_a == a) & (selected.model_b == b)]
                    if len(part) != 1:raise ValueError("Ambiguous heatmap cell")
                    rho, n = part.iloc[0].rho, int(part.iloc[0].n_genes)
                    matrix[r, c] = rho
                    ax.text(c, r, f"{rho:.2f}\nn={n:,}" if np.isfinite(rho) else "NA", ha="center", va="center", fontsize=7)
            image = ax.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap=CMAP, aspect="auto")
            ax.set_xticks(range(3), [NAMES[a] + " / " + NAMES[b] for a, b in PAIRS], rotation=35, ha="right")
            ax.set_yticks(range(len(rows)), [short(d) + " / " + (f if s == "all" else s) for d, f, s in rows])
            ax.set_title(title)
            ax.tick_params(length=0)
        fig.colorbar(image, ax=axes, label="Spearman rho", shrink=.7)
        fig.suptitle("Zero expression baseline" if baseline == "zero" else "Source training-sensitive baseline")
        fig.savefig(out / f"01_gene_spearman_{baseline}.pdf", bbox_inches="tight");plt.close(fig)


def cell_heatmap(table, out):
    rows = sorted(set(zip(table.dataset, table.fold)))
    fig, axes = plt.subplots(2, 2, figsize=(9, max(6, .8 * len(rows))), layout="constrained")
    for ax, (baseline, metric) in zip(axes.flat, [(b, m) for b in ["zero", "sensitive"] for m in ["absolute_rho", "signed_rho"]]):
        values = np.full((len(rows), 3), np.nan)
        for i, (dataset, fold) in enumerate(rows):
            for j, (a, b) in enumerate(PAIRS):
                part = table[(table.dataset == dataset) & (table.fold == fold) & (table.baseline == baseline) &
                             (table.model_a == a) & (table.model_b == b)][metric].dropna()
                median, q1, q3 = part.median(), part.quantile(.25), part.quantile(.75)
                values[i, j] = median
                ax.text(j, i, f"{median:.2f}\n[{q1:.2f}, {q3:.2f}]", ha="center", va="center", fontsize=7)
        image = ax.imshow(np.ma.masked_invalid(values), cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3), [NAMES[a] + " / " + NAMES[b] for a, b in PAIRS], rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), [short(d) + " / " + f for d, f in rows])
        ax.set_title(baseline.title() + " / " + metric.replace("_rho", " IG"));ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, label="Median per-cell Spearman rho", shrink=.7)
    fig.savefig(out / "02_cellwise_spearman.pdf", bbox_inches="tight");plt.close(fig)


def gene_rankings(table, out):
    data = table[table.baseline.eq("sensitive") & table.scope.str.endswith("label_1")].copy()
    data = data[~(data.dataset.eq("GSE223779_TS485T") & data.scope.eq("label_1"))].copy()
    data.loc[data.dataset.eq("GSE223779_TS485T"), "dataset"] = data.loc[data.dataset.eq("GSE223779_TS485T"), "dataset"] + "__" + data.loc[data.dataset.eq("GSE223779_TS485T"), "scope"].str.replace("_label_1", "", regex=False)
    data = data.assign(weighted=data.mean_signed_ig * data.n_cells)
    grouped = data.groupby(["dataset", "model", "gene_symbol"], as_index=False).agg(
        total=("weighted", "sum"), n_cells=("n_cells", "sum"), n_present=("n_present", "sum"))
    denominators = data[["dataset", "model", "fold", "scope", "n_cells"]].drop_duplicates().groupby(["dataset", "model"]).n_cells.sum()
    grouped["n_cells"] = [int(denominators.loc[(row.dataset, row.model)]) for row in grouped.itertuples()]
    grouped["mean_signed_ig"] = grouped.total / grouped.n_cells
    grouped["fraction_present"] = grouped.n_present / grouped.n_cells
    datasets = sorted(grouped.dataset.unique())
    fig, axes = plt.subplots(len(datasets), 3, figsize=(10, 2.7 * len(datasets)), squeeze=False, layout="constrained")
    for i, ds in enumerate(datasets):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            part = grouped[(grouped.dataset == ds) & (grouped.model == model) & grouped.fraction_present.ge(.2) & grouped.mean_signed_ig.gt(0)]
            part = part.sort_values(["mean_signed_ig", "gene_symbol"], ascending=[False, True]).head(8).iloc[::-1]
            if len(part):ax.barh(part.gene_symbol, part.mean_signed_ig, color=COLORS[model])
            else:ax.text(.5, .5, "No positive-mean genes", transform=ax.transAxes, ha="center")
            ax.set_title(short(ds) + " / " + NAMES[model], fontsize=8)
            ax.set_xlabel("Mean signed IG (label 1)", fontsize=7)
    fig.savefig(out / "03_positive_gene_IG_sensitive.pdf", bbox_inches="tight");plt.close(fig)
    grouped.to_csv(out.parent / "summary/pooled_label1_gene_rankings.tsv", sep="\t", index=False)


def networks(root, out):
    for trial in sorted(root.glob("*/*")):
        if not (trial / MODELS[0] / "gnn_class_0.npz").exists():continue
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), layout="constrained")
        for label in [0, 1]:
            for j, model in enumerate(MODELS):
                folder = trial / model
                with np.load(folder / f"gnn_class_{label}.npz", allow_pickle=False) as z:
                    ids, edge, masks = z["gene_id"], z["edge_index"], z["edge_mask"]
                    pairs = {}
                    for k, (u, v) in enumerate(edge.T):
                        if u == v:continue
                        key = tuple(sorted((int(u), int(v))))
                        pairs.setdefault(key, []).append(float(masks[k]))
                    ranked = sorted(((pair, float(np.mean(value))) for pair, value in pairs.items()), key=lambda item: -item[1])[:15]
                    graph = nx.Graph()
                    for (u, v), weight in ranked:graph.add_edge(int(ids[u]), int(ids[v]), weight=weight)
                    genes = pd.read_csv(folder / "sensitive_reference.tsv", sep="\t").gene_symbol.astype(str).to_numpy()
                    ax = axes[label, j]
                    if len(graph):
                        components = sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), min(nodes)))
                        columns = 3
                        pos, text_pos = {}, {}
                        for number, nodes in enumerate(components):
                            center = np.array([number % columns, -(number // columns)], dtype=float)
                            local = nx.spring_layout(graph.subgraph(nodes), seed=42, scale=.30)
                            for k, node in enumerate(sorted(nodes)):
                                pos[node] = local[node] + center
                                text_pos[node] = pos[node] + np.array([0, .15 if k % 2 == 0 else -.15])
                        nx.draw_networkx_nodes(graph, pos, node_color=COLORS[model], node_size=90, alpha=.8, ax=ax)
                        nx.draw_networkx_edges(graph, pos, width=[.4 + 2 * graph[u][v]["weight"] for u, v in graph.edges], edge_color="#777777", alpha=.6, ax=ax)
                        nx.draw_networkx_labels(graph, text_pos, labels={node: genes[node] for node in graph}, font_size=5.5, ax=ax)
                    cell_table = pd.read_csv(folder / "cells.tsv", sep="\t", keep_default_na=False).set_index("cell_id")
                    row = cell_table.loc[str(z["cell_id"])]
                    condition = str(row.get("drug", "")) if "TS485T" in trial.parent.name else ""
                    ax.set_title(NAMES[model] + f" / true {label}, predicted {int(z['target_class'])}" + (" / " + condition if condition else ""), fontsize=8)
                    ax.axis("off")
        fig.suptitle(short(trial.parent.name) + " / " + trial.name + " — top 15 undirected mask pairs", fontsize=10)
        fig.savefig(out / ("04_edge_subgraphs_" + trial.parent.name + "_" + trial.name + ".pdf"), bbox_inches="tight");plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open_path", type=Path, default=BENCH_ROOT / "output/attribution")
    parser.add_argument("--save_path", type=Path, default=BENCH_ROOT / "output/attribution/figures")
    a = parser.parse_args();a.save_path.mkdir(parents=True, exist_ok=True)
    summary = a.open_path / "summary"
    gene_heatmaps(pd.read_csv(summary / "spearman_gene_mean.tsv", sep="\t"), a.save_path)
    cell_heatmap(pd.read_csv(summary / "spearman_per_cell.tsv", sep="\t"), a.save_path)
    gene_rankings(pd.read_csv(summary / "gene_importance.tsv", sep="\t"), a.save_path)
    networks(a.open_path, a.save_path)


if __name__ == "__main__":main()

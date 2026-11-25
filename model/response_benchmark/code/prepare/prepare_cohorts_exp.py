"""Attach fixed split assignments to an ordered cell graph package."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_paths
from split_registry import apply_splits, assignments


def prepare(graph_root, output_root, dataset, mode, state_h5ad=None, state_key="X_state"):
    graph = (graph_root / dataset).resolve()
    cells = pd.read_csv(graph / "cells.tsv", sep="\t", dtype={"cell_id": str}, keep_default_na=False)
    cells = cells.drop(columns="y", errors="ignore")
    cells["graph_row"] = np.arange(len(cells))
    metadata_path = graph / "source_metadata.tsv"
    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path, sep="\t", dtype={"cell_id": str}, keep_default_na=False)
        extra = [column for column in metadata.columns if column not in cells and column not in {"y", "split", "response"}]
        cells = cells.merge(metadata[["cell_id", *extra]], on="cell_id", how="left", sort=False, validate="one_to_one")
    registered_ids = assignments(dataset, mode=mode).cell_id
    cells = apply_splits(cells.loc[cells.cell_id.isin(registered_ids)], dataset, mode=mode)
    cells["response"] = cells.y.map({0: "sensitive", 1: "resistant", -1: "unknown"})
    if "has_signature" in cells and not cells.has_signature.eq(True).all():
        raise ValueError("Registered cells without a graph signature")
    info = {"graph_source": str(graph), "top_k": json.loads((graph / "parameters.json").read_text(encoding="utf-8"))["top_k"]}
    if state_h5ad is not None:
        import anndata as ad

        state = ad.read_h5ad(state_h5ad, backed="r")
        try:
            ids = pd.Index(state.obs_names.astype(str))
            rows = ids.get_indexer(cells.cell_id)
            if (rows < 0).any():
                raise ValueError("Registered cells missing from the SE embeddings")
            values = np.asarray(state.obsm[state_key])[rows]
            if values.ndim != 2 or values.shape[1] != 2058 or not np.isfinite(values).all():
                raise ValueError("Expected 2058 finite SE features per cell")
        finally:
            state.file.close()
        info.update(state_source=str(state_h5ad.resolve()), state_key=state_key, state_dimensions=2048)
    dest = output_root / "data/cohorts" / dataset
    dest.mkdir(parents=True, exist_ok=True)
    if state_h5ad is not None:
        np.save(dest / "state_cell_embeddings.npy", values[:, :2048].astype(np.float32))
    cells.to_csv(dest / "cells.tsv", sep="\t", index=False)
    pd.crosstab(cells.split, cells.response).to_csv(dest / "split_counts.tsv", sep="\t")
    (dest / "data_sources.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open_path", type=Path, required=True, help="Cell graph package root")
    parser.add_argument("--save_path", type=Path, required=True, help="Response model run directory")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["auto", "source", "infer"], default="auto")
    parser.add_argument("--state_h5ad", type=Path)
    parser.add_argument("--state_key", default="X_state")
    args = parser.parse_args()
    prepare(args.open_path, args.save_path, args.dataset, args.mode, args.state_h5ad, args.state_key)


if __name__ == "__main__":
    main()

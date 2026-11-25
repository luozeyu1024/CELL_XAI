"""Align a protein feature matrix to a UniProt inventory by accession."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def state_features(path, hgnc, inventory):
    if path.suffix.lower() == ".pt":
        import torch

        vectors = torch.load(path, map_location="cpu", weights_only=True)
        symbols = sorted(vectors)
        matrix = np.stack([vectors[key].detach().cpu().numpy().reshape(-1) for key in symbols])
    else:
        import h5py

        with h5py.File(path, "r") as source:
            symbols = source["obs/gene_symbol"].asstr()[:]
            matrix = source["X"][:]
    index = pd.Index(symbols)
    if not index.is_unique or matrix.ndim != 2 or len(index) != len(matrix):
        raise ValueError("Invalid STATE gene axis")
    approved = pd.read_csv(hgnc, sep="\t", dtype=str, keep_default_na=False)
    approved = approved.loc[approved.status.eq("Approved") & approved.symbol.isin(index)]
    by_accession = defaultdict(set)
    for row in approved.itertuples(index=False):
        for accession in row.uniprot_ids.split("|"):
            if accession:
                by_accession[accession].add(row.symbol)
    pairs = [(accession, next(iter(by_accession[accession])))
             for accession in inventory.index if len(by_accession[accession]) == 1]
    if not pairs:
        raise ValueError("No unambiguous approved STATE matches in the inventory")
    mapping = pd.DataFrame(pairs, columns=["accession", "hgnc_symbol"])
    mapping["state_row"] = index.get_indexer(mapping.hgnc_symbol)
    return mapping.accession.to_numpy(dtype=str), np.asarray(matrix[mapping.state_row], dtype=np.float32), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--features", type=Path, help="NPZ with ids and embeddings")
    inputs.add_argument("--state_h5ad", type=Path, help="STATE gene embedding H5AD")
    inputs.add_argument("--state_pt", type=Path, help="Official SE-600M protein_embeddings.pt")
    inputs.add_argument("--esm2_matrix", type=Path, help="Original ESM2 feature matrix in NPY format")
    parser.add_argument("--esm2_columns", type=Path, help="XLSX feature column index for the ESM2 matrix")
    parser.add_argument("--hgnc_path", type=Path)
    parser.add_argument("--inventory", type=Path, required=True, help="UniProt TSV or XLSX with Entry and Gene Names")
    parser.add_argument("--save_path", type=Path, required=True)
    args = parser.parse_args()

    if args.inventory.suffix.lower() == ".xlsx":
        inventory = pd.read_excel(args.inventory, dtype=str, keep_default_na=False)
    else:
        inventory = pd.read_csv(args.inventory, sep="\t", dtype=str, keep_default_na=False)
    inventory = inventory.set_index("Entry", verify_integrity=True)
    mapping = None
    state_source = args.state_h5ad or args.state_pt
    if state_source:
        if args.hgnc_path is None:
            parser.error("STATE inputs require --hgnc_path")
        ids, embeddings, mapping = state_features(state_source, args.hgnc_path, inventory)
    elif args.esm2_matrix:
        if args.esm2_columns is None:
            parser.error("--esm2_matrix requires --esm2_columns")
        columns = pd.read_excel(args.esm2_columns).iloc[:, 0].astype(str)
        features = columns[columns.str.startswith("ESM2_")].tolist()
        mean = [i for i, name in enumerate(features) if name.startswith("ESM2_mean")]
        matrix = np.load(args.esm2_matrix, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (len(inventory), len(features)) or len(mean) != 1280:
            parser.error("Expected aligned rows and 1,280 mean-pooled ESM2-650M columns")
        ids = inventory.index.to_numpy(dtype=str)
        embeddings = np.asarray(matrix[:, mean], dtype=np.float32)
    else:
        with np.load(args.features, allow_pickle=False) as data:
            ids = data["ids"].astype(str)
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if ids.ndim != 1 or embeddings.ndim != 2 or len(ids) != len(embeddings):
        parser.error("Expected one accession per embedding row")
    if not pd.Index(ids).is_unique or np.any(ids == ""):
        parser.error("Feature accessions must be unique and nonempty")
    missing = pd.Index(ids).difference(inventory.index)
    if len(missing):
        parser.error(f"Accessions absent from inventory: {missing[:5].tolist()}")
    aligned = inventory.loc[ids]
    output = {"ids": ids, "embeddings": embeddings, "gene_names": aligned["Gene Names"].to_numpy(dtype=str)}
    if "Sequence" in aligned:
        output["sequences"] = aligned["Sequence"].to_numpy(dtype=str)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save_path, **output)
    if mapping is not None:
        mapping.to_csv(args.save_path.with_name("protein_gene_mapping.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()

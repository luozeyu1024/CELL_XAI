"""Read an explicitly selected expression layer; never write or normalize data."""
from pathlib import Path
import argparse


def load_preprocessed_h5ad(path, layer="logNor", gene_column=None):
    """Return AnnData with selected expression in X; preserve sparse storage/order.

    gene_column=None keeps var_names. A supplied column replaces var_names in
    memory only. Missing layers and duplicate/empty gene names raise errors.
    """
    import anndata as ad

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    if layer != "X":
        if layer not in adata.layers:
            raise KeyError(f"Missing layer {layer!r}; available: {list(adata.layers.keys())}")
        adata.X = adata.layers[layer].copy()
    if adata.X is None:
        raise ValueError("Selected expression matrix is empty (None)")
    if gene_column is not None:
        if gene_column not in adata.var:
            raise KeyError(f"Missing gene column: {gene_column}")
        names = adata.var[gene_column]
        if names.isna().any():
            raise ValueError("Gene column contains missing names")
        adata.var_names = names.astype(str)
    if not adata.var_names.is_unique:
        raise ValueError("Duplicate gene names; resolve their mapping before X @ E")
    if any(not str(name).strip() for name in adata.var_names):
        raise ValueError("Empty gene name")
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--layer", default="logNor", help="Layer name, or X; no automatic fallback")
    parser.add_argument("--gene-column", default=None, help="Optional var column; default keeps var_names")
    args = parser.parse_args()
    adata = load_preprocessed_h5ad(args.h5ad, args.layer, args.gene_column)
    print(f"cells={adata.n_obs}, genes={adata.n_vars}, selected={args.layer}")
    print(f"matrix_type={type(adata.X).__name__}, layers={list(adata.layers.keys())}")
    print(f"gene_names={adata.var_names[:5].tolist()}")
    print(f"condition_columns={[k for k in ('perturbation', 'gene', 'condition') if k in adata.obs]}")
    print("Read only: no normalization, embedding computation, or file output.")


if __name__ == "__main__":
    main()

"""Validate local SE safetensors loading and encode a count-compatible GSE."""
import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import load_file
from scipy import sparse
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--counts_save_path", help="Optional separate path for the count-compatible input H5AD")
    parser.add_argument("--state_h5ad_path", help="Optional separate path for the count-containing encoded H5AD")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_cells", type=int, default=0, help="0: all cells")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    from state.emb.nn.model import StateEmbeddingModel
    from state.emb.inference import Inference
    from state.emb.utils import get_precision_config

    out = Path(args.save_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.model_path).resolve()
    source = Path(args.open_path).resolve()
    data = ad.read_h5ad(source)
    if args.max_cells:
        data = data[:args.max_cells].copy()
    matrix = data.layers["counts"] if "counts" in data.layers else data.X
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not (np.isfinite(values).all() and (values >= 0).all()
            and np.allclose(values, np.rint(values), atol=1e-6, rtol=0)
            and np.any(values > 0)):
        raise ValueError("Input is not finite nonnegative integer-compatible counts")
    if "counts" not in data.layers and "log1p" in data.uns:
        raise ValueError("X has a log1p annotation and no explicit counts layer")
    data.X = sparse.csr_matrix(matrix, dtype=np.float32)
    data.var["gene_name"] = data.var_names.astype(str)
    input_path = Path(args.counts_save_path).resolve() if args.counts_save_path else out / "input_counts.h5ad"
    output_path = Path(args.state_h5ad_path).resolve() if args.state_h5ad_path else out / "state_embeddings.h5ad"
    input_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Choose a new output folder: {output_path}")
    data.write_h5ad(input_path, compression="gzip")
    protein = torch.load(checkpoint / "protein_embeddings.pt", map_location="cpu", weights_only=True)
    overlap = sum(g in protein for g in data.var_names)
    if not overlap:
        raise ValueError("No genes match the model protein vocabulary")
    cfg = OmegaConf.load(checkpoint / "config.yaml")
    cfg.model.batch_size = args.batch_size  # encode_adata uses cfg, not its batch_size argument.
    cfg.wandb.enable = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = get_precision_config(device)
    print(f"Device={device}, precision={precision}, cells={data.n_obs}, mapped genes={overlap}", flush=True)
    model = StateEmbeddingModel(token_dim=cfg.tokenizer.token_dim,
        d_model=cfg.model.emsize, nhead=cfg.model.nhead, d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers, output_dim=cfg.model.output_dim, dropout=0.0, cfg=cfg)
    weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    # safetensors stores shared encoder parameters once; the module has both names.
    assert model.gene_embedding_layer is model.encoder
    protein_matrix = torch.stack(list(protein.values()))
    assert torch.equal(weights["pe_embedding.weight"], protein_matrix), "Protein table/order mismatch"
    model.pe_embedding = torch.nn.Embedding.from_pretrained(protein_matrix)
    restored_aliases = []
    for key in model.state_dict():
        if key.startswith("gene_embedding_layer.") and key not in weights:
            canonical = key.replace("gene_embedding_layer.", "encoder.", 1)
            weights[key] = weights[canonical]
            restored_aliases.append(key)
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(device=device, dtype=precision).eval()
    inferer = Inference(cfg=cfg, protein_embeds=protein)
    inferer.model = model
    start = time.perf_counter()
    inferer.encode_adata(input_adata_path=str(input_path), output_adata_path=str(output_path),
        emb_key="X_state", batch_size=args.batch_size, gene_column="gene_name")
    elapsed = time.perf_counter() - start
    result = ad.read_h5ad(output_path)
    emb = np.asarray(result.obsm["X_state"])
    assert result.obs_names.equals(data.obs_names), "Cell order changed"
    assert result.var_names.equals(data.var_names), "Gene order changed"
    assert emb.shape == (data.n_obs, cfg.model.output_dim + 10)
    assert np.isfinite(emb).all() and emb.std() > 0
    unique_cells = np.unique(emb[:, :cfg.model.output_dim], axis=0).shape[0]
    if len(emb) > 1:
        assert unique_cells > 1, "Cell embeddings collapsed to a single vector"
    np.save(out / "cell_embeddings.npy", emb[:, :cfg.model.output_dim])
    np.save(out / "dataset_embeddings.npy", emb[:, cfg.model.output_dim:])
    result.obs.to_csv(out / "cell_metadata.tsv", sep="\t", index_label="cell_id")
    report = {
        "status": "passed", "source_h5ad": str(source), "checkpoint": str(checkpoint),
        "input_counts_h5ad": str(input_path), "encoded_counts_h5ad": str(output_path),
        "cells": data.n_obs, "genes": data.n_vars, "mapped_genes": overlap,
        "cell_embedding_shape": list(emb[:, :cfg.model.output_dim].shape),
        "dataset_embedding_shape": list(emb[:, cfg.model.output_dim:].shape),
        "checkpoint_loading": "strict=True", "device": device, "precision": str(precision),
        "shared_encoder_alias_keys_restored": len(restored_aliases),
        "protein_table_matches_companion": True,
        "batch_size": args.batch_size, "seed": args.seed, "inference_seconds": elapsed,
        "cell_order_preserved": True, "finite_embeddings": True,
        "unique_cell_embeddings": unique_cells,
        "python": sys.executable, "state_source": str(Path(sys.modules["state"].__file__).resolve()),
        "versions": {p: importlib.metadata.version(p) for p in
            ["arc-state", "torch", "numpy", "scanpy", "anndata", "lightning", "transformers"]},
        "scope": "SE cell embedding inference only; no ST or resistance classifier"}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Run gene-level attribution for SCimilarity or scVI on one h5ad.

Pretrained SCimilarity and CZI scVI, plus local scVI with standard Seurat v3 HVG.
All cells are analyzed unless --max-cells explicitly requests a pilot subset.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

import sig_attr_common as C


# --------------------------------------------------------------------------
def _gpu_mem() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def _run_scimilarity(args, adata):
    from SIGnature.models.scimilarity import SCimilarityWrapper

    wrapper = SCimilarityWrapper(model_path=args.model_dir, use_gpu=args.use_gpu)
    t0 = time.time()
    tdata = wrapper.preprocess_adata(adata.copy())
    prep_s = time.time() - t0

    npz = Path(args.out_dir) / "attributions" / f"scimilarity_{args.method}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    atts = wrapper.calculate_attributions(
        tdata.X,
        method=args.method,
        batch_size=args.batch_size,
        npz_path=str(npz),
    )
    calc_s = time.time() - t1

    model_info = {
        "wrapper": "SIGnature.models.scimilarity.SCimilarityWrapper",
        "model_dir": str(Path(args.model_dir).resolve()),
        "n_model_genes": len(wrapper.gene_order),
        "latent_dim": wrapper.latent_dim,
        "preprocessed_shape": list(tdata.shape),
    }
    return atts, tdata.var_names, model_info, prep_s, calc_s, str(npz)


def _run_scvi_local(args, adata):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps"))
    import scanpy as sc
    import scvi
    from SIGnature.models.scvi import SCVIWrapper
    scvi.settings.seed = args.seed
    t0 = time.time()
    work = adata.copy()
    sc.pp.highly_variable_genes(work, layer="counts", flavor="seurat_v3",
                                n_top_genes=args.scvi_hvg, subset=False)
    selected = work.var["highly_variable"].to_numpy()
    work = work[:, selected].copy()
    scvi.model.SCVI.setup_anndata(work, layer="counts")
    model = scvi.model.SCVI(work, n_latent=16, n_hidden=128, n_layers=2)
    model.train(max_epochs=args.scvi_epochs, early_stopping=True,
                accelerator="gpu" if args.use_gpu else "cpu",
                enable_progress_bar=False, enable_checkpointing=False)
    model_dir = Path(args.out_dir) / "scvi_local" / "scvi.model"
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir), overwrite=True)
    wrapper = C.build_local_scvi_wrapper(SCVIWrapper)(str(model_dir), use_gpu=args.use_gpu)
    wrapper.model = model.module.z_encoder
    wrapper.gene_order = work.var_names.tolist()
    prep_s = time.time() - t0
    npz = Path(args.out_dir) / "attributions" / f"scvi_local_{args.method}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    atts = wrapper.calculate_attributions(work.layers["counts"], method=args.method,
                                         batch_size=args.batch_size, npz_path=str(npz), n_steps=50)
    model_info = dict(wrapper="LocalSCVIWrapper", trained_on_input=True,
                      hvg_flavor="seurat_v3", hvg_input="counts", hvg_n=work.n_vars,
                      n_latent=16, max_epochs=args.scvi_epochs,
                      model_dir=str(model_dir.resolve()), encoder_transform="identity_official_signature")
    return atts, work.var_names, model_info, prep_s, time.time()-t1, str(npz)


def _run_scvi(args, adata):
    from SIGnature.models.scvi import SCVIWrapper


    npz = Path(args.out_dir) / "attributions" / f"scvi_czi_{args.method}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)

    wrapper = SCVIWrapper(model_path=args.scvi_model_dir, use_gpu=args.use_gpu)
    t0 = time.time()
    tdata = wrapper.preprocess_adata(
        adata.copy(), ensembl_gene_file=args.ensembl_gene_file or None
    )
    prep_s = time.time() - t0

    t1 = time.time()
    atts = wrapper.calculate_attributions(
        tdata,
        method=args.method,
        batch_size=args.batch_size,
        npz_path=str(npz),
    )
    calc_s = time.time() - t1

    model_info = {
        "wrapper": "SIGnature.models.scvi.SCVIWrapper",
        "model_dir": str(Path(args.scvi_model_dir).resolve()),
        "preprocessed_shape": list(tdata.shape),
        "checkpoint": "2023-12-15-scvi-homo-sapiens",
        "trained_on_input": False,
        "cat_value": wrapper.cat_value,
        "gene_space": "fixed_checkpoint_features",
        "n_latent": 200,
        "n_model_genes": tdata.n_vars,
        "n_measured_model_genes": int(tdata.var["symbol"].notna().sum()) if "symbol" in tdata.var else None,
    }
    import pandas as pd
    from collections import Counter
    mapping = pd.read_table(args.ensembl_gene_file).dropna(subset=["HGNC symbol"])
    reverse = dict(zip(mapping["Gene stable ID"], mapping["HGNC symbol"]))
    symbols = tdata.var["symbol"] if "symbol" in tdata.var else [None] * tdata.n_vars
    names = [str(g) if pd.notna(g) and str(g) else str(reverse.get(str(e), e)) for e, g in zip(tdata.var_names, symbols)]
    counts = Counter(names)
    names = [str(e) if counts[g] > 1 else g for e, g in zip(tdata.var_names, names)]
    C.write_json(npz.with_name(f"scvi_czi_{args.method}_feature_mapping.json"),
                 [{"feature_id": str(e), "gene": g} for e, g in zip(tdata.var_names, names)])
    return atts, names, model_info, prep_s, calc_s, str(npz)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["scimilarity", "scvi_czi", "scvi_local"])
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", help="SCimilarity model directory")
    ap.add_argument("--scvi-model-dir", help="CZI Census scVI checkpoint directory")
    ap.add_argument("--ensembl_gene_file", help="Ensembl-to-symbol mapping for the CZI checkpoint")
    ap.add_argument("--method", default="ig", choices=["ig", "dl", "ixg"])
    ap.add_argument("--scvi-hvg", type=int, default=2000)
    ap.add_argument("--scvi-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--counts-policy", choices=C.COUNTS_POLICIES, default="auto")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    if args.model == "scvi_czi" and not (args.scvi_model_dir and args.ensembl_gene_file):
        ap.error("scvi_czi requires --scvi-model-dir and --ensembl_gene_file")

    args.use_gpu = args.device == "cuda"
    out_dir = Path(args.out_dir)
    check = C.inspect_counts_file(args.h5ad)
    C.write_json(out_dir / "dataset_status.json", check)
    if not check["eligible"]:
        C.remove_stale_results(out_dir)
        warnings.warn(f"Skipping dataset: {check['status']} ({args.h5ad})")
        return
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)

    C.set_seed(args.seed)
    gpu_before = C.torch_report()

    adata = C.load_anndata(args.h5ad)
    n_obs_raw, n_vars_raw = int(adata.n_obs), int(adata.n_vars)
    adata, info = C.ensure_counts_layer(adata, args.counts_policy)
    adata, cell_idx = C.subset_cells(adata, args.max_cells, args.seed)

    if args.model == "scimilarity":
        if not args.model_dir:
            raise SystemExit("--model scimilarity requires --model-dir")
        atts, var_names, model_info, prep_s, calc_s, npz = _run_scimilarity(args, adata)
    elif args.model == "scvi_local":
        atts, var_names, model_info, prep_s, calc_s, npz = _run_scvi_local(args, adata)
    else:
        atts, var_names, model_info, prep_s, calc_s, npz = _run_scvi(args, adata)

    gpu_after = _gpu_mem()

    row_sums = np.asarray(atts.sum(axis=1)).ravel()
    manifest = {
        "run": {
            "model": args.model,
            "method": args.method,
            "batch_size": args.batch_size,
            "max_cells": args.max_cells,
            "seed": args.seed,
            "device": args.device,
            "counts_policy": args.counts_policy,
            "ig_steps": 50 if args.method == "ig" else None,
        },
        "input": {
            "h5ad": str(Path(args.h5ad).resolve()),
            "n_obs_raw": n_obs_raw,
            "n_vars_raw": n_vars_raw,
            "n_cells_used": int(adata.n_obs),
            "cell_indices_used": cell_idx,
            "cell_ids_used": adata.obs_names.astype(str).tolist(),
            "counts_layer": info,
        },
        "model": model_info,
        "attribution": {
            "matrix_shape": list(atts.shape),
            "nnz": int(atts.nnz),
            "density": round(float(atts.nnz) / float(atts.shape[0] * atts.shape[1]), 8),
            "row_sum_min": round(float(row_sums.min()), 4),
            "row_sum_max": round(float(row_sums.max()), 4),
            "target_sum": 1000.0,
            "npz_path": npz,
        },
        "timing_seconds": {
            "preprocess_or_train": round(prep_s, 3),
            "attribution": round(calc_s, 3),
            "per_1k_cells": round(calc_s / max(atts.shape[0], 1) * 1000, 3),
        },
        "gpu": {"before": gpu_before, "after": gpu_after},
    }

    stem = f"{args.model}_{args.method}"
    C.write_json(out_dir / "manifests" / f"{stem}_manifest.json", manifest)

    # Gene order is required to interpret the npz later.
    C.write_json(
        out_dir / "attributions" / f"{stem}_var_names.json",
        [str(v) for v in var_names],
    )

    tops = C.top_genes_per_cell(atts, var_names, top_k=args.top_k)
    C.write_json(out_dir / "attributions" / f"{stem}_top{args.top_k}_per_cell.json", tops)

    agg = C.aggregate_gene_scores(atts, var_names)[: min(200, len(var_names))]
    C.write_json(
        out_dir / "attributions" / f"{stem}_gene_mean_scores.json",
        [{"gene": g, "mean_score": round(s, 6)} for g, s in agg],
    )

    print(f"[ok] {args.model}/{args.method} cells={atts.shape[0]} genes={atts.shape[1]}")
    print(f"[ok] npz={npz}")
    print(f"[ok] manifest={out_dir / 'manifests' / (stem + '_manifest.json')}")
    print(f"[ok] gpu={gpu_after}")


if __name__ == "__main__":
    main()

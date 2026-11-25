# -*- coding: utf-8 -*-
"""Shared helpers for SIGnature attribution runs on Windows.

Design notes
------------
* Nothing in this module edits the upstream SIGnature checkout. It only imports
  ``SIGnature.models.scimilarity`` / ``SIGnature.models.scvi`` as installed.
* Full-matrix checks accept only finite nonnegative integer-like counts.
  When counts are stored in X, they are copied without rounding.
* Device, batch size and cell subsampling are arguments so GPU use stays inside
  the 16 GB budget of this workstation.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch not importable yet during early checks
        pass


def torch_report() -> dict:
    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
    return info


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

COUNTS_POLICIES = ("auto", "as_is", "rint", "fail")


def inspect_counts_file(path, block_size=1000000):
    """Check every stored value; integer-like values are counts-compatible, not proof of provenance."""
    import h5py
    path = Path(path)
    result = {"input": str(path), "eligible": False, "status": "input_unavailable"}
    if not path.exists():
        return result
    with h5py.File(path, "r") as f:
        source = "layers/counts" if "layers/counts" in f else "X"
        result["counts_source"] = source
        if source not in f:
            result["status"] = "no_counts_matrix"
            return result
        node = f[source]
        values = node["data"] if isinstance(node, h5py.Group) else node
        n_positive = 0
        for i in range(0, values.shape[0], max(1, block_size // (values.shape[1] if values.ndim == 2 else 1))):
            step = max(1, block_size // (values.shape[1] if values.ndim == 2 else 1))
            v = np.asarray(values[i:i+step])
            if not np.isfinite(v).all():
                result["status"] = "nonfinite_values"; return result
            if (v < 0).any():
                result["status"] = "negative_values"; return result
            if not np.all(np.abs(v - np.rint(v)) <= 1e-6):
                result["status"] = "noninteger_expression"; return result
            n_positive += int(np.count_nonzero(v > 0))
        if not n_positive:
            result["status"] = "all_zero_matrix"; return result
        if source == "X" and "uns/log1p" in f:
            result["status"] = "X_marked_log1p"; return result
        result.update(eligible=True, status="counts_compatible", evidence="full_matrix_finite_nonnegative_integer_check", provenance="numerical_compatibility_only")
    return result


def stale_result_files(folder):
    folder = Path(folder).resolve()
    allowed = Path(__file__).resolve().parents[1] / "results"
    if not folder.is_relative_to(allowed.resolve()):
        raise ValueError("Cleanup is limited to signature_attr_run/results")
    files = []
    for name in ("attributions", "scvi_local", "manifests", "profiles"):
        sub = folder / name
        if sub.exists():
            files.extend(p for p in sub.rglob("*") if p.is_file())
    return sorted(files)


def remove_stale_results(folder):
    files = stale_result_files(folder)
    print(f"Cleanup exact file count: {len(files)}; all inside active workspace")
    for path in files:
        print(f"Cleanup target: {path}")
    for path in files:
        path.unlink()


def load_anndata(path: str):
    import anndata as ad

    return ad.read_h5ad(path)


def ensure_counts_layer(adata, policy: str = "auto"):
    """Return (adata, info) with ``layers['counts']`` populated.

    policy
    ------
    auto : copy validated integer-compatible X unchanged when counts is absent
    as_is, rint : compatibility aliases; neither allows fractional inputs or rounding
    fail  : raise if layers['counts'] is absent
    """
    import numpy as np
    from scipy.sparse import issparse

    info = {"counts_policy": policy, "counts_layer_present": "counts" in adata.layers}

    if "counts" in adata.layers:
        info["counts_source"] = "existing layers['counts']"
        return adata, info

    if policy == "fail":
        raise ValueError(
            "adata.layers['counts'] missing and --counts-policy=fail was requested"
        )

    X = adata.X
    values = X.data if issparse(X) else np.asarray(X)
    if not np.isfinite(values).all() or (values < 0).any() or not np.all(np.abs(values - np.rint(values)) <= 1e-6):
        raise ValueError("X is not counts-compatible; rounding cannot restore counts")
    adata.layers["counts"] = X.copy()
    info["counts_source"] = "X.copy() (validated integer-compatible)"

    info["counts_min"] = float(adata.layers["counts"].min())
    info["counts_max"] = float(adata.layers["counts"].max())
    return adata, info


def matrix_stats(X, sample_cells: int = 50, seed: int = 0) -> dict:
    """Compact statistics used to justify the counts policy."""
    import numpy as np
    from scipy.sparse import issparse

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_cells, n), replace=False)
    sub = X[idx, :]
    sub = sub.toarray() if issparse(sub) else np.asarray(sub)
    nz = sub[sub > 0]
    return {
        "shape": list(X.shape),
        "sparse": bool(issparse(X)),
        "sample_cells": int(len(idx)),
        "nonzero_fraction": round(float(nz.size / sub.size), 6),
        "min_nonzero": float(nz.min()) if nz.size else None,
        "max": float(sub.max()),
        "integer_valued_fraction": (
            round(float(np.mean(np.isclose(nz, np.round(nz), atol=1e-6))), 6)
            if nz.size
            else None
        ),
    }


def subset_cells(adata, max_cells: int | None, seed: int = 0):
    if max_cells is None or max_cells >= adata.n_obs:
        return adata, list(range(adata.n_obs))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[idx].copy(), idx.tolist()


def top_variable_genes(adata, n_top: int, layer: str | None = "counts"):
    """Boolean mask of the top-``n_top`` genes by variance.

    ``scanpy.pp.highly_variable_genes`` bins log-means and raises
    ``ValueError: cannot specify integer bins when input data contains
    infinity`` on this matrix, so variance ranking is used instead: it is
    deterministic and does not depend on scanpy's binning internals.
    """
    from scipy.sparse import issparse

    X = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if issparse(X):
        mu = np.asarray(X.mean(axis=0)).ravel()
        mu2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mu2 - mu**2
    else:
        var = np.var(np.asarray(X), axis=0)
    n_top = min(n_top, var.size)
    order = np.argsort(-var)[:n_top]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[order] = True
    return mask


# --------------------------------------------------------------------------
# scVI: local-model wrapper (no categorical covariates)
# --------------------------------------------------------------------------


def build_local_scvi_wrapper(base_wrapper_cls):
    """Subclass SIGnature's SCVIWrapper for models trained without cat covariates.

    The upstream forward always builds a categorical tensor (``cat_value=6006``)
    for the CZI model. A locally trained scVI has no such covariate, so both the
    weight computation and the attribution forward must omit it.
    """
    import torch
    from torch import nn

    class LocalSCVIWrapper(base_wrapper_cls):
        def _z_mean(self, x):
            # Follow SIGnature's direct encoder attribution path.
            out = self.model(x)
            dist = out[0] if isinstance(out, (tuple, list)) else out
            return dist.loc if hasattr(dist, "loc") else dist

        def forward(self, inputs, weights):
            summed = torch.sum(torch.abs(weights * self._z_mean(inputs)), dim=1)
            return torch.cat([summed.unsqueeze(1), summed.unsqueeze(1)], dim=-1)

        def calculate_attributions(
            self,
            X,
            method="ig",
            batch_size=64,
            multiply_by_inputs=True,
            disable_tqdm=False,
            target_sum=1e3,
            npz_path=None,
            n_steps=50,
        ):
            from captum.attr import DeepLift, IntegratedGradients, Saliency
            from scipy.sparse import csr_matrix, save_npz, vstack
            from tqdm import tqdm

            from SIGnature import utils

            methods = {"ig": IntegratedGradients, "dl": DeepLift, "ixg": Saliency}
            key = method.lower()
            if key not in methods:
                raise ValueError(f"Unknown method {method}; use {list(methods)}")
            attr_cls = methods[key]

            self.model.eval()
            self.eval()
            device = torch.device("cuda" if self.use_gpu else "cpu")
            self.model.to(device)
            self.to(device)

            attributor = (
                attr_cls(self)
                if key == "ixg"
                else attr_cls(self, multiply_by_inputs=multiply_by_inputs)
            )

            attrs_list = []
            signed_list = []
            deltas = []
            for i in tqdm(range(0, X.shape[0], batch_size), disable=disable_tqdm):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sub = X[i : i + batch_size, :]
                dense = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
                xb = torch.tensor(dense, dtype=torch.float32, device=device)
                xb.requires_grad_(True)
                with torch.no_grad():
                    w = self._z_mean(xb).detach()
                if key == "ixg":
                    a = attributor.attribute(
                        inputs=xb, target=0, additional_forward_args=(w,)
                    )
                    if multiply_by_inputs:
                        a = a * xb
                elif key == "ig":
                    a, delta = attributor.attribute(
                        inputs=xb,
                        baselines=torch.zeros_like(xb),
                        target=0,
                        additional_forward_args=(w,),
                        n_steps=n_steps,
                        return_convergence_delta=True,
                    )
                    deltas.extend(delta.detach().cpu().numpy().tolist())
                    signed_list.append(csr_matrix(a.detach().cpu().numpy()))
                else:
                    a = attributor.attribute(
                        inputs=xb,
                        baselines=torch.zeros_like(xb),
                        target=0,
                        additional_forward_args=(w,),
                    )
                attrs_list.append(
                    csr_matrix(torch.abs(a).detach().cpu().numpy().astype(np.float32))
                )

            if key == "ig" and npz_path:
                prefix = Path(npz_path).with_suffix("")
                signed_matrix = vstack(signed_list)
                save_npz(str(prefix) + "_signed.npz", signed_matrix)
                difference = np.asarray(signed_matrix.sum(axis=1)).ravel() - np.asarray(deltas)
                relative_delta = np.abs(deltas) / np.maximum(np.abs(difference), 1e-8)
                write_json(str(prefix) + "_integration.json", {
                    "n_steps": n_steps, "quadrature": "gausslegendre",
                    "baseline": "zero_counts", "input_space": "counts",
                    "encoder_transform": "identity_official_signature",
                    "convergence_delta": deltas,
                    "absolute_delta_median": float(np.median(np.abs(deltas))),
                    "absolute_delta_max": float(np.max(np.abs(deltas))),
                    "relative_delta_median": float(np.median(relative_delta)),
                    "relative_delta_max": float(np.max(relative_delta)),
                })
            return utils.normalize_attribution_matrix(
                matrix=vstack(attrs_list), target_sum=target_sum, npz_path=npz_path
            )

    return LocalSCVIWrapper


# --------------------------------------------------------------------------
# Attribution post-processing
# --------------------------------------------------------------------------


def top_genes_per_cell(attributions, var_names, top_k: int = 20) -> list[dict]:
    """Top-k genes per cell from a CSR attribution matrix."""
    var_names = list(map(str, var_names))
    rows = []
    for i in range(attributions.shape[0]):
        start, stop = attributions.indptr[i], attributions.indptr[i + 1]
        idx = attributions.indices[start:stop]
        val = attributions.data[start:stop]
        if idx.size == 0:
            rows.append({"cell_index": i, "top_genes": [], "scores": []})
            continue
        order = np.argsort(-val)[:top_k]
        rows.append(
            {
                "cell_index": i,
                "top_genes": [var_names[j] for j in idx[order]],
                "scores": [round(float(v), 6) for v in val[order]],
            }
        )
    return rows


def aggregate_gene_scores(attributions, var_names) -> "list[tuple[str, float]]":
    """Mean attribution per gene across cells."""
    import numpy as np

    var_names = list(map(str, var_names))
    col_sums = np.asarray(attributions.sum(axis=0)).ravel()
    n = max(attributions.shape[0], 1)
    order = np.argsort(-col_sums)
    return [(var_names[j], float(col_sums[j] / n)) for j in order]


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

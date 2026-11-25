#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for graph builders.
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


@dataclass(frozen=True)
class GraphOutput:
    node_ids: List[str]
    x: torch.FloatTensor  # (N, d)
    edge_index: torch.LongTensor  # (2, E)
    edge_weight: torch.FloatTensor  # (E,)
    num_nodes: int


def _decode_names(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for v in arr.tolist():
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def ensure_unique_node_ids(node_ids: List[str], context: str = "", extra_hint: str = "") -> None:
    seen = set()
    dup = []
    for x in node_ids:
        if x in seen:
            dup.append(x)
        else:
            seen.add(x)
    if dup:
        sample = ", ".join(list(dict.fromkeys(dup))[:10])
        ctx = f" {context}" if context else ""
        hint = f" {extra_hint.strip()}" if extra_hint else ""
        raise ValueError(
            f"Duplicate node_ids found{ctx} (count={len(dup)}). "
            f"Example duplicates: {sample}." + hint
        )


def summarize_graph(num_nodes: int, edge_index: np.ndarray) -> Dict[str, float]:
    src = edge_index[0]
    deg = np.bincount(src, minlength=num_nodes).astype(np.int64)
    return {
        "num_nodes": float(num_nodes),
        "num_edges_directed": float(edge_index.shape[1]),
        "avg_out_degree": float(deg.mean()) if deg.size else 0.0,
        "max_out_degree": float(deg.max()) if deg.size else 0.0,
        "min_out_degree": float(deg.min()) if deg.size else 0.0,
    }


def _extract_uniprot_id(full_id: str) -> str:
    """
    Extract core UniProt ID from full ID string.

    Examples:
        sp|A0A023I7E1|ENG1_RHIMI -> A0A023I7E1
        A0A023I7E1 -> A0A023I7E1
    """
    if '|' in full_id:
        parts = full_id.split('|')
        if len(parts) >= 2:
            return parts[1]
    return full_id.strip()


def load_protein_embeddings_from_dir(
    embedding_dir: Path,
    id_column: str = "ID",
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load protein embeddings from directory containing .npy and .xlsx files.

    Expected directory structure:
        embedding_dir/
            *.npy: embedding matrix (N_proteins, emb_dim)
            *.xlsx: protein ID file with id_column

    Args:
        embedding_dir: Directory containing embedding files
        id_column: Column name in xlsx file containing protein IDs

    Returns:
        emb_dict: {uniprot_id: np.ndarray(d,)}
        dim: embedding dimension
    """
    embedding_dir = Path(embedding_dir)

    # Find .npy and .xlsx files
    npy_files = list(embedding_dir.glob("*.npy"))
    xlsx_files = list(embedding_dir.glob("*.xlsx"))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {embedding_dir}")
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {embedding_dir}")

    if len(npy_files) > 1:
        raise ValueError(f"Multiple .npy files found in {embedding_dir}: {[f.name for f in npy_files]}")

    # Find xlsx file with id_column
    valid_xlsx = None
    for xlsx_file in xlsx_files:
        try:
            df = pd.read_excel(xlsx_file, nrows=0)
            if id_column in df.columns:
                valid_xlsx = xlsx_file
                break
        except Exception:
            continue

    if valid_xlsx is None:
        raise ValueError(
            f"No xlsx file contains column '{id_column}' in {embedding_dir}. "
            f"Found xlsx files: {[f.name for f in xlsx_files]}"
        )

    # Load embeddings
    npy_file = npy_files[0]
    embeddings = np.load(npy_file).astype(np.float32)

    # Load IDs
    id_df = pd.read_excel(valid_xlsx)
    protein_ids = [_extract_uniprot_id(str(id_val)) for id_val in id_df[id_column]]

    if len(protein_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Protein ID count ({len(protein_ids)}) != embedding rows ({embeddings.shape[0]})"
        )

    dim = int(embeddings.shape[1])
    emb_dict = {protein_ids[i]: embeddings[i] for i in range(len(protein_ids))}

    return emb_dict, dim


def load_protein_embeddings_npz(emb_path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Supported embedding npz formats (auto-detected):

    A) key=UniProt_ID -> value=emb vector
    B) ids (M,), emb/embeddings (M,d)
    C) gene_names (M,), embeddings (M,d)

    Returns:
      emb_dict: {uniprot_id: np.ndarray(d,)}
      dim: embedding dim
    """
    npz = np.load(emb_path, allow_pickle=True)
    files = list(npz.files)

    # Format B/C: matrix with ID column
    id_key = None
    if "ids" in files:
        id_key = "ids"
    elif "gene_names" in files:
        id_key = "gene_names"

    if id_key is not None and ("emb" in files or "embeddings" in files):
        ids = _decode_names(npz[id_key])
        mat_key = "emb" if "emb" in files else "embeddings"
        mat = np.asarray(npz[mat_key])
        if mat.ndim != 2 or mat.shape[0] != len(ids):
            raise ValueError(f"Embedding matrix shape mismatch: {id_key}={len(ids)}, {mat_key}={mat.shape}")
        mat = mat.astype(np.float32, copy=False)
        dim = int(mat.shape[1])
        emb_dict = {ids[i]: mat[i] for i in range(len(ids))}
        return emb_dict, dim

    # Format A: each key is a UniProt_ID
    reserved = {"names", "scores", "ids", "gene_names", "emb", "embeddings", "arr_0"}
    emb_dict: Dict[str, np.ndarray] = {}
    dim: Optional[int] = None

    for k in files:
        if k in reserved:
            continue
        v = np.asarray(npz[k])
        if v.ndim == 2 and v.shape[0] == 1:
            v = v.reshape(-1)
        if v.ndim != 1:
            raise ValueError(f"Embedding for key={k} must be 1D (d,), got shape {v.shape}")
        v = v.astype(np.float32, copy=False)

        if dim is None:
            dim = int(v.shape[0])
        elif int(v.shape[0]) != dim:
            raise ValueError(f"Inconsistent embedding dim: key={k} dim={v.shape[0]} expected={dim}")

        emb_dict[str(k).strip()] = v

    if dim is None or not emb_dict:
        raise ValueError(
            f"No valid embeddings found in {emb_path}. "
            f"Expected either (ids/gene_names + emb/embeddings) or per-key UniProt vectors."
        )
    return emb_dict, int(dim)


def build_node_feature_matrix(
    node_ids: List[str],
    emb_dict: Dict[str, np.ndarray],
    emb_dim: int,
    missing_policy: str,
) -> Tuple[torch.FloatTensor, List[str]]:
    """
    Align node_ids -> x(N,d)
    missing_policy:
      - error: any missing raises error
      - zero:  fill zeros and return missing list
    """
    n = len(node_ids)
    x = np.zeros((n, emb_dim), dtype=np.float32)
    missing: List[str] = []

    # Try exact match and uppercase to reduce case issues.
    for i, uid in enumerate(node_ids):
        key = str(uid).strip()
        if key in emb_dict:
            x[i] = emb_dict[key]
            continue
        key_up = key.upper()
        if key_up in emb_dict:
            x[i] = emb_dict[key_up]
            continue
        missing.append(key)

    if missing and missing_policy.lower() == "error":
        sample = ", ".join(missing[:20])
        raise KeyError(f"Missing embeddings for {len(missing)} node_ids. Example: {sample}")

    return torch.from_numpy(x).float(), missing


def save_graph_outputs(out_dir: Path, prefix: str, graph: GraphOutput, metadata: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / f"{prefix}.pt"
    torch.save(
        {
            "node_ids": graph.node_ids,
            "x": graph.x,
            "edge_index": graph.edge_index,
            "edge_weight": graph.edge_weight,
            "num_nodes": graph.num_nodes,
            "metadata": metadata,
        },
        pt_path,
    )

    npz_path = out_dir / f"{prefix}.npz"
    np.savez_compressed(
        npz_path,
        node_ids=np.array(graph.node_ids, dtype=object),
        x=graph.x.cpu().numpy().astype(np.float32),
        edge_index=graph.edge_index.cpu().numpy().astype(np.int64),
        edge_weight=graph.edge_weight.cpu().numpy().astype(np.float32),
        num_nodes=np.int64(graph.num_nodes),
    )

    meta_path = out_dir / f"{prefix}.meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

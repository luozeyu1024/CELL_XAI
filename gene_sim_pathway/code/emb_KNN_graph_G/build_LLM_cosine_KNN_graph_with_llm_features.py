#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_protein_knn_graph_with_llm_features.py

核心定位（重要）
----------------
本脚本构建的是 **protein(isoform) 节点** 的 cosine-kNN 图，而不是 gene 节点图。

- 输入：protein ID + protein embedding（例如 UniProt isoform 级别 embedding）
- 输出：protein-protein kNN 图（edge_index / edge_weight）+ protein 节点特征 x

为何这样设计？
-------------
你在 HAN 里的 TM/PPI 通道通常是“protein层关系 -> 通过 isoform->gene 映射 -> gene层关系”的桥接方案。
因此，为了与 TM/PPI 完全一致，本脚本应当先产生 **protein层 kNN prior**。

后续如何与 HAN gene-level meta-path 兼容？
----------------------------------------
本脚本只负责 protein-protein kNN 图的构建。后续兼容 HAN 的两种常见做法：

A) 离线折叠成 gene-gene 边（推荐与 ppi_bridge / tmscore_bridge 统一）
   给定 isoform->gene 映射 M(p)=g：
   (p_i, p_j, w_ij) -> (g_i, g_j, w_ij)，多条映射到同一 (g_i,g_j) 时用 max/mean 聚合。

B) 在异构图里显式建元路径（gene->protein, protein->protein(knn), protein->gene）
   gene --(encodes)--> protein --(knn)--> protein --(encoded_by)--> gene

输入目录格式（仅支持目录输入；已移除 NPZ 输入）
--------------------------------------------
--open_path 指向目录，该目录需要包含：
  - embeddings.npy            : shape (N, D) 的 float/float32 数组
  - ids.xlsx / ids.csv / ids.tsv / node_ids.xlsx / node_ids.csv / node_ids.tsv
        至少包含一列 protein ID，列名由 --protein_id_column 指定（默认 "ID"）

注意：--strip_isoform
--------------------
若设置 --strip_isoform，会将 P12345-2 -> P12345。此时不同 isoform 可能折叠为相同 ID，
会触发重复 node_id（ensure_unique_node_ids 会报错）。如需 canonical-level，请在外部先做聚合。

输出文件（由 util.save_graph_outputs 写出）
-----------------------------------------
- {save_path}/{prefix}.pt
- {save_path}/{prefix}.npz
- {save_path}/{prefix}.meta.json

Usage:
  python build_protein_knn_graph_with_llm_features.py ^
    --open_path "assets/proteins" ^
    --save_path "work/graphs" ^
    --n_neighbors 30 ^
    --prefix "protein_knn_graph" ^
    --make_undirected
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from util import (
    _extract_uniprot_id,
    setup_logger,
    GraphOutput,
    ensure_unique_node_ids,
    summarize_graph,
    save_graph_outputs,
)

logger = logging.getLogger("PROTEIN_KNN_GRAPH")


# --------------------------
# Embedding loader (directory only)
# --------------------------
def _find_col_case_insensitive(columns: Iterable[str], target: str) -> Optional[str]:
    tgt = str(target).lower()
    for c in columns:
        if str(c).lower() == tgt:
            return str(c)
    return None


def _auto_find_id_table(emb_dir: Path) -> Path:
    """
    自动查找目录中的 ID 表文件（任意文件名，支持 .xlsx/.csv/.tsv 后缀）。
    如果只有一个匹配文件则直接使用；如果有多个则使用第一个并发出警告。
    """
    emb_dir = Path(emb_dir)
    valid_suffixes = (".xlsx", ".csv", ".tsv")
    id_files = [f for f in emb_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_suffixes]
    
    if not id_files:
        raise FileNotFoundError(
            f"Missing id table in {emb_dir}. Expect one file with suffix: {valid_suffixes}"
        )
    if len(id_files) > 1:
        logger.warning(f"Multiple id tables found, using first: {id_files[0].name}")
    return id_files[0]


def _auto_find_embedding_npy(emb_dir: Path) -> Path:
    """
    自动查找目录中的 .npy embedding 文件。
    如果只有一个 .npy 文件则直接使用；如果有多个则使用第一个并发出警告。
    """
    emb_dir = Path(emb_dir)
    npy_files = list(emb_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"Missing embedding .npy file in: {emb_dir}")
    if len(npy_files) > 1:
        logger.warning(f"Multiple .npy files found, using first: {npy_files[0].name}")
    return npy_files[0]


def _normalize_uniprot_id(pid: str, strip_isoform: bool) -> str:
    """
    Normalize UniProt ID: extract core ID and optionally strip isoform suffix.

    Examples:
        sp|A0A023I7E1|ENG1_RHIMI -> A0A023I7E1
        P12345-2 -> P12345 (if strip_isoform=True)
        P12345-2 -> P12345-2 (if strip_isoform=False)
    """
    s = _extract_uniprot_id(str(pid).strip())
    if strip_isoform and "-" in s:
        s = s.split("-", 1)[0]
    return s


def load_protein_embeddings_from_dir(
    emb_dir: Path,
    protein_id_column: str = "ID",
    strip_isoform: bool = False,
) -> Tuple[List[str], np.ndarray, str]:
    """
    从目录读取：
      - embeddings.npy: (N,D)
      - ids.{xlsx|csv|tsv}/node_ids.{xlsx|csv|tsv}: 至少包含一列 protein ID（--protein_id_column）

    返回：
      node_ids(list[str])，embeddings(N,D float32)，source_desc(str)
    """
    emb_dir = Path(emb_dir)
    if not emb_dir.exists():
        raise FileNotFoundError(emb_dir)

    emb_path = _auto_find_embedding_npy(emb_dir)

    emb = np.load(str(emb_path))
    if emb.ndim != 2:
        raise ValueError(f"embeddings.npy must be 2D, got shape={emb.shape}")
    emb = emb.astype(np.float32, copy=False)

    id_table = _auto_find_id_table(emb_dir)

    import pandas as pd

    if id_table.suffix.lower() == ".xlsx":
        df = pd.read_excel(id_table)
    else:
        df = pd.read_csv(id_table, sep=None, engine="python")

    col_real = _find_col_case_insensitive(df.columns, protein_id_column)
    if col_real is None:
        raise KeyError(
            f"Protein ID column not found: --protein_id_column={protein_id_column}. "
            f"Available columns={list(df.columns)}"
        )

    raw_ids = df[col_real].astype(str).tolist()
    node_ids = [_normalize_uniprot_id(pid, strip_isoform=strip_isoform) for pid in raw_ids]

    if len(node_ids) != emb.shape[0]:
        raise ValueError(f"IDs count != embeddings rows: {len(node_ids)} vs {emb.shape[0]}")

    src_desc = (
        f"dir={emb_dir}; emb={emb_path.name}; ids={id_table.name}; "
        f"id_col={col_real}; strip_isoform={strip_isoform}"
    )
    return node_ids, emb, src_desc


# --------------------------
# sanitize + normalize
# --------------------------
def sanitize_embeddings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(x)
    if np.any(bad):
        logger.warning("Found NaN/Inf in embeddings. Replacing with 0.")
        x[bad] = 0.0
    return x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return x / norms


# --------------------------
# kNN graph (cosine)
# --------------------------
def dedup_edges_keep_max_weight(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 (src,dst) 去重；若重复边存在，保留 weight 最大者。
    采用 key=src*num_nodes+dst 的向量化去重（避免 Python 循环）。
    """
    n = np.int64(num_nodes)
    src = src.astype(np.int64, copy=False)
    dst = dst.astype(np.int64, copy=False)
    w = w.astype(np.float32, copy=False)

    if src.size == 0:
        return src, dst, w

    key = src * n + dst
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    w_s = w[order]

    boundary = np.empty((key_s.size,), dtype=bool)
    boundary[0] = True
    boundary[1:] = key_s[1:] != key_s[:-1]
    idx = np.flatnonzero(boundary)

    w_max = np.maximum.reduceat(w_s, idx)
    key_u = key_s[idx]

    src_u = (key_u // n).astype(np.int64, copy=False)
    dst_u = (key_u % n).astype(np.int64, copy=False)
    return src_u, dst_u, w_max.astype(np.float32, copy=False)


def build_cosine_knn_graph(
    embeddings_l2: np.ndarray,
    n_neighbors: int,
    make_undirected: bool,
    remove_self_loop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 L2 embeddings 构建 cosine-kNN 图（protein-protein）：
      - sklearn NearestNeighbors(metric='cosine')：distance = 1 - cosine_sim
      - edge_weight 保存 cosine_sim

    输出:
      edge_index (2,E), edge_weight (E,)
    """
    n = int(embeddings_l2.shape[0])
    if n <= 1:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    k = int(n_neighbors)
    if k <= 0:
        raise ValueError("--n_neighbors must be > 0")
    k_eff = int(min(k, n - 1))

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("scikit-learn is required. Please install scikit-learn.") from e

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="auto")
    nn.fit(embeddings_l2)

    dist, idx = nn.kneighbors(embeddings_l2, return_distance=True)  # (n, k+1)
    sim = (1.0 - dist).astype(np.float32)  # cosine similarity

    src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64, copy=False)
    w = sim.reshape(-1).astype(np.float32, copy=False)

    if remove_self_loop:
        m = src != dst
        src, dst, w = src[m], dst[m], w[m]

    m = np.isfinite(w)
    src, dst, w = src[m], dst[m], w[m]
    w = np.clip(w, -1.0, 1.0)

    if make_undirected:
        src0, dst0, w0 = src, dst, w
        src = np.concatenate([src0, dst0], axis=0)
        dst = np.concatenate([dst0, src0], axis=0)
        w = np.concatenate([w0, w0], axis=0)

    src_u, dst_u, w_u = dedup_edges_keep_max_weight(src, dst, w, num_nodes=n)
    edge_index = np.stack([src_u, dst_u], axis=0)
    return edge_index, w_u


# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build cosine-kNN protein graph from protein LLM embeddings (directory input only)."
    )

    # 按你的约定：input 路径用 --open_path，output 路径用 --save_path
    p.add_argument("--open_path", type=str, required=True, help="Protein embedding directory (embeddings.npy + ids table).")
    p.add_argument("--save_path", type=str, required=True, help="Output directory")

    p.add_argument("--prefix", type=str, default="protein_knn_graph", help="Output file prefix")
    p.add_argument("--protein_id_column", type=str, default="ID", help="Column name for protein IDs in id table (default ID).")

    p.add_argument("--n_neighbors", type=int, default=30, help="k in kNN (default 30)")
    p.add_argument(
        "--make_undirected",
        action="store_true",
        help="Output undirected graph as two directed edges (recommended for HAN).",
    )
    p.add_argument(
        "--keep_self_loop",
        action="store_true",
        help="Keep self-loop edges (default: remove).",
    )

    p.add_argument("--max_nodes", type=int, default=0, help="Debug only: use first max_nodes proteins (0 means all).")
    p.add_argument(
        "--strip_isoform",
        action="store_true",
        help="If set, convert P12345-2 -> P12345 (may introduce duplicated IDs without aggregation).",
    )
    p.set_defaults(strip_isoform=False)
    return p.parse_args()


def main() -> None:
    setup_logger("PROTEIN_KNN_GRAPH")
    args = parse_args()

    open_dir = Path(args.open_path)
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    node_ids, x, source_desc = load_protein_embeddings_from_dir(
        emb_dir=open_dir,
        protein_id_column=str(args.protein_id_column),
        strip_isoform=bool(args.strip_isoform),
    )

    if args.max_nodes and int(args.max_nodes) > 0:
        m = int(args.max_nodes)
        node_ids = node_ids[:m]
        x = x[:m, :]
        logger.warning("max_nodes is set: using first %d proteins for debug.", m)

    node_ids = [str(i) for i in node_ids]
    ensure_unique_node_ids(node_ids)

    x = sanitize_embeddings(x)
    x_l2 = l2_normalize_rows(x)

    logger.info(
        "Building cosine-kNN protein graph: N=%d, D=%d, k=%d",
        x_l2.shape[0], x_l2.shape[1], int(args.n_neighbors),
    )

    edge_index_np, edge_weight_np = build_cosine_knn_graph(
        embeddings_l2=x_l2,
        n_neighbors=int(args.n_neighbors),
        make_undirected=bool(args.make_undirected),
        remove_self_loop=not bool(args.keep_self_loop),
    )

    summary = summarize_graph(num_nodes=int(x_l2.shape[0]), edge_index=edge_index_np)
    logger.info("Graph summary: %s", summary)

    graph = GraphOutput(
        node_ids=node_ids,
        x=torch.from_numpy(x_l2).float(),
        edge_index=torch.from_numpy(edge_index_np).long(),
        edge_weight=torch.from_numpy(edge_weight_np).float(),
        num_nodes=int(x_l2.shape[0]),
    )

    metadata = {
        "node_type": "protein_isoform",
        "source_embedding": source_desc,
        "n_nodes": int(graph.num_nodes),
        "emb_dim": int(graph.x.size(1)),
        "knn_k": int(args.n_neighbors),
        "metric": "cosine",
        "l2_normalize": True,
        "make_undirected": bool(args.make_undirected),
        "keep_self_loop": bool(args.keep_self_loop),
        "strip_isoform": bool(args.strip_isoform),
        "graph_summary": summary,
        "output_prefix": str(args.prefix),
        "han_compat_note": (
            "This is a protein-protein kNN prior built from protein(isoform) embeddings. "
            "To use it as a gene-level relation in HAN (like ppi_bridge/tmscore_bridge), "
            "map isoform->gene and aggregate protein edges into gene edges (max/mean), "
            "or build a meta-path gene->protein->protein->gene."
        ),
        "input_dir_required_files": [
            "embeddings.npy (N,D)",
            "ids.xlsx/csv/tsv or node_ids.xlsx/csv/tsv with protein ID column",
        ],
    }

    # util.save_graph_outputs 写出：
    #   {prefix}.pt, {prefix}.npz, {prefix}.meta.json
    save_graph_outputs(save_dir, str(args.prefix), graph, metadata)


if __name__ == "__main__":
    main()

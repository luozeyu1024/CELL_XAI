"""Build exact cosine KNN graphs from the three aligned protein embedding bundles."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from build_LLM_cosine_KNN_graph_with_llm_features import dedup_edges_keep_max_weight
from util import GraphOutput, save_graph_outputs, summarize_graph

MODELS = {'esm2_650m': 'ESM2_650M', 'qwen': 'Qwen', 'state_15b': 'STATE_ESM2_15B'}


def build(args, model):
    path = Path(args.open_path) / MODELS[model] / 'human_swissprot_aligned.npz'
    with np.load(path, allow_pickle=True) as data:
        ids = data['ids'].astype(str)
        x = np.asarray(data['embeddings'], dtype=np.float32)
    if x.ndim != 2 or len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError(f'Invalid embedding shape or duplicate node IDs: {path}')
    norms = np.linalg.norm(x, axis=1)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(norms) & (norms > 0)
    excluded = ids[~valid].tolist()
    ids, x, norms = ids[valid], x[valid], norms[valid]
    order = np.argsort(ids, kind='stable')
    ids, x = ids[order], x[order] / norms[order, None]
    n = len(ids)
    if n < 2:
        raise ValueError('At least two valid nodes are required')
    k = min(args.n_neighbors, n - 1)
    tensor = torch.from_numpy(np.ascontiguousarray(x)).to(args.device)
    sources, destinations, weights = [], [], []
    print(f'{model}: {n} nodes, {x.shape[1]} dimensions, k={k}', flush=True)
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        similarities = (tensor[start:stop] @ tensor.T).clamp(-1, 1)
        similarities[torch.arange(stop-start, device=args.device),
                     torch.arange(start, stop, device=args.device)] = -torch.inf
        # IDs are lexically sorted, so stable sorting resolves exact ties by ID.
        neighbors = torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :k]
        values = similarities.gather(1, neighbors)
        sources.append(np.repeat(np.arange(start, stop, dtype=np.int64), k))
        destinations.append(neighbors.cpu().numpy().reshape(-1))
        weights.append(values.cpu().numpy().reshape(-1))
    src, dst, weight = map(np.concatenate, (sources, destinations, weights))
    directed = np.stack([src, dst])
    # Union symmetrization; each undirected connection has two directed records.
    src, dst, weight = dedup_edges_keep_max_weight(
        np.concatenate([src, dst]), np.concatenate([dst, src]),
        np.concatenate([weight, weight]), n)
    edges = np.stack([src, dst])
    assert np.isfinite(weight).all() and not np.any(src == dst)
    graph = GraphOutput(ids.tolist(), torch.from_numpy(x.copy()),
                        torch.from_numpy(edges), torch.from_numpy(weight), n)
    prefix = f'protein_knn_{model}_k{args.n_neighbors}'
    metadata = {
        'model': model, 'node_type': 'uniprot_protein', 'source_embedding': str(path.resolve()),
        'n_nodes': n, 'emb_dim': x.shape[1], 'knn_k': args.n_neighbors, 'effective_k': k,
        'metric': 'cosine', 'search': 'exact_blockwise_float32',
        'l2_normalize': True, 'make_undirected': True, 'keep_self_loop': False,
        'cosine_threshold': None, 'node_order': 'lexical_uniprot_id',
        'tie_break': 'lexical_uniprot_id_for_exact_float32_ties',
        'symmetrization': 'union_keep_max_weight',
        'batch_size': args.batch_size, 'device': args.device,
        'tf32': False, 'excluded_invalid_or_zero_vector_ids': excluded,
        'graph_summary': summarize_graph(n, edges),
        'embedding_note': ('Mapped STATE ESM2-15B Swiss-Prot subset; multiple proteins may share a gene vector.'
                           if model == 'state_15b' else
                           'Protein annotation text embeddings.' if model == 'qwen' else
                           'ESM2-650M mean-pooled protein sequence embeddings.'),
    }
    save_graph_outputs(Path(args.save_path), prefix, graph, metadata)
    # The directed KNN selection is separately retained: union degree can exceed K.
    np.savez_compressed(Path(args.save_path) / f'{prefix}.directed_knn.npz',
                        node_ids=ids, edge_index=directed,
                        edge_weight=np.concatenate(weights), num_nodes=np.int64(n))
    print(f'{model}: saved {edges.shape[1]} symmetric edge records; excluded {len(excluded)} nodes', flush=True)
    del tensor
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path', required=True, help='Directory containing the three model subdirectories')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--model', nargs='+', choices=[*MODELS, 'all'], default=['all'])
    parser.add_argument('--n_neighbors', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.n_neighbors < 1 or args.batch_size < 1:
        parser.error('n_neighbors and batch_size must be positive')
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        for model in MODELS if 'all' in args.model else dict.fromkeys(args.model):
            build(args, model)


if __name__ == '__main__':
    main()

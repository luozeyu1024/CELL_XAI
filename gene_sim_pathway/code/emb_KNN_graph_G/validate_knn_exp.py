"""Validate saved exact cosine KNN graphs and print compact data summaries."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path', required=True)
    args = parser.parse_args()
    root = Path(args.open_path)
    for model in ['esm2_650m', 'qwen', 'state_15b']:
        prefix = f'protein_knn_{model}_k30'
        with np.load(root / f'{prefix}.npz', allow_pickle=True) as z:
            ids, x, edges, w = (z[k] for k in ['node_ids', 'x', 'edge_index', 'edge_weight'])
        n = len(ids)
        assert n == len(set(ids)) == len(x)
        assert np.all(ids[:-1] < ids[1:])
        assert edges.shape == (2, len(w)) and edges.min() >= 0 and edges.max() < n
        assert np.isfinite(x).all() and np.isfinite(w).all() and (abs(w) <= 1).all()
        assert not np.any(edges[0] == edges[1])
        assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
        keys = edges[0] * n + edges[1]
        assert np.all(keys[:-1] < keys[1:])
        reverse = edges[1] * n + edges[0]
        pos = np.searchsorted(keys, reverse)
        assert (pos < len(keys)).all() and np.array_equal(keys[pos], reverse)
        assert np.array_equal(w[pos], w)
        sample = np.linspace(0, len(w)-1, 1000, dtype=int)
        recomputed = np.einsum('ij,ij->i', x[edges[0, sample]], x[edges[1, sample]])
        assert np.allclose(recomputed, w[sample], atol=2e-5)
        with np.load(root / f'{prefix}.directed_knn.npz') as d:
            di, dw = d['edge_index'], d['edge_weight']
            assert np.array_equal(d['node_ids'], ids)
        assert di.shape == (2, n*30) and np.all(np.bincount(di[0], minlength=n) == 30)
        assert not np.any(di[0] == di[1])
        dk = di[0]*n + di[1]
        assert len(np.unique(dk)) == len(dk)
        dp = np.searchsorted(keys, dk)
        assert np.array_equal(keys[dp], dk) and np.allclose(w[dp], dw, atol=2e-5)
        # Independently verify one complete similarity row and tie ordering.
        row = n//2
        sims = np.clip(x @ x[row], -1, 1)
        sims[row] = -np.inf
        actual = di[1, di[0] == row]
        assert sims[actual].min() >= np.sort(sims)[-30] - 2e-5
        sizes = {suffix: (root/f'{prefix}{suffix}').stat().st_size
                 for suffix in ['.npz', '.pt', '.meta.json', '.directed_knn.npz']}
        print(json.dumps({'model': model, 'n_nodes': n, 'emb_dim': x.shape[1],
                          'directed_knn_records': di.shape[1], 'symmetric_edge_records': len(w),
                          'undirected_connections': len(w)//2, 'size_bytes': sizes,
                          'validation': 'passed'}), flush=True)


if __name__ == '__main__':
    main()

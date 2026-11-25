"""Remove all inter-gene edges while retaining every node and node feature."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import torch


def remove_edges(graph):
    before=graph.edge_index.shape[1]
    graph.edge_index=torch.empty((2,0),dtype=torch.long)
    graph.edge_attr=torch.empty((0,1),dtype=torch.float32)
    return dict(cell_id=graph.cell_id,n_nodes=graph.num_nodes,
                n_undirected_edges_before=before//2,n_undirected_edges_after=0,
                node_inputs_preserved=True)

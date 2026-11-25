"""Frozen protein-derived gene features with the established GATv2/GINE backbone."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_model import GraphClassifier


class FrozenGeneTable(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        # One shared disk table, not a repeated copy in every classifier checkpoint.
        self.register_buffer('weight', torch.from_numpy(matrix), persistent=False)

    def forward(self, gene_id):
        return F.embedding(gene_id, self.weight)


class GeneEmbeddingClassifier(GraphClassifier):
    def __init__(self, n_genes, hidden=64, layers=2, heads=4, gene_dim=16,
                 dropout=.2, expr_mean=0., expr_std=1., embedding_path=None):
        if embedding_path is None:
            raise ValueError('A mapped gene embedding table is required')
        matrix = np.load(embedding_path, allow_pickle=False).astype(np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != n_genes or not np.isfinite(matrix).all():
            raise ValueError('Invalid gene embedding table')
        dimension = matrix.shape[1]
        super().__init__(n_genes, hidden, layers, heads, dimension, dropout, expr_mean, expr_std)
        self.identity = FrozenGeneTable(matrix)
        self.config['embedding_path'] = str(Path(embedding_path).resolve())


def restore(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = GeneEmbeddingClassifier(**saved['model_config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved

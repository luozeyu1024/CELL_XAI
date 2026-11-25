"""Load a cell signature and its induced graph without storing repeated edges."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

RELATIONS = ('esm2_650m', 'qwen', 'state_15b', 'ppi')


class CellGraphDataset:
    def __init__(self, root, dataset, relation='ppi'):
        if relation not in RELATIONS:
            raise ValueError(relation)
        self.root = Path(root)
        self.cells = pd.read_csv(self.root/dataset/'cells.tsv', sep='\t', keep_default_na=False)
        self.genes = pd.read_csv(self.root/'gene_vocab.tsv', sep='\t').gene_symbol.to_numpy()
        with np.load(self.root/dataset/'signatures.npz', allow_pickle=False) as z:
            self.ptr = z['ptr']; self.gene_id = z['gene_id']; self.features = z['features']
        self.adjacency = sparse.load_npz(self.root/'global_graphs'/f'{relation}.npz').tocsr()
        self.relation = relation

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        lo, hi = self.ptr[index:index+2]
        ids = self.gene_id[lo:hi]
        graph = self.adjacency[ids][:, ids].tocoo()
        return {'cell_id': self.cells.iloc[index].cell_id,
                'gene_id': ids.copy(), 'gene_symbol': self.genes[ids],
                'x': self.features[lo:hi].copy(), 'num_nodes': len(ids),
                'edge_index': np.stack([graph.row, graph.col]).astype(np.int64),
                'edge_attr': graph.data.astype(np.float32),
                'y': int(self.cells.iloc[index].y), 'relation': self.relation}

    def to_pyg(self, index):
        import torch
        from torch_geometric.data import Data
        item = self[index]
        return Data(x=torch.from_numpy(item['x']),
                    gene_id=torch.from_numpy(item['gene_id'].astype(np.int64)),
                    edge_index=torch.from_numpy(item['edge_index']),
                    edge_attr=torch.from_numpy(item['edge_attr'][:, None]),
                    y=torch.tensor([item['y']], dtype=torch.long),
                    num_nodes=item['num_nodes'], cell_id=item['cell_id'])

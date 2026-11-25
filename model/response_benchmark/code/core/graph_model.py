"""Parallel residual GATv2/GINE graph classifier with shared gene identity and expression input."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv,GINEConv,global_mean_pool


class GraphClassifier(nn.Module):
    def __init__(self,n_genes,hidden=64,layers=2,heads=4,gene_dim=16,dropout=.2,expr_mean=0.,expr_std=1.):
        super().__init__()
        assert hidden%heads==0
        self.config=dict(n_genes=n_genes,hidden=hidden,layers=layers,heads=heads,gene_dim=gene_dim,
                         dropout=dropout,expr_mean=expr_mean,expr_std=expr_std)
        self.register_buffer('expr_mean',torch.tensor(float(expr_mean)))
        self.register_buffer('expr_std',torch.tensor(float(expr_std)))
        self.identity=nn.Embedding(n_genes,gene_dim)
        self.project=nn.Linear(gene_dim+1,hidden)
        self.gat=nn.ModuleList([GATv2Conv(hidden,hidden//heads,heads=heads,edge_dim=1,
                              add_self_loops=False,dropout=dropout) for _ in range(layers)])
        self.gine=nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)),
                               edge_dim=1,train_eps=True) for _ in range(layers)])
        self.gat_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.gine_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,2))
        self.dropout=dropout

    def forward(self,x,edge_index,edge_attr,batch,gene_id):
        expression=(x[:,:1]-self.expr_mean)/self.expr_std.clamp_min(1e-6)
        h=F.relu(self.project(torch.cat([expression,self.identity(gene_id)],dim=-1)))
        a,b=h,h
        for gat,gine,an,bn in zip(self.gat,self.gine,self.gat_norm,self.gine_norm):
            a=an(a+F.dropout(F.relu(gat(a,edge_index,edge_attr)),p=self.dropout,training=self.training))
            b=bn(b+F.dropout(F.relu(gine(b,edge_index,edge_attr)),p=self.dropout,training=self.training))
        return self.head(torch.cat([global_mean_pool(a,batch),global_mean_pool(b,batch)],dim=-1))


def load_checkpoint(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=GraphClassifier(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint

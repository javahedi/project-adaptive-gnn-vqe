import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing



# ============================================================
# Edge-conditioned Message Passing
# ============================================================
class EdgeMP(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        m = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        return self.upd_mlp(torch.cat([x, m], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


# ============================================================
# Pointer GNN
# ============================================================
class PointerGNN(nn.Module):
    """
    
    """
    def __init__(self, node_in, edge_in, hidden=96, mp_layers=5):
        super().__init__()

        self.node_enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )

        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )

        self.mps = nn.ModuleList([EdgeMP(hidden, hidden, hidden) for _ in range(mp_layers)])

        self.edge_score = nn.Sequential(
            nn.LayerNorm(5 * hidden),
            nn.Linear(5 * hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_enc(x)
        e = self.edge_enc(edge_attr)

        for mp in self.mps:
            x = x + mp(x, edge_index, e)

        src, dst = edge_index
        #edge_feat = torch.cat([x[src], x[dst], e], dim=-1)
        edge_feat = torch.cat([ x[src], x[dst], x[src] - x[dst], x[src] * x[dst],
                              e], dim=-1)
        return self.edge_score(edge_feat).squeeze(-1)
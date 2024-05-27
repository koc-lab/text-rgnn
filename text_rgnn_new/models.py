import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, Linear, SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels) -> torch.nn.Module:
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, use_edge_attr) -> torch.nn.Module:
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

        self.use_edge_attr = use_edge_attr

    def forward(self, x, edge_index, edge_attr):
        if self.use_edge_attr:
            x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr) + self.lin2(x)
        else:
            x = self.conv1(x, edge_index) + self.lin1(x)
            x = x.relu()
            x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class HomoGCN(torch.nn.Module):
    def __init__(self, n_feats, n_class, n_hidden):
        super().__init__()
        self.conv1 = GCNConv(n_feats, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index, edge_attr)

        return F.log_softmax(x, dim=1)

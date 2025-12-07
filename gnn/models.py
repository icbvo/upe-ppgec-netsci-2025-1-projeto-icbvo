# models.py
#
# GNN models for node-level regression on the APS Citation Network.
# Task: predict future citation counts for each paper.

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class GCNRegressor(nn.Module):
    """
    Graph Convolutional Network (GCN) for node-level regression.

    Input:
        - x: node features [num_nodes, in_channels]
        - edge_index: graph edges [2, num_edges]

    Output:
        - y_pred: predicted value for each node [num_nodes]
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        out = self.lin(x).squeeze(-1)  # [num_nodes]
        return out


class GraphSAGERegressor(nn.Module):
    """
    GraphSAGE model for node-level regression.

    This model is often more scalable and robust for large graphs
    compared to vanilla GCN.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        out = self.lin(x).squeeze(-1)
        return out


def get_model(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Helper to create a model by name.

    Args:
        name: "gcn" or "sage"
        in_channels: number of input features per node
        hidden_channels: hidden layer size
        dropout: dropout probability

    Returns:
        Torch nn.Module instance.
    """
    name = name.lower()
    if name == "gcn":
        return GCNRegressor(in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            dropout=dropout)
    elif name == "sage" or name == "graphsage":
        return GraphSAGERegressor(in_channels=in_channels,
                                  hidden_channels=hidden_channels,
                                  dropout=dropout)
    else:
        raise ValueError(f"Unknown model name: {name}. Use 'gcn' or 'sage'.")


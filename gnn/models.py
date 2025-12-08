# gnn/models.py
#
# GNN encoder models and edge predictor for link prediction
# in an undirected collaboration network.

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class GCNEncoder(nn.Module):
    """
    Simple 2-layer GCN encoder.
    Produces node embeddings given node features and edge_index.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # x shape: [num_nodes, out_channels]
        return x


class GraphSAGEEncoder(nn.Module):
    """
    Simple 2-layer GraphSAGE encoder.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # x shape: [num_nodes, out_channels]
        return x


class LinkPredictor(nn.Module):
    """
    MLP-based edge predictor.

    Input:
        embeddings of node u and node v (z_u, z_v) with shape [batch_size, emb_dim]

    Output:
        logits (before sigmoid) for edge existence, shape [batch_size]
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        # Concatenate embeddings of the two endpoints
        x = torch.cat([z_u, z_v], dim=-1)  # [batch_size, 2 * emb_dim]
        logits = self.mlp(x).squeeze(-1)   # [batch_size]
        return logits


def get_encoder(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Helper to create a GNN encoder by name.

    Args:
        name: "gcn" or "sage" / "graphsage"
        in_channels: number of input features per node
        hidden_channels: hidden dimension
        out_channels: embedding dimension
        dropout: dropout rate

    Returns:
        nn.Module: a GCNEncoder or GraphSAGEEncoder
    """
    name = name.lower()
    if name == "gcn":
        return GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
    elif name in ("sage", "graphsage"):
        return GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder name: {name}. Use 'gcn' or 'sage'.")

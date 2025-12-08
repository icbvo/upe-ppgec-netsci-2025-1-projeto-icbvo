# gnn/export_embeddings.py
#
# Compute node embeddings Z using the trained GNN encoder
# and save them to results/embeddings.pt

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from models import get_encoder
from utils import set_seed, print_graph_summary

# Config must be consistente with train_link_prediction_gnn.py
DATA_DIR = "./data"
EDGE_LIST_FILE = os.path.join(DATA_DIR, "collaboration.edgelist.txt")
MODEL_PATH = os.path.join("results", "linkpred_gnn_best.pth")

ENCODER_NAME = "gcn"
EMB_DIM = 64
DROPOUT = 0.2
SEED = 42


def load_edge_list(path: str):
    """
    Load undirected collaboration network from a simple edge list
    file with no header and two integer columns (u, v).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v"])
    df = df[df["u"] != df["v"]].copy()  # remove self-loops

    # canonical undirected representation
    u_min = np.minimum(df["u"].values, df["v"].values)
    v_max = np.maximum(df["u"].values, df["v"].values)
    df["u"] = u_min
    df["v"] = v_max

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def df_to_edge_index(df: pd.DataFrame,
                     num_nodes: int,
                     undirected: bool = True):
    """
    Convert dataframe of edges [u, v] to PyG edge_index tensor.
    """
    u = torch.tensor(df["u"].values, dtype=torch.long)
    v = torch.tensor(df["v"].values, dtype=torch.long)

    if undirected:
        edge_index = torch.stack(
            [torch.cat([u, v]), torch.cat([v, u])],
            dim=0
        )
    else:
        edge_index = torch.stack([u, v], dim=0)

    return edge_index


def make_node_features(df_edges: pd.DataFrame, num_nodes: int):
    """
    Node features = normalized degree of each node.
    """
    deg = np.zeros(num_nodes, dtype=float)
    for _, row in df_edges.iterrows():
        deg[row["u"]] += 1
        deg[row["v"]] += 1

    deg = (deg - deg.mean()) / (deg.std() + 1e-9)
    x = torch.tensor(deg[:, None], dtype=torch.float)  # shape [N, 1]
    return x


def main():
    set_seed(SEED)

    print(f"Loading dataset from: {EDGE_LIST_FILE}")
    df_edges = load_edge_list(EDGE_LIST_FILE)
    num_edges = len(df_edges)
    num_nodes = int(max(df_edges["u"].max(), df_edges["v"].max()) + 1)

    print_graph_summary(num_nodes=num_nodes, num_edges=num_edges)

    edge_index = df_to_edge_index(df_edges, num_nodes, undirected=True)
    x = make_node_features(df_edges, num_nodes)

    data = Data(x=x, edge_index=edge_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Build encoder (mesma configuração do treino)
    encoder = get_encoder(
        name=ENCODER_NAME,
        in_channels=data.num_node_features,
        hidden_channels=EMB_DIM,
        out_channels=EMB_DIM,
        dropout=DROPOUT,
    ).to(device)

    # Carregar pesos treinados
    print(f"Loading trained encoder from: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()

    with torch.no_grad():
        Z = encoder(data.x, data.edge_index)  # [N, EMB_DIM]

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "embeddings.pt")
    torch.save(Z, out_path)

    print(f"Saved embeddings to: {out_path}")
    print(f"Embeddings shape: {tuple(Z.shape)}")


if __name__ == "__main__":
    main()

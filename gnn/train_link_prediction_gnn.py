# train_link_prediction_gnn.py
#
# Full training script for link prediction on a collaboration network
# using a GNN encoder (GCN or GraphSAGE) and an MLP edge predictor.
#
# Expected input file:
#   ../data/collaboration.edgelist.txt
# Format (no header, tab-separated, two integer columns):
#   u<tab>v
#   0   1680
#   0   6918
#   ...

import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from models import get_encoder, LinkPredictor


# -------------------------
# CONFIG
# -------------------------

DATA_DIR = "../data"
EDGE_LIST_FILE = os.path.join(DATA_DIR, "collaboration.edgelist.txt")

ENCODER_NAME = "gcn"       # "gcn" or "sage"
EMB_DIM = 64
HIDDEN_PRED = 64
DROPOUT = 0.2

LR = 1e-3
EPOCHS = 100
WEIGHT_DECAY = 1e-4
SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test will be 1 - train - val


# -------------------------
# UTILS
# -------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_edge_list(path: str):
    """
    Load undirected collaboration network from a simple edge list
    file with no header and two integer columns (u, v).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v"])
    # Remove self-loops
    df = df[df["u"] != df["v"]].copy()

    # Canonical undirected representation: (min, max)
    u_min = np.minimum(df["u"].values, df["v"].values)
    v_max = np.maximum(df["u"].values, df["v"].values)
    df["u"] = u_min
    df["v"] = v_max

    # Drop duplicate edges
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def build_splits(df_edges: pd.DataFrame,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1):
    """
    Split edges into train/val/test sets.
    df_edges: DataFrame with columns [u, v] for undirected edges.
    """
    num_edges = len(df_edges)
    perm = np.random.permutation(num_edges)
    df_edges = df_edges.iloc[perm].reset_index(drop=True)

    n_train = int(train_ratio * num_edges)
    n_val = int(val_ratio * num_edges)
    n_test = num_edges - n_train - n_val

    df_train = df_edges.iloc[:n_train].reset_index(drop=True)
    df_val = df_edges.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_edges.iloc[n_train + n_val:].reset_index(drop=True)

    return df_train, df_val, df_test


def df_to_edge_index(df: pd.DataFrame, num_nodes: int, undirected: bool = True):
    """
    Convert a dataframe of edges [u, v] to a PyG edge_index tensor.
    If undirected=True, adds edges in both directions.
    """
    u = torch.tensor(df["u"].values, dtype=torch.long)
    v = torch.tensor(df["v"].values, dtype=torch.long)

    if undirected:
        edge_index = torch.stack([torch.cat([u, v]),
                                  torch.cat([v, u])], dim=0)
    else:
        edge_index = torch.stack([u, v], dim=0)
    return edge_index


def make_node_features(df_edges: pd.DataFrame, num_nodes: int):
    """
    Very simple node features: normalized degree.
    """
    deg = np.zeros(num_nodes, dtype=float)
    for _, row in df_edges.iterrows():
        deg[row["u"]] += 1.0
        deg[row["v"]] += 1.0
    deg = (deg - deg.mean()) / (deg.std() + 1e-9)
    x = torch.tensor(deg[:, None], dtype=torch.float)  # [num_nodes, 1]
    return x


def sample_negative_edges(edge_index_full: torch.Tensor,
                          num_nodes: int,
                          num_neg_samples: int):
    """
    Uses torch_geometric.utils.negative_sampling to generate negative edges.
    edge_index_full is treated as undirected adjacency for collision checks.
    """
    neg_edge_index = negative_sampling(
        edge_index=edge_index_full,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    )
    return neg_edge_index


def get_edge_batches(pos_edge_index: torch.Tensor,
                     neg_edge_index: torch.Tensor,
                     device: torch.device):
    """
    Build tensors of (u, v, label) for positive and negative edges.
    Returns:
        u_all: [num_edges]
        v_all: [num_edges]
        y_all: [num_edges] (0/1)
    """
    pos_u, pos_v = pos_edge_index
    neg_u, neg_v = neg_edge_index

    u_all = torch.cat([pos_u, neg_u], dim=0).to(device)
    v_all = torch.cat([pos_v, neg_v], dim=0).to(device)

    y_pos = torch.ones(pos_u.size(0), dtype=torch.float32)
    y_neg = torch.zeros(neg_u.size(0), dtype=torch.float32)
    y_all = torch.cat([y_pos, y_neg], dim=0).to(device)

    return u_all, v_all, y_all


def edge_predict(encoder, predictor, data, u, v):
    """
    Compute logits for edges (u, v) given encoder and predictor.
    """
    z = encoder(data.x, data.edge_index)  # [num_nodes, emb_dim]
    z_u = z[u]
    z_v = z[v]
    logits = predictor(z_u, z_v)  # [num_edges]
    return logits


# -------------------------
# TRAIN / EVAL
# -------------------------

def train_one_epoch(encoder, predictor, data,
                    pos_edge_index_train, neg_edge_index_train,
                    optimizer, criterion, device):

    encoder.train()
    predictor.train()
    optimizer.zero_grad()

    u_all, v_all, y_all = get_edge_batches(
        pos_edge_index_train,
        neg_edge_index_train,
        device,
    )

    logits = edge_predict(encoder, predictor, data, u_all, v_all)
    loss = criterion(logits, y_all)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(encoder, predictor, data,
             pos_edge_index, neg_edge_index, device):
    encoder.eval()
    predictor.eval()

    u_all, v_all, y_all = get_edge_batches(
        pos_edge_index,
        neg_edge_index,
        device,
    )

    logits = edge_predict(encoder, predictor, data, u_all, v_all)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y_all.cpu().numpy()

    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    return auc, ap


# -------------------------
# MAIN
# -------------------------

def main():
    set_seed(SEED)

    print("Loading edge list from:", EDGE_LIST_FILE)
    df_edges = load_edge_list(EDGE_LIST_FILE)
    print("Number of undirected edges:", len(df_edges))

    num_nodes = int(max(df_edges["u"].max(), df_edges["v"].max()) + 1)
    print("Number of nodes:", num_nodes)

    # Splits
    df_train, df_val, df_test = build_splits(df_edges, TRAIN_RATIO, VAL_RATIO)
    print("Train edges:", len(df_train))
    print("Val edges:", len(df_val))
    print("Test edges:", len(df_test))

    # Edge indices (positive edges)
    pos_train_edge_index = df_to_edge_index(df_train, num_nodes, undirected=False)
    pos_val_edge_index = df_to_edge_index(df_val, num_nodes, undirected=False)
    pos_test_edge_index = df_to_edge_index(df_test, num_nodes, undirected=False)

    # Full undirected edge_index for negative sampling collision checks
    full_edge_index = df_to_edge_index(df_edges, num_nodes, undirected=True)

    # Train adjacency (only train edges) for GNN message passing
    train_edge_index = df_to_edge_index(df_train, num_nodes, undirected=True)

    # Node features (based on full graph degree)
    x = make_node_features(df_edges, num_nodes)

    data = Data(x=x, edge_index=train_edge_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Instantiate models
    encoder = get_encoder(
        name=ENCODER_NAME,
        in_channels=data.num_node_features,
        hidden_channels=EMB_DIM,
        out_channels=EMB_DIM,
        dropout=DROPOUT,
    ).to(device)

    predictor = LinkPredictor(emb_dim=EMB_DIM, hidden_dim=HIDDEN_PRED).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR,

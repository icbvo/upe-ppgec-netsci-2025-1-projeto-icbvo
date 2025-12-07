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
from utils import set_seed, print_graph_summary, save_metrics


# =============================================================================
# CONFIG
# =============================================================================

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
VAL_RATIO = 0.1  # test = 1 - train - val


# =============================================================================
# UTILS (LOCAL)
# =============================================================================

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


def build_splits(df_edges: pd.DataFrame,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1):
    """
    Split edges into train/val/test sets.
    """
    num_edges = len(df_edges)
    perm = np.random.permutation(num_edges)
    df_edges = df_edges.iloc[perm].reset_index(drop=True)

    n_train = int(train_ratio * num_edges)
    n_val = int(val_ratio * num_edges)

    df_train = df_edges.iloc[:n_train].reset_index(drop=True)
    df_val = df_edges.iloc[n_train:n_train+n_val].reset_index(drop=True)
    df_test = df_edges.iloc[n_train+n_val:].reset_index(drop=True)

    return df_train, df_val, df_test


def df_to_edge_index(df: pd.DataFrame,
                     num_nodes: int,
                     undirected: bool = True):
    """
    Convert dataframe of edges [u, v] to PyG edge_index tensor.
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
    Node features = normalized degree of each node.
    """
    deg = np.zeros(num_nodes, dtype=float)
    for _, row in df_edges.iterrows():
        deg[row["u"]] += 1
        deg[row["v"]] += 1

    deg = (deg - deg.mean()) / (deg.std() + 1e-9)
    x = torch.tensor(deg[:, None], dtype=torch.float)  # shape [N, 1]
    return x


def sample_negative_edges(edge_index_full: torch.Tensor,
                          num_nodes: int,
                          num_neg_samples: int):
    """
    Negative sampling based on existing edges (collision-free).
    """
    return negative_sampling(
        edge_index=edge_index_full,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    )


def get_edge_batches(pos_edge_index: torch.Tensor,
                     neg_edge_index: torch.Tensor,
                     device: torch.device):

    pos_u, pos_v = pos_edge_index
    neg_u, neg_v = neg_edge_index

    u_all = torch.cat([pos_u, neg_u]).to(device)
    v_all = torch.cat([pos_v, neg_v]).to(device)

    y_pos = torch.ones(pos_u.size(0), dtype=torch.float32)
    y_neg = torch.zeros(neg_u.size(0), dtype=torch.float32)
    y_all = torch.cat([y_pos, y_neg]).to(device)

    return u_all, v_all, y_all


def edge_predict(encoder, predictor, data, u, v):
    z = encoder(data.x, data.edge_index)
    z_u = z[u]
    z_v = z[v]
    return predictor(z_u, z_v)


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_epoch(
    encoder, predictor, data,
    pos_train_edge_index, neg_train_edge_index,
    optimizer, criterion, device
):
    encoder.train()
    predictor.train()
    optimizer.zero_grad()

    u_all, v_all, y_all = get_edge_batches(
        pos_train_edge_index, neg_train_edge_index, device
    )

    logits = edge_predict(encoder, predictor, data, u_all, v_all)
    loss = criterion(logits, y_all)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    encoder, predictor, data,
    pos_edge_index, neg_edge_index, device
):
    encoder.eval()
    predictor.eval()

    u_all, v_all, y_all = get_edge_batches(
        pos_edge_index, neg_edge_index, device
    )

    logits = edge_predict(encoder, predictor, data, u_all, v_all)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y_all.cpu().numpy()

    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    return auc, ap


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(SEED)

    print("\nLoading dataset...")
    df_edges = load_edge_list(EDGE_LIST_FILE)
    num_edges = len(df_edges)
    num_nodes = int(max(df_edges["u"].max(), df_edges["v"].max()) + 1)

    df_train, df_val, df_test = build_splits(df_edges, TRAIN_RATIO, VAL_RATIO)

    print_graph_summary(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_train_edges=len(df_train),
        num_val_edges=len(df_val),
        num_test_edges=len(df_test),
    )

    # Build edge_index tensors
    pos_train_edge_index = df_to_edge_index(df_train, num_nodes, undirected=False)
    pos_val_edge_index = df_to_edge_index(df_val, num_nodes, undirected=False)
    pos_test_edge_index = df_to_edge_index(df_test, num_nodes, undirected=False)

    # Full adjacency for negative sampling
    full_edge_index = df_to_edge_index(df_edges, num_nodes, undirected=True)

    # Only training graph is used by GNN
    train_edge_index = df_to_edge_index(df_train, num_nodes, undirected=True)

    # Node features
    x = make_node_features(df_edges, num_nodes)
    data = Data(x=x, edge_index=train_edge_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # ---------------------
    # Model
    # ---------------------
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
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Pre-sample negatives for val/test
    print("Sampling fixed negative edges for validation and test...")
    neg_val_edge_index = sample_negative_edges(full_edge_index, num_nodes, pos_val_edge_index.size(1))
    neg_test_edge_index = sample_negative_edges(full_edge_index, num_nodes, pos_test_edge_index.size(1))

    best_val_auc = 0.0
    best_state = None

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        # Fresh negative samples for training
        neg_train_edge_index = sample_negative_edges(full_edge_index, num_nodes, pos_train_edge_index.size(1)).to(device)

        loss = train_one_epoch(
            encoder, predictor, data,
            pos_train_edge_index.to(device),
            neg_train_edge_index,
            optimizer, criterion, device
        )

        val_auc, val_ap = evaluate(
            encoder, predictor, data,
            pos_val_edge_index.to(device),
            neg_val_edge_index.to(device),
            device
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss={loss:.4f} | "
                f"Val AUC={val_auc:.4f} | "
                f"Val AP={val_ap:.4f}"
            )

    # Load best model
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])

    # Final test
    test_auc, test_ap = evaluate(
        encoder, predictor, data,
        pos_test_edge_index.to(device),
        neg_test_edge_index.to(device),
        device
    )

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP:  {test_ap:.4f}")

    # Save metrics
    save_metrics(
        "linkpred_metrics.json",
        {
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
            "encoder": ENCODER_NAME,
            "epochs": EPOCHS,
        }
    )

    # Save model
    torch.save(best_state, "linkpred_gnn_best.pth")
    print("\nSaved best model to linkpred_gnn_best.pth")


if __name__ == "__main__":
    main()

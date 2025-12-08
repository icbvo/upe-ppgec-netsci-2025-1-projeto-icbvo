# gnn/train_link_prediction_gnn.py
#
# Full training script for link prediction on a collaboration network
# using a GNN encoder (GCN or GraphSAGE) and an MLP edge predictor.
#
# Expected input file (relative to project root):
#   ./data/collaboration.edgelist.txt
#
# Format (no header, whitespace-separated, two integer columns):
#   u v
#   0 1680
#   0 6918
#   ...

import os
import math
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from gnn.models import get_encoder, LinkPredictor
from gnn.utils import set_seed, print_graph_summary, save_metrics


# =============================================================================
# CONFIG
# =============================================================================

# Caminho do arquivo de arestas (do ponto de vista do container/projeto)
DATA_DIR = "./data"
EDGE_LIST_FILE = os.path.join(DATA_DIR, "collaboration.edgelist.txt")

# Encoder: "gcn" ou "sage"
ENCODER_NAME = "gcn"

# Dimensões
EMB_DIM = 64        # dimensão das embeddings dos nós
HIDDEN_PRED = 64    # dimensão da camada oculta do MLP do preditor de arestas
DROPOUT = 0.2

# Hiperparâmetros de treino
LR = 1e-3
EPOCHS = 100
WEIGHT_DECAY = 1e-4
SEED = 42

# Splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test = 1 - train - val

# Diretório de resultados
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# LOADING E PRÉ-PROCESSAMENTO DO GRAFO
# =============================================================================

def load_edge_list(path: str) -> pd.DataFrame:
    """
    Carrega a rede de colaboração não-dirigida a partir de um arquivo
    de lista de arestas sem cabeçalho, duas colunas (u, v) inteiras.
    Remove self-loops e duplicatas.

    Retorna um DataFrame com colunas ["u", "v"] em forma canônica:
    u <= v.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Edge list file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v"])
    # remove self-loops
    df = df[df["u"] != df["v"]].copy()

    # representação canônica para arestas não-dirigidas
    u_min = np.minimum(df["u"].values, df["v"].values)
    v_max = np.maximum(df["u"].values, df["v"].values)
    df["u"] = u_min
    df["v"] = v_max

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_splits(
    df_edges: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Embaralha e divide as arestas em conjuntos de treino/val/test.
    """
    num_edges = len(df_edges)
    perm = np.random.permutation(num_edges)
    df_edges_shuffled = df_edges.iloc[perm].reset_index(drop=True)

    n_train = int(train_ratio * num_edges)
    n_val = int(val_ratio * num_edges)

    df_train = df_edges_shuffled.iloc[:n_train].reset_index(drop=True)
    df_val = df_edges_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_edges_shuffled.iloc[n_train + n_val:].reset_index(drop=True)

    return df_train, df_val, df_test


def df_to_edge_index(
    df: pd.DataFrame,
    num_nodes: int,
    undirected: bool = True
) -> torch.Tensor:
    """
    Converte um DataFrame de arestas [u, v] em um tensor edge_index
    no formato PyG: shape [2, num_edges] (ou [2, 2 * num_edges] se undirected=True).
    """
    u = torch.tensor(df["u"].values, dtype=torch.long)
    v = torch.tensor(df["v"].values, dtype=torch.long)

    if undirected:
        # duplicamos as arestas nas duas direções
        edge_index = torch.stack(
            [torch.cat([u, v]),
             torch.cat([v, u])],
            dim=0
        )
    else:
        edge_index = torch.stack([u, v], dim=0)

    return edge_index


def make_node_features(df_edges: pd.DataFrame, num_nodes: int) -> torch.Tensor:
    """
    Cria features de nós baseadas no grau normalizado.
    Retorna tensor x de shape [num_nodes, 1].
    """
    deg = np.zeros(num_nodes, dtype=float)
    for _, row in df_edges.iterrows():
        deg[row["u"]] += 1
        deg[row["v"]] += 1

    mean_deg = deg.mean()
    std_deg = deg.std()
    deg_norm = (deg - mean_deg) / (std_deg + 1e-9)

    x = torch.tensor(deg_norm[:, None], dtype=torch.float32)
    return x


def sample_negative_edges(
    edge_index_full: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int
) -> torch.Tensor:
    """
    Faz negative sampling com base nas arestas existentes, sem colisão.
    Utiliza torch_geometric.utils.negative_sampling.
    """
    neg_edge_index = negative_sampling(
        edge_index=edge_index_full,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    )
    return neg_edge_index


def get_edge_batches(
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dado um conjunto de arestas positivas e negativas,
    retorna (u_all, v_all, y_all), prontos para alimentar o modelo.
    """
    pos_u, pos_v = pos_edge_index
    neg_u, neg_v = neg_edge_index

    u_all = torch.cat([pos_u, neg_u]).to(device)
    v_all = torch.cat([pos_v, neg_v]).to(device)

    y_pos = torch.ones(pos_u.size(0), dtype=torch.float32)
    y_neg = torch.zeros(neg_u.size(0), dtype=torch.float32)
    y_all = torch.cat([y_pos, y_neg]).to(device)

    return u_all, v_all, y_all


def edge_predict(
    encoder: nn.Module,
    predictor: nn.Module,
    data: Data,
    u: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """
    Executa o encoder GNN para obter embeddings de nós e depois
    usa o preditor de links para calcular os logits das arestas (u, v).
    """
    z = encoder(data.x, data.edge_index)  # [num_nodes, emb_dim]
    z_u = z[u]
    z_v = z[v]
    logits = predictor(z_u, z_v)
    return logits


# =============================================================================
# TREINO / AVALIAÇÃO
# =============================================================================

def train_one_epoch(
    encoder: nn.Module,
    predictor: nn.Module,
    data: Data,
    pos_train_edge_index: torch.Tensor,
    neg_train_edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Executa uma época de treinamento.
    """
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
    return float(loss.item())


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    predictor: nn.Module,
    data: Data,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device
) -> Tuple[float, float]:
    """
    Avalia o modelo em um conjunto de arestas positivas e negativas.
    Retorna (AUC, AP).
    """
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
    return float(auc), float(ap)


@torch.no_grad()
def compute_node_embeddings(
    encoder: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Computa as embeddings de nós usando o encoder treinado.
    Retorna um tensor [num_nodes, emb_dim] na CPU.
    """
    encoder.eval()
    data = Data(x=x, edge_index=edge_index).to(device)
    z = encoder(data.x, data.edge_index)
    return z.cpu()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Semente global para reprodutibilidade
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading dataset from: {EDGE_LIST_FILE}")
    df_edges = load_edge_list(EDGE_LIST_FILE)

    num_edges = len(df_edges)
    num_nodes = int(max(df_edges["u"].max(), df_edges["v"].max()) + 1)

    # Divisão em treino/val/test
    df_train, df_val, df_test = build_splits(df_edges, TRAIN_RATIO, VAL_RATIO)

    print_graph_summary(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_train_edges=len(df_train),
        num_val_edges=len(df_val),
        num_test_edges=len(df_test),
    )

    # Arestas positivas (direcionadas) para treino/val/test
    pos_train_edge_index = df_to_edge_index(df_train, num_nodes, undirected=False)
    pos_val_edge_index = df_to_edge_index(df_val, num_nodes, undirected=False)
    pos_test_edge_index = df_to_edge_index(df_test, num_nodes, undirected=False)

    # Grafo completo (não-dirigido) para negative sampling
    full_edge_index = df_to_edge_index(df_edges, num_nodes, undirected=True)

    # Grafo de treino (não-dirigido) usado pelo GNN
    train_edge_index = df_to_edge_index(df_train, num_nodes, undirected=True)

    # Node features
    x = make_node_features(df_edges, num_nodes)
    data = Data(x=x, edge_index=train_edge_index).to(device)

    # ---------------------
    # Modelo: encoder GNN + preditor de links
    # ---------------------
    encoder = get_encoder(
        name=ENCODER_NAME,
        in_channels=data.num_node_features,
        hidden_channels=EMB_DIM,
        out_channels=EMB_DIM,
        dropout=DROPOUT,
    ).to(device)

    predictor = LinkPredictor(
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_PRED,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Negative samples fixos para val/test
    print("\nSampling fixed negative edges for validation and test...")
    neg_val_edge_index = sample_negative_edges(
        full_edge_index, num_nodes, pos_val_edge_index.size(1)
    )
    neg_test_edge_index = sample_negative_edges(
        full_edge_index, num_nodes, pos_test_edge_index.size(1)
    )

    best_val_auc = 0.0
    best_state = None
    training_history: List[Dict[str, float]] = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        # Negative samples reamostrados a cada época para o treino
        neg_train_edge_index = sample_negative_edges(
            full_edge_index, num_nodes, pos_train_edge_index.size(1)
        ).to(device)

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

        # Salvar melhor modelo com base no AUC de validação
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
            }

        # Guardar histórico para plot posterior
        history_entry = {
            "epoch": epoch,
            "train_loss": loss,
            "val_auc": val_auc,
            "val_ap": val_ap,
        }
        training_history.append(history_entry)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss={loss:.4f} | "
                f"Val AUC={val_auc:.4f} | "
                f"Val AP={val_ap:.4f}"
            )

    # Recarregar o melhor modelo (se encontrado)
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])

    # Avaliação final no conjunto de teste
    test_auc, test_ap = evaluate(
        encoder, predictor, data,
        pos_test_edge_index.to(device),
        neg_test_edge_index.to(device),
        device
    )

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP:  {test_ap:.4f}")

    # -------------------------------------------------------------------------
    # Salvar métricas agregadas
    # -------------------------------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "linkpred_metrics.json")
    save_metrics(
        metrics_path,
        {
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
            "encoder": ENCODER_NAME,
            "epochs": EPOCHS,
        }
    )

    # -------------------------------------------------------------------------
    # Salvar histórico de treino (para notebooks)
    # -------------------------------------------------------------------------
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)
    print(f"Saved training history to {history_path}")

    # -------------------------------------------------------------------------
    # Salvar embeddings finais dos nós (para t-SNE / visualização)
    # -------------------------------------------------------------------------
    # Aqui usamos o grafo de treino (não-dirigido); você pode mudar para full_edge_index
    # se quiser propagar em todas as arestas.
    z = compute_node_embeddings(
        encoder,
        x=x,
        edge_index=train_edge_index,
        device=device,
    )
    emb_path = os.path.join(RESULTS_DIR, "node_embeddings.pt")
    torch.save(z, emb_path)
    print(f"Saved node embeddings to {emb_path}")

    # Salvar também o melhor modelo completo
    model_path = os.path.join(RESULTS_DIR, "linkpred_gnn_best.pth")
    torch.save(best_state, model_path)
    print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    main()

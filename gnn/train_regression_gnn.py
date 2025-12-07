# train_regression_gnn.py
#
# Treinamento de uma GNN de regressao (GCN ou GraphSAGE)
# para prever o numero de citacoes futuras de artigos da APS
# a partir da estrutura da APS Citation Network.

import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

from models import get_model
from utils import set_seed, print_data_summary, save_metrics


# ==========================
# CONFIGURACOES GERAIS
# ==========================

# Diretoria dos dados (ajuste se necessario)
DATA_DIR = "../data"
NODES_CSV = os.path.join(DATA_DIR, "nodes.csv")
EDGES_CSV = os.path.join(DATA_DIR, "edges.csv")

# Nome das colunas esperadas nos arquivos CSV:
# Ajuste para refletir seus dados reais.
COL_PAPER_ID = "paper_id"
COL_YEAR = "year"

COL_CITING = "citing"
COL_CITED = "cited"
COL_YEAR_CITING = "year_citing"  # ano do artigo que cita

# Parametros da tarefa temporal
Y_CUT = 2005       # ano de corte do grafo
DELTA_T = 5        # janela futura de citacoes (anos apos Y_CUT)

# Parametros de modelo / treino
MODEL_NAME = "gcn"          # "gcn" ou "sage"
HIDDEN_CHANNELS = 64
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 50
WEIGHT_DECAY = 1e-4
SEED = 42


# ==========================
# FUNCOES AUXILIARES
# ==========================

def load_data(nodes_csv: str, edges_csv: str):
    """
    Carrega os dados de nos e arestas a partir de arquivos CSV.
    Espera, por padrao:
    - nodes.csv: paper_id, year, ...
    - edges.csv: citing, cited, year_citing, ...
    """
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    return nodes, edges


def build_temporal_graph(nodes: pd.DataFrame, edges: pd.DataFrame,
                         y_cut: int, delta_t: int) -> Data:
    """
    Constroi o grafo dirigido em Y_cut e gera:
    - features de nos (x)
    - alvo (y) = numero de citacoes futuras entre (Y_cut, Y_cut + delta_t]
    - masks de treino, validacao e teste baseadas no ano de publicacao.
    """

    # Filtra nos ate o ano de corte
    nodes_cut = nodes[nodes[COL_YEAR] <= y_cut].copy().reset_index(drop=True)

    # Mapear ID de artigo -> indice de no
    id2idx = {pid: i for i, pid in enumerate(nodes_cut[COL_PAPER_ID])}
    num_nodes = len(nodes_cut)

    if num_nodes == 0:
        raise ValueError("Nao ha nos apos aplicar o filtro de ano. Verifique COL_YEAR e Y_CUT.")

    # Filtra arestas ate o ano de corte
    edges_cut = edges[
        (edges[COL_CITING].isin(id2idx)) &
        (edges[COL_CITED].isin(id2idx)) &
        (edges[COL_YEAR_CITING] <= y_cut)
    ].copy()

    src_idx = edges_cut[COL_CITING].map(id2idx).to_numpy()
    dst_idx = edges_cut[COL_CITED].map(id2idx).to_numpy()

    if len(src_idx) == 0:
        raise ValueError("Nao ha arestas apos o filtro de corte temporal. Verifique o dataset ou as colunas de ano.")

    # Monta matriz esparsa de adjacencia (direcionada)
    data_adj = np.ones_like(src_idx, dtype=np.float32)
    adj = coo_matrix((data_adj, (src_idx, dst_idx)), shape=(num_nodes, num_nodes))

    edge_index, _ = from_scipy_sparse_matrix(adj)  # [2, num_edges]

    # -------- Features de nos (x) --------
    years = nodes_cut[COL_YEAR].to_numpy().astype(float)
    year_norm = (years - years.min()) / (years.max() - years.min() + 1e-9)

    # In-degree ate Y_CUT
    in_deg = np.zeros(num_nodes, dtype=float)
    for d in dst_idx:
        in_deg[d] += 1.0
    in_deg_norm = (in_deg - in_deg.mean()) / (in_deg.std() + 1e-9)

    # Aqui apenas 2 features: ano normalizado e in_degree normalizado
    # Voce pode adicionar outras (PageRank, area, etc.) se tiver.
    X = np.stack([year_norm, in_deg_norm], axis=1)
    x = torch.tensor(X, dtype=torch.float)

    # -------- Alvo (y) = citacoes futuras --------
    future_edges = edges[
        (edges[COL_CITED].isin(id2idx)) &
        (edges[COL_YEAR_CITING] > y_cut) &
        (edges[COL_YEAR_CITING] <= y_cut + delta_t)
    ].copy()

    y_arr = np.zeros(num_nodes, dtype=float)
    for _, row in future_edges.iterrows():
        cited_idx = id2idx[row[COL_CITED]]
        y_arr[cited_idx] += 1.0

    y = torch.tensor(y_arr, dtype=torch.float)

    # -------- Masks temporais --------
    years_tensor = torch.tensor(years, dtype=torch.long)

    # Exemplo de split temporal simples:
    # treino: anos <= y_cut - 10
    # valid: (y_cut - 10, y_cut - 5]
    # teste: anos > y_cut - 5
    train_mask = years_tensor <= (y_cut - 10)
    val_mask = (years_tensor > (y_cut - 10)) & (years_tensor <= (y_cut - 5))
    test_mask = years_tensor > (y_cut - 5)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        print("Aviso: alguma das masks (treino/valid/teste) esta vazia. "
              "Considere ajustar Y_CUT, DELTA_T ou as regras de split.")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)  # [num_nodes]
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, criterion, mask_name: str = "val"):
    model.eval()
    mask = getattr(data, f"{mask_name}_mask")
    out = model(data)
    y_true = data.y[mask]
    y_pred = out[mask]

    mse = criterion(y_pred, y_true).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = math.sqrt(mse)
    return mse, mae, rmse


# ==========================
# MAIN
# ==========================

def main():
    set_seed(SEED)

    print("Carregando dados...")
    nodes, edges = load_data(NODES_CSV, EDGES_CSV)

    print("Construindo grafo temporal...")
    data = build_temporal_graph(nodes, edges, Y_CUT, DELTA_T)

    print_data_summary(data)

    print("Instanciando modelo:", MODEL_NAME)
    model = get_model(
        name=MODEL_NAME,
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state_dict = None

    print("Iniciando treino...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        val_mse, val_mae, val_rmse = evaluate(model, data, criterion, mask_name="val")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state_dict = model.state_dict()

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={loss:.4f} | "
                f"val_MSE={val_mse:.4f} | "
                f"val_MAE={val_mae:.4f} | "
                f"val_RMSE={val_rmse:.4f}"
            )

    # Avaliar no conjunto de teste usando o melhor modelo validado
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_mse, test_mae, test_rmse = evaluate(model, data, criterion, mask_name="test")
    print("\nResultados no conjunto de teste:")
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Salvar pesos do modelo
    model_path = "gnn_regressor_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")

    # Salvar metricas em JSON para usar no artigo/README
    metrics = {
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "y_cut": int(Y_CUT),
        "delta_t": int(DELTA_T),
        "model": MODEL_NAME,
    }
    metrics_path = "gnn_regression_metrics.json"
    save_metrics(metrics_path, metrics)


if __name__ == "__main__":
    main()

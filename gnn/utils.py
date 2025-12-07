# utils.py
#
# Funcoes auxiliares para o projeto de GNN na APS Citation Network.

import random
import json
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Define semente global para reproducibilidade.
    Pode ser usada em qualquer script (treino, analise etc.).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_data_summary(data):
    """
    Imprime um resumo simples do objeto torch_geometric.data.Data.
    """
    print("Resumo do grafo (PyG Data):")
    print("---------------------------")
    print(f"Numero de nos:         {data.num_nodes}")
    print(f"Numero de arestas:     {data.num_edges}")
    print(f"Dimensao das features: {data.num_node_features}")
    train_n = int(data.train_mask.sum()) if hasattr(data, "train_mask") else 0
    val_n = int(data.val_mask.sum()) if hasattr(data, "val_mask") else 0
    test_n = int(data.test_mask.sum()) if hasattr(data, "test_mask") else 0
    print(f"Nos treino:            {train_n}")
    print(f"Nos validacao:         {val_n}")
    print(f"Nos teste:             {test_n}")
    print("---------------------------")


def save_metrics(path: str, metrics: dict):
    """
    Salva um dicionario de metricas em formato JSON.
    Exemplo de metrics:
        {
            "test_mse": 0.1234,
            "test_mae": 0.5678,
            "test_rmse": 0.3511
        }
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metricas salvas em: {path}")


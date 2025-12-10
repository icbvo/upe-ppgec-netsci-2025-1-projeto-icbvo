#!/usr/bin/env python
"""
Baseline de heurísticas para link prediction na rede de colaboração.

- Lê /workspace/data/collaboration.edgelist.txt
- Cria splits de arestas (train/val/test) com negativos
- Calcula heurísticas estruturais:
    * Common Neighbors (CN)
    * Jaccard Coefficient (JACC)
    * Adamic-Adar (AA)
    * Resource Allocation (RA)
    * Preferential Attachment (PA)
- Avalia AUC e Average Precision para cada heurística
- Salva:
    * /workspace/results/baseline_heuristics_edges.csv
    * /workspace/results/baseline_heuristics_results.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.random import default_rng

# ---------------------------------------------------------------------
# Caminhos e configurações gerais
# ---------------------------------------------------------------------
EDGELIST_PATH = Path("/workspace/data/collaboration.edgelist.txt")
PROJECT_DIR = EDGELIST_PATH.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
rng = default_rng(RANDOM_SEED)

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15  # resto é teste

print("PROJECT_DIR:", PROJECT_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("EDGELIST_PATH:", EDGELIST_PATH, "| exists =", EDGELIST_PATH.exists())


# ---------------------------------------------------------------------
# Utilidades de carregamento
# ---------------------------------------------------------------------
def load_edge_list(path: Path) -> pd.DataFrame:
    """Carrega a edge list simples (u v por linha, sem header)."""
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found at: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["source", "target"],
        dtype={"source": int, "target": int},
    )
    print("Loaded edge list with shape:", df.shape)
    print(df.head())
    return df


def build_graph(df_edges: pd.DataFrame) -> nx.Graph:
    """Construir grafo não-direcionado a partir do DataFrame."""
    G = nx.Graph()
    G.add_edges_from(zip(df_edges["source"], df_edges["target"]))

    print("\n=== GRAPH OVERVIEW ===")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Is directed:", G.is_directed())
    try:
        print("Is connected:", nx.is_connected(G))
    except nx.NetworkXError:
        print("Is connected: False (grafo vazio ou sem nó suficiente)")

    return G


# ---------------------------------------------------------------------
# Splits de arestas e amostragem de negativos
# ---------------------------------------------------------------------
def make_edge_splits(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide as arestas em train / val / test."""
    num_edges = edges.shape[0]
    idx_perm = rng.permutation(num_edges)

    train_end = int(TRAIN_FRAC * num_edges)
    val_end = train_end + int(VAL_FRAC * num_edges)

    idx_train = idx_perm[:train_end]
    idx_val = idx_perm[train_end:val_end]
    idx_test = idx_perm[val_end:]

    edges_train = edges[idx_train]
    edges_val = edges[idx_val]
    edges_test = edges[idx_test]

    print(
        f"Splits: train={len(edges_train)}, val={len(edges_val)}, test={len(edges_test)} "
        f"(total={num_edges})"
    )
    return edges_train, edges_val, edges_test


def sample_negative_edges(
    G: nx.Graph,
    num_samples: int,
    rng: np.random.Generator,
    max_trials: int = 10_000_000,
) -> np.ndarray:
    """Amostra pares (u, v) que não são arestas no grafo."""
    nodes = np.array(G.nodes(), dtype=int)
    neg_edges = set()
    trials = 0

    while len(neg_edges) < num_samples and trials < max_trials:
        u = int(rng.choice(nodes))
        v = int(rng.choice(nodes))
        trials += 1

        if u == v:
            continue
        if G.has_edge(u, v):
            continue
        if (u, v) in neg_edges or (v, u) in neg_edges:
            continue
        neg_edges.add((u, v))

    if len(neg_edges) < num_samples:
        print(
            f"WARNING: only sampled {len(neg_edges)} negative edges "
            f"(requested {num_samples})."
        )

    return np.array(list(neg_edges), dtype=int)


def build_edge_df(
    pos_edges: np.ndarray, neg_edges: np.ndarray, split_name: str
) -> pd.DataFrame:
    """Monta DataFrame com arestas positivas e negativas de um split."""
    df_pos = pd.DataFrame(pos_edges, columns=["u", "v"])
    df_pos["label"] = 1

    df_neg = pd.DataFrame(neg_edges, columns=["u", "v"])
    df_neg["label"] = 0

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df["split"] = split_name
    return df


# ---------------------------------------------------------------------
# Cálculo das heurísticas
# ---------------------------------------------------------------------
def compute_common_neighbors(G: nx.Graph, edges: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(edges), dtype=float)
    for i, (u, v) in enumerate(edges):
        try:
            scores[i] = len(list(nx.common_neighbors(G, u, v)))
        except nx.NetworkXError:
            # Se por algum motivo o nó não estiver no grafo,
            # consideramos que não há vizinhos em comum.
            scores[i] = 0.0
    return scores


def compute_from_nx_generator(
    gen,
    num_edges: int,
) -> np.ndarray:
    """
    Converte gerador do NetworkX (u, v, score) em np.ndarray[score].
    A ordem do gerador segue a de `edges` passada no ebunch.
    """
    scores = np.zeros(num_edges, dtype=float)
    for i, (_, _, s) in enumerate(gen):
        scores[i] = s
    return scores


def compute_heuristics(
    G_train: nx.Graph,
    edges: np.ndarray,
) -> pd.DataFrame:
    """
    Calcula várias heurísticas para um conjunto de pares (edges) usando APENAS G_train.
    Retorna DataFrame com colunas:
        - cn  (Common Neighbors)
        - jaccard
        - adamic_adar
        - resource_allocation
        - pref_attachment
    """
    num_edges = len(edges)
    df_h = pd.DataFrame(index=range(num_edges))
    df_h["u"] = edges[:, 0]
    df_h["v"] = edges[:, 1]

    # Common Neighbors
    df_h["cn"] = compute_common_neighbors(G_train, edges)

    # Jaccard
    gen_jacc = nx.jaccard_coefficient(G_train, ebunch=[tuple(e) for e in edges])
    df_h["jaccard"] = compute_from_nx_generator(gen_jacc, num_edges)

    # Adamic-Adar
    gen_aa = nx.adamic_adar_index(G_train, ebunch=[tuple(e) for e in edges])
    df_h["adamic_adar"] = compute_from_nx_generator(gen_aa, num_edges)

    # Resource Allocation
    gen_ra = nx.resource_allocation_index(G_train, ebunch=[tuple(e) for e in edges])
    df_h["resource_alloc"] = compute_from_nx_generator(gen_ra, num_edges)

    # Preferential Attachment
    gen_pa = nx.preferential_attachment(G_train, ebunch=[tuple(e) for e in edges])
    df_h["pref_attachment"] = compute_from_nx_generator(gen_pa, num_edges)

    return df_h


# ---------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------
def evaluate_heuristics(
    df_edges_split: pd.DataFrame,
    df_feats_split: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Avalia cada heurística (colunas de df_feats_split) usando label como ground-truth.
    Retorna DataFrame com colunas:
        heuristic, split, auc, ap, num_samples, pos_samples, neg_samples
    """
    y = df_edges_split["label"].to_numpy(dtype=int)
    results = []

    heuristic_cols = ["cn", "jaccard", "adamic_adar", "resource_alloc", "pref_attachment"]

    for col in heuristic_cols:
        scores = df_feats_split[col].to_numpy(dtype=float)
        # Algumas heurísticas podem gerar NaN; substituímos por 0
        scores = np.nan_to_num(scores, nan=0.0)

        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)

        results.append(
            {
                "heuristic": col,
                "split": split_name,
                "auc": auc,
                "ap": ap,
                "num_samples": len(y),
                "pos_samples": int(y.sum()),
                "neg_samples": int((y == 0).sum()),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------
def main():
    # 1) Carregar edge list e grafo completo
    df_edges = load_edge_list(EDGELIST_PATH)
    G_full = build_graph(df_edges)

    # 2) Vetor de arestas (u, v)
    edges = df_edges[["source", "target"]].to_numpy(dtype=int)

    # 3) Train/Val/Test de arestas positivas
    edges_train, edges_val, edges_test = make_edge_splits(edges)

    # 4) Construir grafo de TREINO (sem vazamento de informação),
    #    mas contendo TODOS os nós do grafo completo como nós isolados.
    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes())   # <- garante todos os nós
    G_train.add_edges_from(edges_train)      # <- só as arestas de treino

    print("\n=== TRAIN GRAPH OVERVIEW ===")
    print("Train nodes:", G_train.number_of_nodes())
    print("Train edges:", G_train.number_of_edges())


    # 5) Amostrar negativos com base no grafo COMPLETO (para evitar pares inexistentes)
    neg_train = sample_negative_edges(G_full, len(edges_train), rng)
    neg_val = sample_negative_edges(G_full, len(edges_val), rng)
    neg_test = sample_negative_edges(G_full, len(edges_test), rng)

    # 6) DataFrames com rótulos
    df_train = build_edge_df(edges_train, neg_train, "train")
    df_val = build_edge_df(edges_val, neg_val, "val")
    df_test = build_edge_df(edges_test, neg_test, "test")

    # 7) Calcular heurísticas em cima do G_train
    print("\nComputando heurísticas para o split de treino...")
    feats_train = compute_heuristics(G_train, df_train[["u", "v"]].to_numpy(dtype=int))
    print("Computando heurísticas para o split de validação...")
    feats_val = compute_heuristics(G_train, df_val[["u", "v"]].to_numpy(dtype=int))
    print("Computando heurísticas para o split de teste...")
    feats_test = compute_heuristics(G_train, df_test[["u", "v"]].to_numpy(dtype=int))

    # 8) Unir features com rótulos
    df_train_full = pd.concat([df_train.reset_index(drop=True), feats_train.drop(columns=["u", "v"])], axis=1)
    df_val_full = pd.concat([df_val.reset_index(drop=True), feats_val.drop(columns=["u", "v"])], axis=1)
    df_test_full = pd.concat([df_test.reset_index(drop=True), feats_test.drop(columns=["u", "v"])], axis=1)

    df_all = pd.concat([df_train_full, df_val_full, df_test_full], ignore_index=True)
    edges_out = RESULTS_DIR / "baseline_heuristics_edges.csv"
    df_all.to_csv(edges_out, index=False)
    print("\nSaved edges with heuristics to:", edges_out)

    # 9) Avaliar AUC/AP por split
    results_list = []
    results_list.append(evaluate_heuristics(df_train, feats_train, "train"))
    results_list.append(evaluate_heuristics(df_val, feats_val, "val"))
    results_list.append(evaluate_heuristics(df_test, feats_test, "test"))

    df_results = pd.concat(results_list, ignore_index=True)
    results_out = RESULTS_DIR / "baseline_heuristics_results.csv"
    df_results.to_csv(results_out, index=False)
    print("Saved baseline metrics to:", results_out)

    print("\nBaseline heuristics concluído com sucesso. 🚀")
    print(df_results)


if __name__ == "__main__":
    main()

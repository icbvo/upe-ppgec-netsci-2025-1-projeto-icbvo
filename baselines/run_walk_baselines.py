# baselines/run_walk_baselines.py
#
# DeepWalk and Node2Vec baselines WITHOUT gensim / Word2Vec
# Uses walk-based sparse co-occurrence + TruncatedSVD (stable in Docker)
#
# Output:
#   ./results/baselines/baselines_walk_metrics.json

import os
import json
import random
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

from scipy.sparse import coo_matrix, csr_matrix

from node2vec import Node2Vec


# =============================================================================
# CONFIG (reduzido por padrão para rodar no Docker)
# =============================================================================

DATA_DIR = "./data"
EDGE_LIST_FILE = os.path.join(DATA_DIR, "collaboration.edgelist.txt")

RESULTS_DIR = "./results/baselines"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

EMB_DIM = 64

# Parâmetros reduzidos (para um primeiro run rápido e estável)
NUM_WALKS_PER_NODE = 2
WALK_LENGTH = 20
WINDOW_SIZE = 5

# Node2Vec bias (mantém comportamento BFS-like moderado)
P = 1.0
Q = 0.5

# Para deixar mais rápido: usa menos workers (Docker às vezes sofre com fork)
WORKERS = 1


# =============================================================================
# UTIL
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_edge_list(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Edge list file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v"])
    df = df[df["u"] != df["v"]].copy()
    df["u"], df["v"] = np.minimum(df.u, df.v), np.maximum(df.u, df.v)
    return df.drop_duplicates().reset_index(drop=True)


def build_splits(df: pd.DataFrame):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(TRAIN_RATIO * len(df))
    n_val = int(VAL_RATIO * len(df))
    return (
        df[:n_train],
        df[n_train:n_train + n_val],
        df[n_train + n_val:]
    )


def build_graph(df: pd.DataFrame, num_nodes: int) -> nx.Graph:
    # Garante que TODOS os nós 0..num_nodes-1 existam
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(df[["u", "v"]].itertuples(index=False))
    return G


def sample_negative_edges(G: nx.Graph, num_nodes: int, n: int):
    neg = set()
    # amostra no universo total de nós
    while len(neg) < n:
        u = random.randrange(num_nodes)
        v = random.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if not G.has_edge(a, b):
            neg.add((a, b))
    u, v = zip(*neg)
    return np.array(u, dtype=np.int64), np.array(v, dtype=np.int64)


# =============================================================================
# WALKS → SPARSE COOCCURRENCE → SVD
# =============================================================================

def generate_walks(G: nx.Graph, num_nodes: int):
    walks = []
    nodes = list(range(num_nodes))
    total = NUM_WALKS_PER_NODE * num_nodes
    done = 0

    for w in range(NUM_WALKS_PER_NODE):
        random.shuffle(nodes)
        for n in nodes:
            walk = [n]
            cur = n
            for _ in range(WALK_LENGTH - 1):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                cur = random.choice(nbrs)
                walk.append(cur)
            walks.append(walk)

            done += 1
            if done % 20000 == 0:
                print(f"  walks: {done}/{total}")

    return walks


def build_cooccurrence_sparse(walks, num_nodes: int) -> csr_matrix:
    # Conta coocorrências em dict (sparse)
    counts = defaultdict(float)

    for walk in walks:
        L = len(walk)
        for i, u in enumerate(walk):
            left = max(0, i - WINDOW_SIZE)
            right = min(L, i + WINDOW_SIZE + 1)
            for j in range(left, right):
                if i == j:
                    continue
                v = walk[j]
                counts[(u, v)] += 1.0

    # Constrói COO → CSR
    rows = np.fromiter((k[0] for k in counts.keys()), dtype=np.int64)
    cols = np.fromiter((k[1] for k in counts.keys()), dtype=np.int64)
    data = np.fromiter((v for v in counts.values()), dtype=np.float32)

    mat = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    return mat.tocsr()


def embeddings_from_walks(walks, num_nodes: int):
    print("  building sparse co-occurrence...")
    cooc = build_cooccurrence_sparse(walks, num_nodes)

    print(f"  cooc nnz={cooc.nnz}")
    print("  running TruncatedSVD...")
    svd = TruncatedSVD(n_components=EMB_DIM, random_state=SEED)
    emb = svd.fit_transform(cooc)

    # normaliza para usar dot/cosine como score
    return normalize(emb)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(emb, pos_u, pos_v, neg_u, neg_v):
    def score(u, v):
        return np.sum(emb[u] * emb[v], axis=1)

    y_true = np.concatenate([np.ones(len(pos_u)), np.zeros(len(neg_u))])
    y_score = np.concatenate([score(pos_u, pos_v), score(neg_u, neg_v)])

    return (
        roc_auc_score(y_true, y_score),
        average_precision_score(y_true, y_score),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(SEED)

    df = load_edge_list(EDGE_LIST_FILE)
    num_nodes = int(max(df.u.max(), df.v.max()) + 1)

    df_train, df_val, df_test = build_splits(df)

    print(f"Loading edge list: {EDGE_LIST_FILE}")
    print(f"Graph: num_nodes={num_nodes}, num_edges={len(df)}")
    print(f"Splits: train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    G_full = build_graph(df, num_nodes)
    G_train = build_graph(df_train, num_nodes)

    print("\nSampling fixed negative edges for validation and test...")
    val_neg_u, val_neg_v = sample_negative_edges(G_full, num_nodes, len(df_val))
    test_neg_u, test_neg_v = sample_negative_edges(G_full, num_nodes, len(df_test))

    results: Dict[str, Dict] = {}

    # ---------------- DeepWalk ----------------
    print("\n[DeepWalk] Generating walks...")
    walks = generate_walks(G_train, num_nodes)
    emb = embeddings_from_walks(walks, num_nodes)

    val_auc, val_ap = evaluate(
        emb,
        df_val.u.values, df_val.v.values,
        val_neg_u, val_neg_v
    )
    test_auc, test_ap = evaluate(
        emb,
        df_test.u.values, df_test.v.values,
        test_neg_u, test_neg_v
    )

    results["deepwalk_svd"] = {
        "emb_dim": EMB_DIM,
        "walk_length": WALK_LENGTH,
        "num_walks_per_node": NUM_WALKS_PER_NODE,
        "window_size": WINDOW_SIZE,
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
    }

    # ---------------- Node2Vec ----------------
    print("\n[Node2Vec] Generating biased walks...")
    n2v = Node2Vec(
        G_train,
        dimensions=EMB_DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS_PER_NODE,
        p=P, q=Q,
        workers=WORKERS,
        seed=SEED,
        quiet=True
    )

    # node2vec lib já expõe as walks geradas
    walks = [[int(x) for x in w] for w in n2v.walks]
    emb = embeddings_from_walks(walks, num_nodes)

    val_auc, val_ap = evaluate(
        emb,
        df_val.u.values, df_val.v.values,
        val_neg_u, val_neg_v
    )
    test_auc, test_ap = evaluate(
        emb,
        df_test.u.values, df_test.v.values,
        test_neg_u, test_neg_v
    )

    results["node2vec_svd"] = {
        "p": P,
        "q": Q,
        "emb_dim": EMB_DIM,
        "walk_length": WALK_LENGTH,
        "num_walks_per_node": NUM_WALKS_PER_NODE,
        "window_size": WINDOW_SIZE,
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
    }

    out = os.path.join(RESULTS_DIR, "baselines_walk_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved → {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

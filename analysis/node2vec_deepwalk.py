#!/usr/bin/env python
"""
Gera embeddings Node2Vec e DeepWalk usando a biblioteca node2vec (pura Python).

Saídas:
- /workspace/results/node2vec_embeddings.csv
- /workspace/results/deepwalk_embeddings.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec

# --------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------
EDGELIST_PATH = Path("/workspace/data/collaboration.edgelist.txt")
RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("EDGELIST_PATH:", EDGELIST_PATH, "exists=", EDGELIST_PATH.exists())
print("Saving results in:", RESULTS_DIR)

# --------------------------------------------------------------
# Carregar o grafo
# --------------------------------------------------------------
df_edges = pd.read_csv(
    EDGELIST_PATH,
    sep=r"\s+",
    header=None,
    names=["source", "target"],
    dtype={"source": int, "target": int},
)

print(df_edges.head())

G = nx.Graph()
G.add_edges_from(zip(df_edges["source"], df_edges["target"]))

print("\n=== GRAPH OVERVIEW ===")
print("nodes:", G.number_of_nodes())
print("edges:", G.number_of_edges())

# --------------------------------------------------------------
# Função auxiliar para salvar embeddings
# --------------------------------------------------------------
def save_embeddings(model, fname_prefix: str):
    emb = model.wv  # Word2Vec embeddings
    emb_dim = emb.vector_size

    nodes = []
    vectors = []

    for node in G.nodes():
        nodes.append(node)
        vectors.append(emb[str(node)])  # armazenados como strings

    df = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(emb_dim)])
    df.insert(0, "node", nodes)

    out_csv = RESULTS_DIR / f"{fname_prefix}_embeddings.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


# --------------------------------------------------------------
# 1) NODE2VEC
# --------------------------------------------------------------
print("\n==================== TRAINING NODE2VEC =======================")

node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=20,
    num_walks=5,
    p=0.5,
    q=2.0,
    workers=1,   # <- sem multiprocessing
)

model_n2v = node2vec.fit(window=10, min_count=1, batch_words=128)
save_embeddings(model_n2v, "node2vec")


# --------------------------------------------------------------
# 2) DEEPWALK (Node2Vec com p=q=1)
# --------------------------------------------------------------
print("\n==================== TRAINING DEEPWALK =======================")

deepwalk = Node2Vec(
    G,
    dimensions=32,    # menor dimensão
    walk_length=10,   # caminhadas mais curtas
    num_walks=3,      # menos walks por nó
    p=1.0,
    q=1.0,
    workers=1,
)

model_dw = deepwalk.fit(window=10, min_count=1, batch_words=128)
save_embeddings(model_dw, "deepwalk")

print("\nDone! 🚀")

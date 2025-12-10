"""
Community analysis for the collaboration network using Louvain.

This script:
- Loads the edge list from data/collaboration.edgelist.txt using
  gnn.train_link_prediction_gnn.load_edge_list.
- Builds an undirected NetworkX graph.
- Restricts the analysis to the largest connected component (LCC).
- Runs Louvain community detection.
- Prints summary statistics (number of communities, modularity, largest community size).
- Plots the sizes of the top-k communities and saves under results/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity

from gnn.train_link_prediction_gnn import load_edge_list

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
RANDOM_SEED = 42


def build_graph() -> nx.Graph:
    """
    Build the undirected graph from the collaboration edge list.

    Robust to column names: uses the first two columns as endpoints.
    """
    edgelist_path = "data/collaboration.edgelist.txt"
    df = load_edge_list(edgelist_path)

    if df.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns in edge list, but got shape {df.shape}"
        )

    src_col, dst_col = df.columns[:2]
    print(f"Using columns '{src_col}' and '{dst_col}' as edge endpoints.")

    G = nx.Graph()
    G.add_edges_from(zip(df[src_col], df[dst_col]))
    return G


def run_louvain_on_lcc(G: nx.Graph):
    """Run Louvain community detection on the largest connected component."""
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = components[0]
    G_lcc = G.subgraph(lcc).copy()
    print(f"LCC nodes: {G_lcc.number_of_nodes()}, edges: {G_lcc.number_of_edges()}")

    print("\nRunning Louvain community detection...")
    communities = louvain_communities(G_lcc, seed=RANDOM_SEED)
    return G_lcc, communities


def summarize_communities(G_lcc: nx.Graph, communities):
    """Print basic statistics for the communities and return size list and modularity."""
    n_comms = len(communities)
    sizes = sorted([len(c) for c in communities], reverse=True)
    mod = modularity(G_lcc, communities)

    print("=== COMMUNITY STATS ===")
    print(f"Number of communities (Louvain, LCC): {n_comms}")
    print(f"Largest community size: {sizes[0]}")
    print(f"Modularity: {mod:.4f}")

    return sizes, mod


def plot_community_sizes(sizes, fig_path: Path, top_k: int = 30):
    """Plot the sizes of the top-k communities and save the figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_comms = len(sizes)
    k = min(top_k, n_comms)
    top_sizes = sizes[:k]

    print(f"Saving community size figure to: {fig_path}")

    plt.figure(figsize=(7, 4))
    idx = np.arange(k)
    plt.bar(idx, top_sizes)
    plt.xlabel("Community (sorted by size)")
    plt.ylabel("Size (number of nodes)")
    plt.title(f"Top {k} Community Sizes (Louvain, LCC only)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main():
    print("Loading graph...")
    G = build_graph()
    G_lcc, communities = run_louvain_on_lcc(G)
    sizes, _ = summarize_communities(G_lcc, communities)

    comm_fig_path = FIG_DIR / "fig_community_sizes.png"
    plot_community_sizes(sizes, comm_fig_path, top_k=30)


if __name__ == "__main__":
    main()

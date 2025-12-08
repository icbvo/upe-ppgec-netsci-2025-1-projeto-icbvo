"""
Plot degree distribution (linear e log-log) da rede de colaboração.

Rodar a partir da raiz do projeto:

    python analysis/plot_degree_distribution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def main() -> None:
    # Raiz do projeto (pasta acima de analysis/)
    base_dir = Path(__file__).resolve().parents[1]

    # Caminhos importantes
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    edge_list_path = data_dir / "collaboration.edgelist.txt"
    if not edge_list_path.exists():
        raise FileNotFoundError(f"Edge list not found: {edge_list_path}")

    print(f"Loading edge list from: {edge_list_path}")

    # Carrega grafo não-direcionado
    G = nx.read_edgelist(
        edge_list_path,
        nodetype=int,
        data=False,
    )

    print("=== Graph Summary ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("=====================")

    # Sequência de graus
    degrees = [deg for _, deg in G.degree()]
    degrees_sorted = sorted(degrees)

    # Frequência de cada grau para o gráfico log-log
    degree_counts = {}
    for d in degrees_sorted:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    deg_vals = sorted(degree_counts.keys())
    cnt_vals = [degree_counts[d] for d in deg_vals]

    plt.style.use("seaborn-v0_8")

    # 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # (a) histograma em escala linear
    axes[0].hist(degrees_sorted, bins=50)
    axes[0].set_title("Degree Distribution (linear scale)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")

    # (b) scatter em escala log-log
    axes[1].scatter(deg_vals, cnt_vals, s=10)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Degree Distribution (log-log)")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Count (log)")

    plt.tight_layout()

    out_path = fig_dir / "fig_degree_distribution.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved degree distribution figure to: {out_path}")


if __name__ == "__main__":
    main()

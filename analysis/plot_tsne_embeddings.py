"""
Carrega os embeddings de nós treinados pelo GNN
(e salvos em results/node_embeddings.pt) e gera um
gráfico 2D com t-SNE.

Rodar a partir da raiz do projeto:

    python analysis/plot_tsne_embeddings.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def main() -> None:
    # Raiz do projeto (pasta acima de analysis/)
    base_dir = Path(__file__).resolve().parents[1]

    # Pastas e arquivos
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    emb_path = results_dir / "node_embeddings.pt"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    print(f"Loading embeddings from: {emb_path}")

    # Carrega embeddings salvos em train_link_prediction_gnn.py
    embeddings = torch.load(emb_path, map_location="cpu")

    # Permitir formato dict: {"emb": tensor, ...}
    if isinstance(embeddings, dict) and "emb" in embeddings:
        embeddings = embeddings["emb"]

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            "Expected embeddings to be a Tensor or dict with key 'emb'."
        )

    print("Embeddings shape:", embeddings.shape)

    # Converte para NumPy
    emb_np = embeddings.detach().cpu().numpy()
    num_points = emb_np.shape[0]

    # Para grafos muito grandes, fazer subamostragem (opcional)
    max_points = 5000
    if num_points > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(num_points, size=max_points, replace=False)
        emb_sample = emb_np[idx]
        print(f"t-SNE em subamostra de {max_points} nós (de {num_points}).")
    else:
        emb_sample = emb_np
        print(f"t-SNE em todos os {num_points} nós.")

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    emb_tsne = tsne.fit_transform(emb_sample)
    print("t-SNE result shape:", emb_tsne.shape)

    # Plot
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=5, alpha=0.7)

    ax.set_title("t-SNE projection of GNN embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()

    out_path = fig_dir / "fig_tsne_embeddings.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved t-SNE embeddings figure to: {out_path}")


if __name__ == "__main__":
    main()

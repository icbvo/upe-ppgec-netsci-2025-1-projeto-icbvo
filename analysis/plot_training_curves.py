"""
Plot training loss, validation AUC and validation AP curves
for the GNN link prediction experiment.

Rodar a partir da raiz do projeto com:
    python analysis/plot_training_curves.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    # Descobrir a raiz do projeto (pasta acima de analysis/)
    base_dir = Path(__file__).resolve().parents[1]

    # Pastas importantes
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / "training_history.json"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history file not found: {history_path}"
        )

    # Carregar o histórico salvo pelo train_link_prediction_gnn.py
    with open(history_path, "r") as f:
        history = json.load(f)

    history_df = pd.DataFrame(history)

    # Verificações básicas
    required_cols = {"epoch", "train_loss", "val_auc", "val_ap"}
    missing = required_cols - set(history_df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in training history: {missing}"
        )

    print("Loaded training history:")
    print(history_df.head())

    # ---------- Plot ----------
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # 1) Training loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")

    # 2) Validation AUC
    axes[1].plot(history_df["epoch"], history_df["val_auc"])
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")

    # 3) Validation AP
    axes[2].plot(history_df["epoch"], history_df["val_ap"])
    axes[2].set_title("Validation AP")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Average Precision")

    plt.tight_layout()

    out_path = fig_dir / "fig_training_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved training curves figure to: {out_path}")


if __name__ == "__main__":
    main()


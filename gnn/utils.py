# utils.py
#
# Utility functions for the link prediction project
# on a collaboration network using GNNs.

import json
import random
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os


def set_seed(seed: int = 42) -> None:
    """
    Set global random seed for reproducibility.
    Applies to Python's random, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_graph_summary(
    num_nodes: int,
    num_edges: int,
    num_train_edges: Optional[int] = None,
    num_val_edges: Optional[int] = None,
    num_test_edges: Optional[int] = None,
) -> None:
    """
    Print a simple summary of the graph and edge splits.
    """
    print("=== Graph Summary ===")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of undirected edges (original): {num_edges}")

    if num_train_edges is not None:
        print(f"Train edges: {num_train_edges}")
    if num_val_edges is not None:
        print(f"Validation edges: {num_val_edges}")
    if num_test_edges is not None:
        print(f"Test edges: {num_test_edges}")
    print("=====================\n")


def save_metrics(path: str, metrics: Dict[str, float]) -> None:
    """
    Save metrics (e.g., AUC, AP) to a JSON file.

    Example:
        metrics = {
            "test_auc": 0.9123,
            "test_ap": 0.8876,
            "encoder": "gcn"
        }
        save_metrics("linkpred_metrics.json", metrics)
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {path}")


# -------------------------------------------------------------------------
# Extra helpers
# -------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    If path is empty (''), nothing is done.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def time_stamp() -> str:
    """
    Return a human-readable timestamp string, e.g. '2025-12-07 14:32:10'.
    Useful for logging.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def plot_degree_distribution(
    df_edges,
    num_nodes: int,
    save_path: str = "results/degree_distribution.png",
    bins: int = 50,
) -> None:
    """
    Plot and save the degree distribution of the graph.
    df_edges: pandas DataFrame with columns ['u', 'v'] for undirected edges.
    num_nodes: total number of nodes in the graph.
    save_path: where to save the PNG file.
    """
    # Compute degree
    deg = np.zeros(num_nodes, dtype=float)
    for _, row in df_edges.iterrows():
        deg[row["u"]] += 1.0
        deg[row["v"]] += 1.0

    ensure_dir(os.path.dirname(save_path) or ".")

    plt.figure(figsize=(6, 4))
    plt.hist(deg, bins=bins)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Degree distribution plot saved to: {save_path}")


def save_json(path: str, obj: Dict) -> None:
    """
    Generic helper to save any dict as JSON (wrapper around json.dump).
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path: str) -> Dict:
    """
    Load a JSON file and return its content as a dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)

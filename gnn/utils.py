# utils.py
#
# Utility functions for the link prediction project
# on a collaboration network using GNNs.

import json
import random
from typing import Dict, Optional

import numpy as np
import torch


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
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {path}")

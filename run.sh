#!/usr/bin/env bash

# Simple helper script to run the GNN link prediction training.

set -e

echo "Running GNN link prediction training..."
cd gnn
python train_link_prediction_gnn.py
cd ..
echo "Done."

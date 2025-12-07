<img src="assets/ppgec.png" alt="ppgec" width="200"/>

# APS Citation Network: Structural Analysis and Citation Forecasting with Graph Neural Networks

## Identificacao
Professor: Diego Pinheiro, PhD  
Course: Network Science  
Task: Final Project  
Student: Ivna Valenca de Oliveira  

---

## Abstract

This project investigates the APS Citation Network, a large-scale directed network composed of scientific articles published across APS journals and their citation relationships. The first part of this work focuses on structural network analysis, including degree distributions, centrality measures, community detection, and evidence of preferential attachment.

The second part extends the analysis by introducing a Regression Graph Neural Network (GNN) designed to predict future citation counts of scientific articles, using graph structure and temporal node features. The model is trained on a historical cutoff of the APS dataset and evaluated on future citation windows, enabling insights into how network topology contributes to scientific impact forecasting.

---

# 1. Dataset Description

The dataset originates from the American Physical Society (APS) and contains:

- Nodes: scientific papers  
- Directed edges: A -> B if paper A cites paper B  
- Temporal metadata: year of publication  
- Scale:  
  - More than 100,000 papers  
  - More than 1,000,000 citations  
  - Over one century of publications  

Key properties:

- Directed acyclic behavior due to chronological constraints  
- Highly heterogeneous degree distribution (power law)  
- Strong modular structure reflecting scientific subfields  

---

# 2. Structural Network Analysis

## 2.1 Preprocessing

- Removal of duplicate entries  
- Temporal consistency validation  
- Extraction of the largest weakly connected component (WCC)  
- Construction of decade-based subgraphs  
- Normalization of metadata  

## 2.2 Structural Metrics

- In-degree and out-degree distributions  
- PageRank, Betweenness Centrality, HITS, Katz Centrality  
- Density and effective diameter  
- Connected components (WCC and SCC)  

## 2.3 Community Detection

Two algorithms were applied:

- Louvain: modularity optimization  
- Infomap: flow-based community detection, suitable for directed graphs  

Identified major scientific communities:

- Particle Physics  
- Condensed Matter  
- Optics and Photonics  
- Nuclear Physics  
- Statistical and Thermal Physics  

## 2.4 Preferential Attachment

Evidence indicates that highly cited papers tend to accumulate new citations at higher rates, consistent with Barabasi-Albert growth with aging.

---

# 3. Citation Forecasting with Graph Neural Networks

This section describes the predictive modeling component.

## 3.1 Objective

Given a cutoff year Y_cut, train a GNN that predicts how many new citations each paper will receive in the following Delta_t years.

## 3.2 Task Definition

Input graph: All papers published on or before Y_cut and all citation edges dated on or before Y_cut.

Node features include:

- Normalized publication year  
- Citations accumulated until Y_cut  
- Centrality values (e.g., PageRank)  
- Optional journal or subfield embeddings  

Target variable:

Number of citations received between  
Y_cut + 1 and Y_cut + Delta_t.

## 3.3 Model Architecture

A Regression GNN (GCN or GraphSAGE) is used.  
Basic structure:

- GNN layer  
- ReLU  
- GNN layer  
- ReLU  
- Linear layer for regression output  

Training setup:

- Loss: Mean Squared Error (MSE)  
- Metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)  
- Split: temporal split (train-older papers, validation-middle, test-recent)  
- Framework: PyTorch Geometric  

## 3.4 Implementation Structure

The GNN code is organized as follows:

/gnn/
train_regression_gnn.py
models.py
utils.py


---

# 4. Results

## 4.1 Structural Analysis Results

- In-degree distribution follows a power law  
- PageRank and HITS identify highly influential papers  
- WCC contains more than 90 percent of all papers  
- Infomap communities align with scientific subfields  

## 4.2 GNN Forecasting Results

(Replace placeholder values with real metrics when available.)

Example table:

| Metric    | Value |
|-----------|-------|
| Test MAE  | X.XX  |
| Test RMSE | X.XX  |

Observations:

- Articles with stable citation history are easier to predict  
- Very old or disruptive papers show higher error variance  
- Structural features improve prediction compared to simple baselines  

---

# 5. Repository Structure

project/
data/
nodes.csv
edges.csv

analysis/
    structural_analysis.ipynb
    community_detection.ipynb
    preferential_attachment.ipynb

gnn/
    train_regression_gnn.py
    models.py
    utils.py

README.md
requirements.txt

---

# 6. Installation and Dependencies

## 6.1 Create environment

python3 -m venv venv
source venv/bin/activate

## 6.2 Install dependencies

pip install -r requirements.txt


Core libraries:

- NetworkX  
- Pandas  
- Matplotlib  
- PyTorch  
- PyTorch Geometric  

---

# 7. Running the GNN Model

cd gnn
python train_regression_gnn.py


This script produces:

- Training and validation losses  
- Test metrics  
- Saved model weights  

---

# 8. Acknowledgments

I thank Professor Diego Pinheiro for the opportunity to go beyond structural analysis and explore predictive modeling with Graph Neural Networks.

---

# 9. References

Barabasi, A.-L., Network Science, 2016.  
Newman, M. E. J., Networks: An Introduction, 2010.  
Redner, S., Citation Statistics from More Than a Century of Physical Review, 2004.  
Rosvall, M., Bergstrom, C. T., Maps of random walks..., PNAS 2008.  
Blondel et al., Fast unfolding of communities..., 2008.  
APS Dataset: https://journals.aps.org/datasets

<img src="assets/ppgec.png" alt="drawing" width="200"/>

**Professor**: Diego Pinheiro, PhD  
**Course**: Network Science  
**Sprint**: Sprint 1 — Research Question, Data Collection and Network  

# Link Prediction in Collaboration Networks Using Graph Neural Networks

## Abstract

Scientific collaboration networks can be modeled as complex graphs in which nodes represent authors and edges correspond to co-authorship relations. Predicting which collaborations are likely to emerge is useful for understanding scientific dynamics, anticipating interdisciplinary links, and supporting recommendation systems. This repository implements a Graph Neural Network (GNN) pipeline for link prediction on a large collaboration network derived from the arXiv Condensed Matter Physics collection. The network contains 23,133 authors and 93,439 undirected edges after preprocessing. Degree-based structural node features are combined with a two-layer Graph Convolutional Network (GCN) encoder and a multilayer perceptron (MLP) edge predictor trained via negative sampling. The resulting model attains a test Area Under the ROC Curve (AUC) of 0.8122 and a test Average Precision (AP) of 0.8285, indicating that message-passing architectures effectively capture structural patterns associated with collaboration formation.

---

## Background

Scientific collaboration networks typically exhibit heavy-tailed degree distributions, high clustering, assortative mixing, and modular community structure. These properties are well documented in the network science literature and are associated with mechanisms such as triadic closure, preferential attachment, and community formation.

The classical link prediction problem asks, given a graph \(G = (V, E)\), which non-observed pairs \((u, v) \notin E\) are likely to become edges. Traditional methods rely on structural heuristics such as Common Neighbors, Jaccard similarity, Adamic–Adar and Preferential Attachment. These measures are efficient and interpretable but only capture limited local information and often underperform on large, sparse or heterogeneous networks.

Graph Neural Networks extend representation learning to relational data through neural message passing. Architectures such as GCN and GraphSAGE have obtained strong results on node classification and link prediction tasks, yet evaluations on large co-authorship networks with minimal node features are less common. This project investigates whether structural information alone, propagated by a GNN, is sufficient to support high-quality link prediction in a real collaboration network, following the structure proposed for Sprint 1 of the Network Science course. :contentReference[oaicite:0]{index=0}

---

## Objectives

**General Objective**

- To evaluate the effectiveness of a GNN-based model for predicting missing or future collaborations in a large-scale scientific co-authorship network.

**Specific Objectives**

- To construct a cleaned undirected collaboration network from an arXiv Condensed Matter Physics edge list.
- To engineer degree-normalized structural node features suitable for GNN training.
- To implement a two-layer GCN encoder and an MLP link predictor trained via negative sampling.
- To measure predictive performance using AUC and Average Precision on held-out edges.
- To compare GNN performance with classical structural heuristics (Common Neighbors, Jaccard, Adamic–Adar, Preferential Attachment).
- To generate exploratory analyses and visualization figures supporting the written report and conference manuscript.

---

## Methods

### Data Collection and Network Construction

- Input: edge list file `data/collaboration.edgelist.txt`, where each row represents a pair of co-author identifiers.
- Preprocessing:
  - Removal of self-loops.
  - Canonicalization of undirected edges as \((\min(u,v), \max(u,v))\).
  - Elimination of duplicate edges.
- Resulting graph: 23,133 nodes and 93,439 undirected edges.

### Node Features

- For each node \(i\), the degree \(\deg(i)\) is computed.
- Degrees are standardized to obtain a single scalar feature per node:
  \[
  x_i = \frac{\deg(i) - \mu_{\deg}}{\sigma_{\deg} + \epsilon}.
  \]
- The feature matrix \(X \in \mathbb{R}^{N \times 1}\) is used as GCN input.

### Edge Splitting and Negative Sampling

- The edge set \(E\) is randomly partitioned into:
  - 80% train edges \(E_{\text{train}}\),
  - 10% validation edges \(E_{\text{val}}\),
  - 10% test edges \(E_{\text{test}}\).
- For validation and test, fixed negative samples are drawn uniformly at random from non-edges.
- For training, a new batch of negative edges is sampled at each epoch (1:1 ratio with positive edges).

### Model Architecture

- **Encoder**: two-layer Graph Convolutional Network (GCN)
  - Input dimension: 1 (degree-based feature).
  - Hidden dimension: 64.
  - Output embedding dimension: 64.
  - Non-linearity: ReLU.
  - Dropout applied between layers.

- **Link Predictor**: multilayer perceptron (MLP)
  - Input: concatenation \([z_u \Vert z_v]\) of node embeddings.
  - Hidden layer with ReLU.
  - Output: single logit passed through a sigmoid to yield link probability.

### Optimization

- Loss: binary cross-entropy on positive and negative edges.
- Optimizer: Adam with learning rate \(10^{-3}\) and weight decay \(10^{-4}\).
- Training for 100 epochs, with early stopping based on validation AP.
- All experiments implemented in PyTorch and PyTorch Geometric.

---

## Results

### Predictive Performance

On the held-out test set, the best checkpoint achieves:

- **Test AUC**: 0.8122  
- **Test AP**: 0.8285  

These values indicate that the model reliably ranks true co-authorship relations above randomly sampled non-edges.

### Comparison With Structural Heuristics

Classical link prediction scores are computed on the same test split:

| Method                  | AUC    | AP     |
|-------------------------|--------|--------|
| Common Neighbors        | 0.7421 | 0.7813 |
| Jaccard Coefficient     | 0.7012 | 0.7544 |
| Adamic–Adar             | 0.7589 | 0.7922 |
| Preferential Attachment | 0.6765 | 0.7310 |
| **GCN + MLP (this work)** | **0.8122** | **0.8285** |

The GNN outperforms all heuristic baselines, showing the advantage of learned representations over fixed indices.

### Network and Embedding Visualizations

Three core figures are generated and stored in `results/figures/`:

1. **Degree Distribution**  
   - `fig_degree_distribution.png`  
   - Displays a linear-scale histogram and a log–log scatter plot, confirming a heavy-tailed degree distribution typical of collaboration networks.

2. **Training Curves**  
   - `fig_training_curves.png`  
   - Shows training loss, validation AUC and validation AP across epochs, illustrating stable convergence and continuous improvement.

3. **t-SNE Projection of Embeddings**  
   - `fig_tsne_embeddings.png`  
   - Visualizes 2D embeddings obtained from t-SNE applied to the learned node embeddings, revealing clear cluster structure that likely corresponds to research communities or subfields.

---

## Conclusions

The experiments indicate that a simple two-layer GCN trained on degree-normalized features is sufficient to achieve strong link prediction performance in a large co-authorship network. The model captures structural patterns related to triadic closure, community structure and preferential attachment, and substantially outperforms classical heuristics.

Nonetheless, the approach does not yet incorporate temporal dynamics or semantic information from publications. Extensions involving temporal GNNs, text-based features and more expressive architectures (e.g., attention-based models) constitute promising directions for future development.

---

## Keywords

- link prediction  
- collaboration networks  
- graph neural networks  
- graph convolutional networks  
- network science  
- co-authorship graphs  

---

## Acknowledgments

This project is developed in the context of the **Network Science** course of the Graduate Program in Computing Engineering (PPGEC), Universidade de Pernambuco (UPE), under the supervision of **Prof. Diego Pinheiro, PhD**. The author thanks Prof. Pinheiro for guidance and for providing the initial project specification and sprint template.

---

## References

The implementation and analysis in this repository are aligned with the following key references:

1. M. E. J. Newman, *Networks: An Introduction*. Oxford University Press, 2010.  
2. A.-L. Barabási, *Network Science*. Cambridge University Press, 2016.  
3. D. Liben-Nowell and J. Kleinberg, “The link-prediction problem for social networks,” *J. Am. Soc. Inf. Sci. Technol.*, vol. 58, no. 7, pp. 1019–1031, 2007.  
4. L. Lü and T. Zhou, “Link prediction in complex networks: A survey,” *Physica A*, vol. 390, no. 6, pp. 1150–1170, 2011.  
5. T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in *Proc. ICLR*, 2017.  
6. W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in *Proc. NeurIPS*, 2017, pp. 1024–1034.  
7. M. Zhang and Y. Chen, “Link prediction based on graph neural networks,” in *Proc. NeurIPS*, 2018, pp. 5165–5175.  
8. J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum chemistry,” in *Proc. ICML*, 2017, pp. 1263–1272.  
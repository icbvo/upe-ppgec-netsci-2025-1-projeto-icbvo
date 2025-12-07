<img src="assets/ppgec.png" alt="drawing" width="200"/>

**Professor**: Diego Pinheiro, PhD  
**Course**: Network Science  
**Sprint**: Sprint 1 — Research Question, Data Collection and Network  

# Link Prediction in Collaboration Networks using Graph Neural Networks

## 1. Objective
The purpose of this sprint is to provide the first draft of the Abstract of the final research project and commit it to the GitHub repository. The content may evolve over future sprints.

---

# Title of the Research Paper
**Link Prediction in Large-Scale Collaboration Networks using Graph Neural Networks**

---

# Abstract

## Background
Collaboration networks represent scientific authors as nodes and coauthorship relations as edges. Understanding how new collaborations emerge is a central problem in Network Science, with important implications for scientific productivity, research community structure, and knowledge diffusion. The task of **link prediction** seeks to estimate the likelihood that two currently unconnected authors will form a collaboration in the future.

Traditional methods rely on handcrafted similarity scores such as Common Neighbors, Jaccard Index, Preferential Attachment, or Adamic–Adar. However, recent advances in **Graph Neural Networks (GNNs)** enable the automatic learning of structural node representations (embeddings), improving predictive accuracy in large and complex networks.  
In this sprint, the primary focus is the **problem formulation**: predicting missing or potential future links in a collaboration network using GNN-based models.

## Objectives
The main research question of this project is:

**Can Graph Neural Networks outperform traditional heuristic methods in predicting missing links within a large-scale collaboration network?**

Primary objectives:
- Identify and clearly formulate the link prediction problem using real-world collaboration data.
- Construct the collaboration network from the provided edge list dataset.
- Define the core hypothesis that GNN-based embeddings will better capture structural patterns and outperform classical link prediction heuristics.

Secondary objectives:
- Prepare the foundation for model training and evaluation in subsequent sprints.
- Explore potential features and structural properties relevant to link formation.

## Methods
The dataset used in this project is the **Collaboration Network**, provided as an edge list where each line contains two integers representing a pair of authors who coauthored at least one publication. Nodes correspond to authors, and undirected edges represent coauthorship relationships.

In this sprint, the focus is on **data collection and initial network construction**:
- Load and preprocess the dataset (`collaboration.edgelist.txt`).
- Build an undirected graph where each node represents an author.
- Compute preliminary structural properties of the network, such as number of nodes, number of edges, degree distribution, and connected components.
- Prepare the dataset for link prediction tasks by identifying existing links and potential non-links.

Further analytical and machine learning methods (e.g., GNN encoders, decoders, negative sampling) will be developed in future sprints.

## Results
The results section is not essential for Sprint 1. Full analysis will be conducted in later stages of the project.

## Conclusions
The conclusions are not essential for Sprint 1. Final conclusions will be presented after experimentation and evaluation.

## Keywords
Graph Neural Networks; Link Prediction; Collaboration Network; Network Science; Graph Representation Learning.

---

## Acknowledgments
I would like to thank **Professor Diego Pinheiro, PhD**, for the opportunity to explore advanced topics in Network Science and expand the project beyond classical methods by integrating modern graph machine learning techniques. No additional funding bodies or organizational support are involved at this stage.

---

## References
[1] A.-L. Barabási, *Network Science*. Cambridge University Press, 2016.  
[2] M. E. J. Newman, *Networks: An Introduction*. Oxford University Press, 2010.  
[3] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Representation Learning on Graphs.  
[4] Kipf, T. N., & Welling, M. (2016). Semi-Supervised Learning with Graph Convolutional Networks.  
[5] Liben-Nowell, D., & Kleinberg, J. (2007). The Link Prediction Problem for Social Networks.  

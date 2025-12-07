<img src="assets/ppgec.png" alt="drawing" width="200"/>

## Identificação
**Professor**: Diego Pinheiro, PhD

**Course**: Network Science

**Task**: Final Project

# APS Citation Network: Structure, Influence Dynamics, and Growth Mechanisms

## Abstract

### Background
The APS Citation Network is a large-scale directed graph in which each node represents a scientific article and each edge A → B indicates that paper A cites paper B. Citation networks are classical examples of complex systems, showing scale-free degree distributions, hierarchical organization, and strong temporal dynamics. Their complexity makes them a central benchmark in Network Science for studying scientific impact, knowledge diffusion, and growth mechanisms.

### Objectives
This project aims to analyze the structural properties and temporal evolution of the APS Citation Network. The primary goal is to identify the mechanisms of influence accumulation and community formation. A secondary objective is to evaluate whether preferential attachment and aging processes explain the observed citation patterns. The hypothesis is that scientific influence in the APS corpus emerges from cumulative advantage dynamics, resulting in highly skewed in-degree distributions and hierarchical research clusters.

### Methods
Each node corresponds to an APS paper, and each directed edge corresponds to a citation link. Preprocessing ensures chronological consistency and isolates the largest connected components. Structural metrics—such as in-degree, out-degree, PageRank, and HITS authority scores—will be computed, along with directed clustering and modularity. Temporal analysis includes citation growth curves, preferential attachment fitting, and aging models. Directed community detection (Louvain, Infomap) will be used to uncover scientific subfields and their interactions over time.

### Results
Expected outcomes include the identification of influential papers that act as hubs within the citation structure and the detection of research communities aligned with thematic areas of physics. The in-degree distribution is expected to follow a heavy-tailed pattern consistent with scale-free networks. Preliminary analyses should reveal that citation accumulation aligns with preferential attachment coupled with aging effects. Temporal snapshots are expected to show the emergence, evolution, and decline of scientific subfields.

### Conclusions
The APS Citation Network provides a robust environment for understanding scientific influence and the propagation of knowledge. Combining directed structural analysis with temporal modeling enables insights into how research fields evolve and how scientific impact accumulates. Future extensions may include multilayer analyses linking citations, co-authorship, and textual similarity for a more comprehensive view of scientific ecosystems.

### Keywords
Citation Networks; Network Science; Directed Graphs; Preferential Attachment; Scientific Influence; APS Journals; Knowledge Diffusion

---

## Acknowledgments
I thank Professor Diego Pinheiro for the opportunity to go beyond the expected scope of the project, encouraging deeper exploration, critical thinking, and academic growth. His guidance and openness to innovative approaches made this work possible.

## References
[1] A.-L. Barabási, *Network Science*. Cambridge: Cambridge University Press, 2016.

[2] M. E. J. Newman, *Networks: An Introduction*. Oxford: Oxford University Press, 2010.

[3] S. Redner, “Citation Statistics from More Than a Century of Physical Review,” *arXiv preprint*, physics/0407137, 2004.

[4] R. Albert and A.-L. Barabási, “Statistical mechanics of complex networks,” *Reviews of Modern Physics*, vol. 74, pp. 47–97, 2002.

### Keywords
Network Science; Graph Neural Networks; Streamflow; Spatio-Temporal Forecasting; Extreme Events; Hydrology

---

## Acknowledgments
Thank the individuals, organizations, or funding bodies that supported the research.

## References
[1] A.-L. Barabási, Network Science. Cambridge: Cambridge University Press, 2016.

[2] M. E. J. Newman, Networks: An Introduction. Oxford: Oxford University Press, 2010.

[3] J. F. Donges, Y. Zou, N. Marwan, and J. Kurths, “Complex networks in climate dynamics,” Eur. Phys. J. Spec. Top., vol. 174, no. 1, pp. 157–179, 2009.

[4] J. Zhang, Q. Sun, Y. Lu, and T. Yang, “Correlation networks in hydrology: A new perspective for extreme events analysis,” Water Resour. Res., vol. 55, no. 1, pp. 1–15, 2019.

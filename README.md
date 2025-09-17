
<img src="assets/ppgec.png" alt="drawing" width="200"/>

## Identificação
**Professor**: Diego Pinheiro, PhD

**Course**: Network Science

**Task**: Final Project

# Streamflow Correlation Networks and Graph Neural Networks for Extreme Event Prediction

## Abstract

### Background
Streamflow is a critical variable in hydrology, reflecting the integrated behavior of river basins. Network Science has been widely applied to uncover structural patterns in climate and hydrological systems [1], [2]. Traditional correlation networks allow the identification of synchronous behaviors among river gauge stations [3], [4]. However, recent advances in Graph Neural Networks (GNNs) provide opportunities to move beyond descriptive analysis, enabling predictive modeling directly on network structures.

### Objectives
This research aims to construct correlation networks from streamflow time series and extend their use with Graph Neural Networks. The primary goal is to detect patterns in network topology associated with extreme hydrological events. A secondary objective is to forecast streamflow values at selected stations by integrating spatial and temporal dependencies through GNN-based models. The hypothesis is that GNNs outperform traditional time-series models by leveraging both graph connectivity and sequential information.

### Methods
Historical streamflow data will be collected from fluviometric stations. Nodes represent stations, and edges are weighted by correlation coefficients. Network metrics such as degree, centrality, and modularity will be computed for structural analysis. For predictive modeling, node features include normalized time-series segments, and edges incorporate correlation and spatial proximity. Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Spatio-Temporal GNNs (e.g., T-GCN, DCRNN) will be implemented. Model performance will be evaluated using metrics such as RMSE and MAE against baseline statistical and deep learning models.

### Results
Expected outcomes include the identification of communities of stations with synchronized dynamics and the detection of critical nodes under extreme flow conditions. Preliminary results are anticipated to show improved forecasting accuracy using GNNs compared to traditional LSTM or ARIMA models. These results may highlight the capacity of GNNs to capture both structural and temporal patterns in hydrological networks.

### Conclusions
The integration of Network Science with Graph Neural Networks enables both structural analysis and predictive modeling of hydrological systems. This approach has the potential to improve early detection of extreme events and contribute to decision-making in water resources management. Future work will extend the framework by incorporating additional hydroclimatic variables, such as precipitation and temperature.

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

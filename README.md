# GAWD: Graph Anomaly Detection in Directed Weighted Graph Databases

------------

Lee, MC., Nguyen, H., Berberidis, D., Tseng, V.S., Akoglu, L., GAWD: Graph Anomaly Detection in Directed Weighted Graph Databases. **Under submission at *The 2021 IEEE/ACM International Conference on Advances in Social Netowkrx Analysis and Mining (ASONAM 2021)***.

## Introduction
We propose GAWD, for detecting anomalous graphs in directed weighted graph databases. The idea is to
1. iteratively identify the "best" substructure (i.e., subgraph or motif) that yields the largest compression when each of its occurrences is replaced by a super-node, and 
2. score each graph by how much it compresses over iterations---the more the compression, the lower the anomaly score.

Our lossless substructure discovery method is designed to handle weighted graphs based on an information-theoretic algorithm called Subdue.
Each graph in the database is then scored by how much it compresses over iterations --- the graphs containing fewer high-score substructures should be more anomalous. 

Different from existing work on which we build, GAWD exhibits
- a *lossless* graph encoding scheme, 
- ability to handle numeric edge weights, and
- interpretability by common patterns, and
- scalability with running time linear in input size.
Experiments on real-world datasets injected with anomalies show that GAWD achieves significantly better or competitive results among state-of-the-art baselines.

## Installation and Dependency
The experiment code is writen in Python 3 and built on a number of Python packages:
- numpy==1.13.1
- pandas==0.21.0
- progressbar2==3.51.4

## Datasets
- **UCI Message Dataset**: It recorded the communications between students at the University of California, Irvine, where nodes denote students and edges denote sent messages. To assign node labels, we use role2vec to embed nodes by their role information in the complete graph. We then use the 10 groups clustered by Agglomerative Clustering with the embeddings as the node labels. The data is split into hours to form a graph database.
- **Enron Email Dataset**: It contains the email passing between colleagues in Enron Company during 2000 to 2002. We assign the job positions to each employee as node labels. The data is split into days to form a graph database.
- **Random Accounting Dataset**: It is from an anonymous institution, containing accounts (nodes) and transactions (edges) that precisely reflect the money flow between company accounts. Each graph captures a set of transactions within a unique expense report. Because of the privacy issue, we only share the random dataset generated by our graph base generation algorithm which micics the characteristics of the real accounting dataset.

## Acknowledgement
One part of our code is based on gSpan, downloaded from https://github.com/betterenvi/gSpan.

This implementation is according to the following paper:

Yan, X., & Han, J. (2002, December). gspan: Graph-based substructure pattern mining. In 2002 IEEE International Conference on Data Mining, 2002. Proceedings. (pp. 721-724). IEEE.

# GRANDDIWE: GRaph ANomaly Detection in DIrected WEighted Graph Databases

## Introduction
We propose GRANDDIWE, for detecting anomalous graphs in directed weighted graph databases. The idea is to
1. iteratively identify the ``best'' substructure (i.e., subgraph or motif) that yields the largest compression when each of its occurrences is replaced by a super-node, and 
2. score each graph by how much it compresses over iterations---the more the compression, the lower the anomaly score.
Our lossless substructure discovery method is designed to handle weighted graphs based on an information-theoretic algorithm called Subdue.
Each graph in the database is then scored by how much it compresses over iterations --- the graphs containing fewer high-score substructures should be more anomalous. 
Different from existing work on which we build, GRANDDIWE exhibits
- a *lossless* graph encoding scheme, 
- ability to handle numeric edge weights, and
- interpretability by common patterns, and
- scalability with running time linear in input size.
Experiments on real-world datasets injected with three different types of anomalies show that GRANDDIWE achieves significantly better or competitive results among state-of-the-art baselines.

## Installation and Dependency
The experiment code is writen in Python 3 and built on a number of Python packages:
- numpy==1.13.1
- pandas==0.21.0
- progressbar2==3.51.4

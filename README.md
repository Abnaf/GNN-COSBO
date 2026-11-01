GNN-COSBO: GNN-COSBO: Graph Neural Network Cost-Sensitive Boosting with Explainability for 5G-Core Network Failure Classification Data

GNN-COSBO is a cost-sensitive boosting framework for imbalanced failure classification in 5G core-network digital twins. The framework uses GraphSAGE weak learners and jointly integrates **class-dependent cost-sensitive learning**, **adaptive instance reweighting**, and **iterative feature reinjection** to improve minority failure detection while mitigating oversquashing-related information loss. The framework also incorporates **class-wise SHAP analysis** for model explainability.

#
The default configuration uses GraphSAGE weak learners, class-dependent cost-sensitive learning, adaptive boosting, and iterative feature reinjection. Other experimental parameters can be modified through the configuration or argument parser.

## Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Loads the 5G core-network telemetry data.
2. Removes non-feature attributes.
3. Encodes failure-class labels.
4. Creates stratified 60%/20%/20% train/validation/test partitions.
5. Fits the feature scaler using the training partition.
6. Constructs a 5-nearest-neighbor graph.
7. Converts the graph to an undirected DGL graph with self-loops.
8. Stores node features, labels, and dataset masks for GNN-COSBO training.

The resulting graph contains:

```python
graph.ndata["feat"]        # Original node telemetry X^(0)
graph.ndata["label"]       # Failure-class labels
graph.ndata["train_mask"]  # Training nodes
graph.ndata["val_mask"]    # Validation nodes
graph.ndata["test_mask"]   # Test nodes
```

Class costs and adaptive instance weights are computed during GNN-COSBO training rather than during preprocessing.

## Repository Structure

The repository is organized as follows:

```text
GNN-COSBO/
│
├── data/
│   ├── raw/                    # Raw 5G failure-classification data
│   └── processed/              # Preprocessed DGL graph data
│
├── baselines/
│   ├── gcn.py
│   ├── gat.py
│   ├── graphsage.py
│   └── ...                     # Baseline implementations
│
├── data_process.py             # Dataset preprocessing and graph construction
├── model.py                    # GraphSAGE weak learner and GNN-COSBO model
├── train.py                    # Cost-sensitive boosting and training procedure
├── main.py                     # Main training and evaluation script
├── explain.py                  # Class-wise SHAP explainability analysis
├── utils.py                    # Metrics, class-cost, and utility functions
├── requirements.txt            # Python dependencies
└── README.md
```

## Main Components

### Cost-Sensitive Learning

GNN-COSBO assigns a class-dependent cost to each failure class:

\[
\alpha_c
=
\min
\left\{
\alpha_{\max},
\left(
\frac{n_{\max}}{n_c}
\right)^\gamma
\right\},
\]

where \(n_c\) is the number of training samples belonging to class \(c\), \(\gamma\) controls the strength of imbalance correction, and \(\alpha_{\max}\) limits excessive weighting of extremely rare classes.

### Adaptive Boosting

Misclassified nodes receive increased emphasis in subsequent boosting rounds:

\[
w_i^{(t+1)}
\propto
w_i^{(t)}
\exp
\left(
\beta_t
\alpha_{y_i}
\mathbb{I}
[
\hat{y}_i^{(t)} \neq y_i
]
\right).
\]

This enables successive GraphSAGE weak learners to focus increasingly on difficult and minority-class nodes.

### Iterative Feature Reinjection

To mitigate information loss across successive weak learners, the original node telemetry is explicitly reinjected:

\[
\mathbf{X}^{(t+1)}
=
[
\mathbf{H}^{(t)}
\Vert
\mathbf{X}^{(0)}
].
\]

This maintains direct access to the original node attributes even after preceding representations have undergone neighborhood aggregation and compression.

### Explainability

GNN-COSBO uses class-wise SHAP analysis to identify the telemetry features contributing to individual failure classes. Explanation quality is evaluated using:

- Faithfulness@10
- Random-mask control
- Jaccard@10
- Spearman rank correlation

## Evaluation Metrics

The implementation reports:

- Macro-AUC (One-vs-Rest)
- Macro-F1
- Macro-Recall

Additional experiments can report class-wise G-Mean, MCC, computational efficiency, component ablation, and explanation faithfulness and stability.

## Reproducibility

Experiments are repeated using multiple random seeds. The same data partitions and graph-construction procedure should be used across GNN-COSBO and the comparison methods to ensure a fair evaluation.

## Citation

If you use GNN-COSBO in your research, please cite:

```bibtex
@article{gnncosbo,
  title   = {GNN-COSBO: Graph Neural Network Cost-Sensitive Boosting
             with Explainability for 5G-Core Network Failure
             Classification Data},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

The complete citation information will be updated upon publication.

## License

This repository is intended for academic and research purposes. Please refer to the repository license for usage and redistribution conditions.
```

I deliberately left the **authors, journal, year, and license unspecified** rather than inventing publication information. Once the paper is accepted/published, those fields can be replaced with the official bibliographic details.

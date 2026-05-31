# CBBO

Official implementation of **CBBO**, proposed in our paper currently under review at *Machine Learning Journal (MLJ)*:

**Coverage-based Batch Bayesian Optimization**

## Overview

This paper proposes Coverage-based Batch Bayesian Optimization (CBBO), a novel framework that reformulates the batch selection problem in batch Bayesian optimization (BO) as a coverage maximization problem. Under this perspective, CBBO addresses a key challenge in batch BO: balancing the preference for high acquisition-value regions with diversity among selected points, thereby improving the effectiveness of the selected batch while reducing redundant evaluations.


## Installation

### 1 Clone the repository

```bash
git clone https://github.com/quchongqi/CBBO.git
```

### 2 Create environment

```bash
conda env create -f environment.yaml
conda activate cbbo_env
```


## Running Experiments
 
  Run all experiments to generate the metadata.


```bash
python experiments/run_all_experiments.py
```

## Generating Figures
After all experiments are finished, you can reproduce the figures in the paper using the following scripts.


### Experiment 1
Visualize the batch selection process based on the generated metadata

```bash
python experiments/plot_all_points.py
```

### Experiment 2

1. Generate the tabular data of the final performance based on the generated metadata

```bash
python experiments/result_excel.py
```

2. Compute the Average Normalized Score based on the generated metadata

```bash
python experiments/average_scores.py
```

3. Generate the regret plots based on the generated metadata

```bash
python experiments/plot_normalized_regret.py
```

### Hyperparameter sensitivity analysis experiments

The hyperparameter sensitivity analysis experiments can be executed via the Python scripts located in `examples/my_test_CBBO_parameters`, producing the corresponding result plots.




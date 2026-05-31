# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sys
import os
pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)


def summarize_one(method, dataset, q, num_seeds):
    """Return mean +- std string for one (method, dataset, q)."""
    values = []

    base_dir = os.path.join(
        result_root, dataset, method, f"q{q}"
    )

    for seed in range(num_seeds):
        file_path = os.path.join(base_dir, f"best_values_{seed}.xlsx")
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path)
        values.append(df.iloc[-1, 0])

    if len(values) == 0:
        return "N/A"

    values = np.array(values)
    mean = values.mean()
    std = values.std(ddof=1)

    return f"{mean:.4f} +- {std:.4f}"



# -----------------------------
# Experiment configuration
# -----------------------------

# Benchmark datasets
datasets_ackley =  ['ackley2', 'ackley10', 'ackley20', 'ackley50']
datasets_rastrigin = ['rastrigin2', 'rastrigin10', 'rastrigin20', 'rastrigin50']
datasets_rosenbrock = ['rosenbrock2', 'rosenbrock10', 'rosenbrock20', 'rosenbrock50']
datasets_styblinskitang = ['styblinskitang2', 'styblinskitang10', 'styblinskitang20', 'styblinskitang50']
datasets_powell = ['powell4', 'powell6', 'powell24', 'powell40']
dataset_hartmann6 =['hartmann6']
dataset_cosine8 = ['cosine8']
dataset_shekel = ['shekel4']


# datasets = (datasets_ackley + datasets_rastrigin + datasets_rosenbrock + datasets_styblinskitang + datasets_powell 
        #    + dataset_hartmann6 + dataset_cosine8 + dataset_shekel)

datasets = ['ackley2', 'ackley10']

methods_1 = [
    'EI-LP', 'EI-KB', 'EI-CL', 'BUCB', 'UCB-PE', 'UCB-LP'
]
methods_2 = [
'qLogEI', 'qEI','qUCB', 'qKG', 'qMES', 'GIBBON', 'BEEBO'
]
methods_3 = [
    'CBBO-LogEI', 'CBBO-EI', 'CBBO-UCB', 'CBBO-KG', 'CBBO-MES', 'CBBO-EE'
]

# Acquisition methods under comparison
# methods = methods_1 + methods_2 + methods_3

methods = ['CBBO-EI', 'CBBO-UCB']

num_seeds = 10
result_root = "result_hard_N_100" 

Q = [2, 5, 10, 20, 50]

save_dir = "experiments/final_results/hard_N_100"
os.makedirs(save_dir, exist_ok=True) 

for q in Q:
 
    output_file = f"experiments/final_results/hard_N_100/summary_q_{q}.xlsx"

    # -----------------------------
    # Build result table
    # -----------------------------
    table = pd.DataFrame(index=methods, columns=datasets)

    for method in methods:
        for dataset in datasets:
            table.loc[method, dataset] = summarize_one(
                method, dataset, q, num_seeds
            )
    table = table.T

    # -----------------------------
    # Save to Excel
    # -----------------------------
    table.index.name = "Method"
    table.to_excel(output_file)

    print(f"Saved summary table to {output_file}")



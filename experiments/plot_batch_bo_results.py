# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import sys
import os
pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)



# -----------------------------
# Experiment configuration
# -----------------------------

# Benchmark datasets
# datasets_ackley =  ['ackley2', 'ackley10', 'ackley20', 'ackley50']
# datasets_rastrigin = ['rastrigin2', 'rastrigin10', 'rastrigin20', 'rastrigin50']
# datasets_rosenbrock = ['rosenbrock2', 'rosenbrock10', 'rosenbrock20', 'rosenbrock50']
# datasets_styblinskitang = ['styblinskitang2', 'styblinskitang10', 'styblinskitang20', 'styblinskitang50']
# datasets_powell = ['powell4', 'powell6', 'powell24', 'powell40']
# dataset_hartmann6 =['hartmann6']
# dataset_cosine8 = ['cosine8']
# dataset_shekel = ['shekel4']


datasets_branin2 = ['branin2']
datasets_ackley = ['ackley2', 'ackley6', 'ackley10']
datasets_rastrigin = ['rastrigin2', 'rastrigin6', 'rastrigin10']
datasets_rosenbrock = ['rosenbrock2', 'rosenbrock6', 'rosenbrock10']
datasets_styblinskitang = ['styblinskitang2', 'styblinskitang6', 'styblinskitang10']

datasets_powell = ['powell4']
dataset_hartmann = ['hartmann3', 'hartmann6']
dataset_cosine8 = ['cosine8']
dataset_shekel = ['shekel4']

datasets = (datasets_branin2 + datasets_ackley + datasets_rastrigin + datasets_rosenbrock + datasets_styblinskitang + datasets_powell 
           + dataset_hartmann + dataset_cosine8 + dataset_shekel)


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
methods = methods_1 + methods_2 + methods_3

num_seeds = 10
result_root = "result_N_5000" 

Q = [2, 5, 10, 20, 50]

save_dir = "experiments/final_results/N_5000"
os.makedirs(save_dir, exist_ok=True)


METHOD_GROUP = {
    # EI family
    "EI-LP": "EI",
    "EI-KB": "EI",
    "EI-CL": "EI",
    "qEI": "EI",
    "CBBO-EI": "EI",
    # LogEI family
    "qLogEI": "LogEI",
    "CBBO-LogEI": "LogEI",

    # UCB family
    "BUCB": "UCB",
    "UCB-PE": "UCB",
    "UCB-LP": "UCB",
    "qUCB": "UCB",
    "CBBO-UCB": "UCB",

    # KG / MES / EE
    "qKG": "KG",
    "CBBO-KG": "KG",

    "qMES": "MES",
    "GIBBON": "MES",
    "CBBO-MES": "MES",

    "BEEBO": "EE",
    "CBBO-EE": "EE",
}

GROUP_COLOR = {
    "EI": "tab:blue",
    "LogEI": "tab:purple",
    "UCB": "tab:orange",
    "KG": "tab:green",
    "MES": "tab:red",
    "EE": "tab:pink",
}

def load_runs(path):
    """Load all seeds under a (dataset, method, q) directory"""
    runs = []
    for fname in sorted(os.listdir(path)):
        if fname.startswith("best_values_") and fname.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(path, fname))
            runs.append(df["best_y"].to_numpy())

    return np.array(runs)

def get_linestyle(method):
    if method.startswith("CBBO"):
        return "-"      # CBBO family
    elif method.startswith("q"):
        return ":"       # q-batch BO
    else:               # EI-KB, EI-CL, EI-LP, UCB-LP, UCB-PE, BUCB, GIBBON, BEEBO
        return "--"  

def get_marker(method):
    if method.startswith("CBBO"):
        return "D"      # CBBO family
    elif 'LP' in method:
        return "o"       # LP
    elif method == 'EI-KB':
        return "s"
    elif method == 'UCB-PE':
        return "*"
    elif method == 'GIBBON':
        return "P"
    elif method == 'BEEBO':
        return "X"
    else:
        return None  

def adaptive_marker_params(T):
    # 
    if T <= 10:
        markevery = 1
    elif T <= 20:
        markevery = 2
    elif T <= 35:
        markevery = 3
    else:  # T <= 50
        markevery = 4

    # 
    markersize = np.clip(10 - 0.08 * T, 5, 9)

    return markevery, markersize


def plot_dataset_q(dataset, q, result_root, methods):
    dataset_dir = os.path.join(result_root, dataset)

    plt.figure(figsize=(7, 5))

    for method in methods:
        q_dir = os.path.join(dataset_dir, method, f"q{q}")
        if not os.path.isdir(q_dir):
            continue

        data = load_runs(q_dir)
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        group = METHOD_GROUP.get(method, "OTHER")
        color = GROUP_COLOR[group]
        linestyle = get_linestyle(method)
        marker = get_marker(method)

        print('---mean:',mean)

        x = np.arange(len(mean))
        T = len(x)
        markevery, markersize = adaptive_marker_params(T)

        plt.plot(
            x,
            mean,
            label=method,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=markevery,
            markersize=markersize,
            linewidth=2,
        )
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Batch iteration")
    plt.ylabel("Best value so far")
    plt.title(f"{dataset} (q={q})")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = f"{save_dir}/{dataset}_q{q}.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================
# Main loop
# =============================

for dataset in datasets:
    for q in Q:
        plot_dataset_q(
            dataset=dataset,
            q=q,
            result_root=result_root,
            methods=methods,
        )

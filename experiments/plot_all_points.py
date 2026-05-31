# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from botorch.test_functions import Ackley

import sys
import os
from matplotlib.ticker import MaxNLocator
pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)



# -----------------------------
# Experiment configuration
# -----------------------------

# Benchmark datasets
datasets_ackley =  ['ackley2']

datasets = datasets_ackley

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

methods = methods_1 + methods_2 + methods_3

results_root_cbbo = "results_ccbo_liner_add2000" 
results_root_comparison = "results_comparison"

Q = [2, 5, 10, 20, 50]

save_dir = "experiments/final_results/cbbo_liner_add2000/plot_all_points"
os.makedirs(save_dir, exist_ok=True) 

# Ackley(2D)

f = Ackley(dim=2, negate=True)
bounds = f.bounds

n_grid = 101
x1 = torch.linspace(bounds[0, 0], bounds[1, 0], n_grid)
x2 = torch.linspace(bounds[0, 1], bounds[1, 1], n_grid)

X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
X_grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)

with torch.no_grad():
    Z = f(X_grid).reshape(n_grid, n_grid)
    print(Z.max())

plt.figure(figsize=(6, 5))

contour = plt.contourf(X1, X2, Z, levels=30)
cbar = plt.colorbar(contour)
cbar.ax.tick_params(labelsize=20)
cbar.locator = MaxNLocator(nbins=7)
cbar.update_ticks()

plt.xlabel("x1", fontsize=24)
plt.ylabel("x2", fontsize=24)

ax = plt.gca()  
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Ackley(2D)', fontsize=26)
plt.tight_layout()

save_path = f"{save_dir}/Ackley(2D).pdf"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()


def plot_runs(path, q, method, dataset):

    for fname in sorted(os.listdir(path)):
        if fname.startswith("records_") and fname.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(path, fname))
            seed = int(fname.replace(".xlsx", "").split("_")[-1])      

            if seed == 0:
                plt.figure(figsize=(6, 5))

                contour = plt.contourf(X1, X2, Z, levels=30)
                # cbar = plt.colorbar(contour)
                # cbar.ax.tick_params(labelsize=20)
                # cbar.locator = MaxNLocator(nbins=5)
                # cbar.update_ticks()

                # plt.xlabel("x1", fontsize=24)
                # plt.ylabel("x2", fontsize=24)
                if dataset == 'ackley2':
                    title_name = 'Ackley(2D)'
                plt.title(f"{title_name} (q={q}) - {method}", fontsize=26)

                x1_points_init = df.iloc[:4, 3].to_numpy()  
                x2_points_init = df.iloc[:4, 4].to_numpy()   

                plt.scatter(
                    x1_points_init,
                    x2_points_init,
                    marker="o",
                    color="white",
                    s=40,
                    edgecolors="black",
                    linewidths=0.5,
                    label="Initial points"
                )

                x1_points = df.iloc[4:, 3].to_numpy()  
                x2_points = df.iloc[4:, 4].to_numpy()   

                plt.scatter(
                    x1_points,
                    x2_points,
                    marker="x",
                    color="red",
                    s=40,
                    linewidths=1.0,
                    label="Selected points"
                )    

                # ax = plt.gca()  
                # ax.xaxis.set_major_locator(MaxNLocator(nbins=2))  
                # ax.yaxis.set_major_locator(MaxNLocator(nbins=2))  
                # plt.xticks(fontsize=20)
                # plt.yticks(fontsize=20)
                plt.xticks([])  
                plt.yticks([])  
                plt.gca().tick_params(bottom=False, left=False)                         

                if method in 'qLogEI':
                    plt.legend(fontsize=16, framealpha=0.5, edgecolor='none')

                plt.tight_layout()
                save_path = f"{save_dir}/{dataset}_q{q}_{method}_seed{seed}.pdf"
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"[Saved] {save_path}")



def plot_dataset_q(dataset, q, methods):

    for method in methods:
        if method in methods_3:
            dataset_dir = os.path.join(results_root_cbbo, dataset)
        else:
            dataset_dir = os.path.join(results_root_comparison, dataset)

        q_dir = os.path.join(dataset_dir, method, f"q{q}")
        if not os.path.isdir(q_dir):
            print(f"[Skip] {q_dir} not found")
            continue

        plot_runs(q_dir, q, method, dataset)



# =============================
# Main loop
# =============================

for dataset in datasets:
    for q in Q:
        plot_dataset_q(
            dataset=dataset,
            q=q,
            methods=methods,
        )

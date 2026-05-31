import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ========================================
# Configuration
# ========================================
ROOT_DIRS = ["results_comparison", "results_ccbo_liner_add2000"]
QS = ["q2", "q5", "q10", "q20", "q50"]

# Batch BO groups
GROUPS = { 
    "EI": ['qEI', 'EI-KB', 'EI-CL', 'EI-LP', 'CBBO-EI'],
    "UCB": ['BUCB', 'UCB-PE', 'UCB-LP', 'qUCB', 'CBBO-UCB'],
    "LogEI": ['qLogEI', 'CBBO-LogEI'],
    "KG": ['qKG', 'CBBO-KG'],
    "MES": ['qMES', 'GIBBON', 'CBBO-MES'], 
    "EE": ['BEEBO', 'CBBO-EE'],
}

# Datasets to include in the plot
datasets_plot = [
    'ackley2', 'rastrigin2', "levy4", 'ackley6', 'rastrigin6', "levy6",
    'cosine8', 'ackley10', 'rastrigin10', "levy10"
]

# CBBO methods highlighted in purple with circle markers
CBBO_METHODS = ['CBBO-EI', 'CBBO-UCB', 'CBBO-LogEI', 'CBBO-KG', 'CBBO-MES', 'CBBO-EE']

# ========================================
# Utility functions
# ========================================

def find_q_path(dataset, method, q):
    """Find the directory path for a given dataset, method, and batch size q."""
    for root in ROOT_DIRS:
        path = os.path.join(root, dataset, method, q)
        if os.path.exists(path):
            return path
    return None

def load_best_values(dataset, method, q):
    """Load best_y values from best_values_0..9.xlsx for all runs."""
    q_path = find_q_path(dataset, method, q)
    if q_path is None:
        return []

    runs = []
    for i in range(10):
        file_path = os.path.join(q_path, f"best_values_{i}.xlsx")
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            y = df['best_y'].to_numpy()
            runs.append(y)
    return runs

def compute_simple_regret(runs):
    """Compute simple regret for multiple runs (maximization problem)."""
    if len(runs) == 0:
        return None, None, None

    # Global best is the maximum across all runs at their last iteration
    global_best = np.max([run[-1] for run in runs])
    max_len = max(len(r) for r in runs)
    all_regret = []

    for run in runs:
        # Pad shorter runs with their last value
        if len(run) < max_len:
            padded = np.concatenate([run, np.full(max_len - len(run), run[-1])])
        else:
            padded = run
        regret = global_best - padded  
        all_regret.append(regret)

    all_regret = np.vstack(all_regret)
    mean_regret = np.mean(all_regret, axis=0)
    std_regret = np.std(all_regret, axis=0)
    return mean_regret, std_regret, max_len

# ========================================
# Plot function
# ========================================

def plot_dataset_group(q, dataset, group_name, methods):
    """Plot log simple regret for a single dataset, group, and batch size."""
    plt.figure(figsize=(6,4))
    ax = plt.gca()

    for method in methods:
        runs = load_best_values(dataset, method, q)
        mean_regret, std_regret, num_iters = compute_simple_regret(runs)
        if mean_regret is None:
            continue

        x = np.arange(num_iters)

        if method in CBBO_METHODS:
            ax.plot(x, np.log10(mean_regret + 1e-8), color='purple', label=method)
            # ax.fill_between(x,
            #                 np.log10(np.maximum(mean_regret - std_regret, 1e-8)),
            #                 np.log10(mean_regret + std_regret),
            #                 color='purple', alpha=0.2)
        else:
            ax.plot(x, np.log10(mean_regret + 1e-8), label=method)
            # ax.fill_between(x,
            #                 np.log10(np.maximum(mean_regret - std_regret, 1e-8)),
            #                 np.log10(mean_regret + std_regret),
            #                 alpha=0.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("log(Simple Regret)")
    ax.set_title(f"{dataset}, Group {group_name}, q={q}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(fontsize=8)
    plt.tight_layout()

    # Folder to save the figures
    FIGURE_DIR = "simple_regret_figures"
    os.makedirs(FIGURE_DIR, exist_ok=True)  # create folder if it doesn't exist

    # Inside your plot function, change the save path:
    out_file = os.path.join(FIGURE_DIR, f"simple_regret_{dataset}_{group_name}_{q}.pdf")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

# ========================================
# Main
# ========================================

def main():
    """Generate 50 plots: all datasets x all groups x all batch sizes."""
    for q in QS:
        for dataset in datasets_plot:
            for group_name, methods in GROUPS.items():
                plot_dataset_group(q, dataset, group_name, methods)

if __name__ == "__main__":
    main()
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

GROUPS = { 
    "EI": ['qEI', 'EI-KB', 'EI-CL', 'EI-LP', 'CBBO-EI'],
    "UCB": ['BUCB', 'UCB-PE', 'UCB-LP', 'qUCB', 'CBBO-UCB'],
    "LogEI": ['qLogEI', 'CBBO-LogEI'],
    "KG": ['qKG', 'CBBO-KG'],
    "MES": ['qMES', 'GIBBON', 'CBBO-MES'], 
    "EE": ['BEEBO', 'CBBO-EE'],
}

datasets_plot = [
    'ackley2', 'rastrigin2', "levy4", 'ackley6', 'rastrigin6', "levy6",
    'cosine8', 'ackley10', 'rastrigin10', "levy10"
]

CBBO_METHODS = ['CBBO-EI', 'CBBO-UCB', 'CBBO-LogEI', 'CBBO-KG', 'CBBO-MES', 'CBBO-EE']

FIGURE_DIR = "normalized_regret_figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ========================================
# Utilities
# ========================================

def find_q_path(dataset, method, q):
    for root in ROOT_DIRS:
        path = os.path.join(root, dataset, method, q)
        if os.path.exists(path):
            return path
    return None


def load_best_values(dataset, method, q):
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


# ========================================
# Core: Normalized Regret
# ========================================

def compute_normalized_regret(runs):
    """
    Compute normalized simple regret (maximization problem).

    Steps:
    1. Compute simple regret: f* - f_t
    2. Normalize within each dataset to [0,1]
    """

    if len(runs) == 0:
        return None, None, None

    # Global best across all runs
    global_best = np.max([run[-1] for run in runs])
    max_len = max(len(r) for r in runs)

    all_regret = []

    for run in runs:

        # Pad shorter runs
        if len(run) < max_len:
            padded = np.concatenate([run, np.full(max_len - len(run), run[-1])])
        else:
            padded = run

        # Step 1: simple regret (maximization)
        regret = global_best - padded

        # Step 2: normalize per run
        r_min = np.min(regret)
        r_max = np.max(regret)

        if r_max > r_min:
            regret = (regret - r_min) / (r_max - r_min)
        else:
            regret = np.zeros_like(regret)

        all_regret.append(regret)

    all_regret = np.vstack(all_regret)

    mean_regret = np.mean(all_regret, axis=0)
    std_regret = np.std(all_regret, axis=0)

    return mean_regret, std_regret, max_len


# ========================================
# Plot
# ========================================

def plot_dataset_group(q, dataset, group_name, methods):

    plt.figure(figsize=(6,4))
    ax = plt.gca()

    for method in methods:

        runs = load_best_values(dataset, method, q)
        mean_regret, std_regret, num_iters = compute_normalized_regret(runs)

        if mean_regret is None:
            continue

        x = np.arange(num_iters)

        # CBBO highlighted
        if method in CBBO_METHODS:
            ax.plot(x, mean_regret, 'o-', color='purple', label=method)
            ax.fill_between(x,
                            mean_regret - std_regret,
                            mean_regret + std_regret,
                            color='purple', alpha=0.2)
        else:
            ax.plot(x, mean_regret, label=method)
            ax.fill_between(x,
                            mean_regret - std_regret,
                            mean_regret + std_regret,
                            alpha=0.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Simple Regret")
    ax.set_title(f"{dataset}, {group_name}, q={q}")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(fontsize=8)

    plt.tight_layout()

    out_file = os.path.join(
        FIGURE_DIR,
        f"normalized_regret_{dataset}_{group_name}_{q}.pdf"
    )

    plt.savefig(out_file)
    plt.close()

    print(f"Saved {out_file}")


# ========================================
# Main
# ========================================

def main():

    for q in QS:
        print(f"\nProcessing {q}")

        for dataset in datasets_plot:
            for group_name, methods in GROUPS.items():
                plot_dataset_group(q, dataset, group_name, methods)


if __name__ == "__main__":
    main()
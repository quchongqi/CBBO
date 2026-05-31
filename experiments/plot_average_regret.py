import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ========================================
# Configuration
# ========================================

ROOT_DIRS = ["results_comparison", "results_ccbo_liner_add2000"]
QS = [2, 5, 10, 20, 50]

GROUPS = { 
    "EI": ['qEI', 'EI-KB', 'EI-CL', 'EI-LP', 'CBBO-EI'],
    "UCB": ['BUCB', 'UCB-PE', 'UCB-LP', 'qUCB', 'CBBO-UCB'],
    "LogEI": ['qLogEI', 'CBBO-LogEI'],
    "KG": ['qKG', 'CBBO-KG'],
    "MES": ['qMES', 'GIBBON', 'CBBO-MES'], 
    "EE": ['BEEBO', 'CBBO-EE'],
}

datasets_plot = [
    'ackley2', 'rastrigin2', "levy4", 
    'ackley6', 'rastrigin6', "levy6",
    'cosine8', 'ackley10', 'rastrigin10', "levy10"
]

# Known global optimum f*
FSTAR = {
    "ackley2": 0.0, "ackley6": 0.0, "ackley10": 0.0,
    "rastrigin2": 0.0, "rastrigin6": 0.0, "rastrigin10": 0.0,
    "levy4": 0.0, "levy6": 0.0, "levy10": 0.0,
    "cosine8": 0.8,
}

CBBO_METHODS = [
    'CBBO-EI', 'CBBO-UCB', 'CBBO-LogEI',
    'CBBO-KG', 'CBBO-MES', 'CBBO-EE'
]

# ========================================
# Utility functions
# ========================================

def find_q_path(dataset, method, q):
    q_star = f"q{q}"
    for root in ROOT_DIRS:
        path = os.path.join(root, dataset, method, q_star)
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
# Core: normalized regret (with init)
# ========================================

def compute_normalized_regret_with_init(runs, f_star):
    if len(runs) == 0:
        return None, None, None

    max_len = max(len(r) for r in runs)
    all_regret = []

    for run in runs:

        # pad to same length
        if len(run) < max_len:
            padded = np.concatenate([run, np.full(max_len - len(run), run[-1])])
        else:
            padded = run

        f_init = padded[0]

        # avoid division by zero
        if abs(f_star - f_init) < 1e-12:
            regret = np.zeros_like(padded)
        else:
            regret = (f_star - padded) / (f_star - f_init)

        all_regret.append(regret)

    all_regret = np.vstack(all_regret)

    mean_regret = np.mean(all_regret, axis=0)
    std_regret = np.std(all_regret, axis=0)

    return mean_regret, std_regret, max_len


# ========================================
# Plot (average across datasets)
# ========================================

def plot_group_avg(q, group_name, methods):

    plt.figure(figsize=(6,4))
    ax = plt.gca()

    method_curves = {m: [] for m in methods}

    # ===== loop over datasets =====
    for dataset in datasets_plot:

        if dataset not in FSTAR:
            continue

        f_star = FSTAR[dataset]

        for method in methods:

            runs = load_best_values(dataset, method, q)

            mean_regret, std_regret, num_iters = \
                compute_normalized_regret_with_init(runs, f_star)

            if mean_regret is None:
                continue

            method_curves[method].append(mean_regret)

    # ===== average across datasets =====
    for method in methods:

        curves = method_curves[method]
        if len(curves) == 0:
            continue

        # align length
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]

        curves = np.vstack(curves)

        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        x = np.arange(len(mean_curve))

        # ===== plot =====
        if method in CBBO_METHODS:
            ax.plot(x, np.log10(mean_curve + 1e-8), 
                    'o-', color='purple', markevery=max(1, len(x)//15), markersize=5, label=method)

            ax.fill_between(
                x,
                np.log10(np.maximum(mean_curve - 0.5*std_curve, 1e-8)),
                np.log10(mean_curve + 0.5*std_curve),
                color='purple', alpha=0.2
            )
        else:
            ax.plot(x, np.log10(mean_curve + 1e-8), label=method)

            ax.fill_between(
                x,
                np.log10(np.maximum(mean_curve - 0.5*std_curve, 1e-8)),
                np.log10(mean_curve + 0.5*std_curve),
                alpha=0.2
            )

    # ===== style =====
    ax.set_xlabel("Batch Iteration", fontsize=18)
    ax.set_ylabel(r"log$_{10}$(ANSR)", fontsize=18)
    ax.set_title(f"{group_name}-based Methods, q={q}", fontsize=20)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    # ===== save =====
    FIGURE_DIR = "average_regret_figures"
    os.makedirs(FIGURE_DIR, exist_ok=True)

    out_file = os.path.join(FIGURE_DIR, f"avg_regret_{group_name}_q{q}.pdf")
    plt.savefig(out_file)
    plt.close()

    print(f"Saved {out_file}")


# ========================================
# Main
# ========================================

def main():

    for q in QS:
        print(f"\nProcessing q{q}")

        for group_name, methods in GROUPS.items():
            plot_group_avg(q, group_name, methods)


if __name__ == "__main__":
    main()
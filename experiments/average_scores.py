import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Batch sizes
# ============================================================
QS = [2, 5, 10, 20, 50]

# ============================================================
# Method groups
# ============================================================
GROUPS = {
    "EI": ['qEI', 'EI-KB', 'EI-CL', 'EI-LP', 'CBBO-EI'],
    "UCB": ['BUCB', 'UCB-PE', 'UCB-LP', 'qUCB', 'CBBO-UCB'],
    "LogEI": ['qLogEI', 'CBBO-LogEI'],
    "KG": ['qKG', 'CBBO-KG'],
    "MES": ['qMES', 'GIBBON', 'CBBO-MES'],
    "EE": ['BEEBO', 'CBBO-EE'],

    # Global comparison
    "All_Methods": [
        'qEI', 'EI-KB', 'EI-CL', 'EI-LP', 'CBBO-EI',
        'BUCB', 'UCB-PE', 'UCB-LP', 'qUCB', 'CBBO-UCB',
        'qLogEI', 'CBBO-LogEI',
        'qKG', 'CBBO-KG',
        'qMES', 'GIBBON', 'CBBO-MES',
        'BEEBO', 'CBBO-EE'
    ],
}

# ============================================================
# Extract mean from "mean +- std"
# ============================================================
def extract_mean(val):
    if pd.isna(val):
        return np.nan
    return float(str(val).split("+-")[0].strip())


# ============================================================
# Compute Average Normalized Score (ANS)
# ============================================================
def compute_ans(df, methods):

    scores = {m: [] for m in methods}

    for dataset in df.index:

        values = []
        for m in methods:
            if m in df.columns:
                val = extract_mean(df.loc[dataset, m])
            else:
                val = np.nan
            values.append(val)

        values = np.array(values)

        # skip invalid dataset
        if np.all(np.isnan(values)):
            continue

        f_min = np.nanmin(values)
        f_max = np.nanmax(values)

        # normalization
        if f_max == f_min:
            normalized = np.ones_like(values)
        else:
            normalized = (values - f_min) / (f_max - f_min)

        # store
        for i, m in enumerate(methods):
            if not np.isnan(normalized[i]):
                scores[m].append(normalized[i])

    # average
    ans_dict = {}
    for m in methods:
        if len(scores[m]) > 0:
            ans_dict[m] = np.mean(scores[m])
        else:
            ans_dict[m] = np.nan

    return ans_dict


# ============================================================
# Plot clean ANS ranking (adaptive version)
# ============================================================
def plot_ans_clean(ans_dict, group_name, q):

    sorted_items = sorted(ans_dict.items(), key=lambda x: x[1], reverse=True)

    methods = [x[0] for x in sorted_items]
    scores = np.array([x[1] for x in sorted_items])

    n_methods = len(methods)

    # ====================================================
    # Dynamic layout
    # ====================================================
    is_all = group_name.lower() == "all_methods"

    width = 9 if not is_all else 12
    height = 3 if not is_all else max(3, 0.25 * n_methods)

    plt.figure(figsize=(width, height))

    if is_all:
        offsets = np.linspace(0.3, 0.3 * n_methods, n_methods)
        ylim_top = 0.3 * n_methods + 0.5
        fontsize = 10
    else:
        offsets = np.linspace(0.3, 1.2, n_methods)
        ylim_top = max(offsets) + 0.2
        fontsize = 16

    xmin = min(scores) - 0.02
    xmax = max(scores) + 0.02

    # ====================================================
    # Plot
    # ====================================================
    for i, (m, s) in enumerate(zip(methods, scores)):
        color = 'blue' if 'CBBO' in m else 'black'

        # point
        plt.plot(s, 0, 'o', color=color)

        # vertical line
        y_top = offsets[i]
        plt.vlines(s, 0, y_top, color=color, linewidth=1)

        # horizontal line
        x_text = s + 0.01
        plt.hlines(y_top, s, x_text, color=color, linewidth=1)

        # label
        plt.text(x_text, y_top, m, va='center', color=color, fontsize=fontsize)

    # ====================================================
    # Labels
    # ====================================================
    plt.xlabel("Average Normalized Score", fontsize=18)

    if is_all:
        plt.title(f"All Methods (q={q})", fontsize=20)
    else:
        plt.title(f"{group_name}-based Methods (q={q})", fontsize=20)

    plt.yticks([])
    plt.xlim(xmin, xmax)
    plt.ylim(-0.5, ylim_top)

    ax = plt.gca()

    # align x-axis to y=0
    ax.spines['bottom'].set_position(('data', 0))

    # remove other spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.margins(y=0)

    ax.spines['bottom'].set_linewidth(1.5)
    plt.tick_params(axis='x', labelsize=14)

    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"average_scores/ans_clean_{group_name}_q{q}.pdf")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():

    for q in QS:
        print(f"\nProcessing q={q}")

        file_path = f"summary_q{q}.xlsx"

        if not os.path.exists(file_path):
            print("File not found:", file_path)
            continue

        df = pd.read_excel(file_path, index_col=0)

        for group_name, methods in GROUPS.items():

            methods_in_df = [m for m in methods if m in df.columns]

            ans_dict = compute_ans(df, methods_in_df)

            print(f"\n{group_name} ANS:")
            for k, v in ans_dict.items():
                print(f"  {k}: {v:.4f}")

            plot_ans_clean(ans_dict, group_name, q)


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    main()
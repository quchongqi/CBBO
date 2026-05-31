import os
import pandas as pd
import numpy as np

# ============================================================
# Root directories containing experimental results
# ============================================================

ROOT_DIRS = [
    "results_comparison",
    "results_ccbo_liner_add2000",
]

# Batch sizes (q values) to summarize
QS = ["q2", "q5", "q10", "q20", "q50"]

# If True: report variance
# If False: report standard deviation
USE_VARIANCE = False

# ============================================================
# Desired order of methods in the final table
# ============================================================

METHOD_ORDER = [
    'qEI', 'EI-KB', 'EI-CL', 'EI-LP', 'CBBO-EI',
    'BUCB', 'UCB-PE', 'UCB-LP', 'qUCB', 'CBBO-UCB',
    'qLogEI', 'CBBO-LogEI',
    'qKG', 'CBBO-KG',
    'qMES', 'GIBBON', 'CBBO-MES',
    'BEEBO', 'CBBO-EE'
]

# ============================================================
# Manually selected datasets (custom subset for reporting)
# ============================================================

datasets_branin2 = ['branin2']
datasets_ackley = ['ackley2', 'ackley6', 'ackley10']
datasets_rastrigin = ['rastrigin2', 'rastrigin6', 'rastrigin10']
datasets_rosenbrock = ['rosenbrock2', 'rosenbrock6', 'rosenbrock10']
datasets_styblinskitang = ['styblinskitang2', 'styblinskitang6', 'styblinskitang10']

datasets_powell = ['powell4']
dataset_hartmann = ['hartmann3', 'hartmann6']
dataset_cosine8 = ['cosine8']
dataset_shekel = ['shekel4']

dataset_beale2 = ["beale2"]
dataset_bukin2 = ["bukin2"]
dataset_griewank2 = ["griewank2"]
dataset_michalewicz = ["michalewicz2", "michalewicz6", "michalewicz10"]

dataset_levy = ["levy4", "levy6", "levy10"]

# Final dataset list used in the table (custom selection)
datasets = (
    datasets_ackley + datasets_rastrigin + dataset_cosine8 + dataset_levy
)

datasets = [
'ackley2', 'rastrigin2', "levy4", 'ackley6', 'rastrigin6', "levy6", 'cosine8', 'ackley10', 'rastrigin10', "levy10"
]

# ============================================================
# Utility functions
# ============================================================

def extract_last_value(xlsx_path):
    """
    Read an Excel file and return the last value
    from the first column (final performance).
    """
    df = pd.read_excel(xlsx_path)
    return df.iloc[-1, 0]


def collect_methods():
    """
    Collect all available methods across ROOT_DIRS.

    Returns
    -------
    methods : list
        Sorted list of method names
    """
    methods = set()

    for root in ROOT_DIRS:
        if not os.path.exists(root):
            continue

        for dataset in os.listdir(root):
            dataset_path = os.path.join(root, dataset)

            if not os.path.isdir(dataset_path):
                continue

            for method in os.listdir(dataset_path):
                method_path = os.path.join(dataset_path, method)

                if os.path.isdir(method_path):
                    methods.add(method)

    return sorted(methods)


def filter_existing_datasets(datasets):
    """
    Filter out datasets that do not exist in ROOT_DIRS.

    This avoids producing rows with all NaN values.
    """
    valid = []

    for d in datasets:
        for root in ROOT_DIRS:
            if os.path.exists(os.path.join(root, d)):
                valid.append(d)
                break

    return valid


def find_q_path(dataset, method, q):
    """
    Locate the directory path:
        root/dataset/method/q

    Returns
    -------
    path or None
    """
    for root in ROOT_DIRS:
        q_path = os.path.join(root, dataset, method, q)

        if os.path.exists(q_path):
            return q_path

    return None


def summarize_one_q(q, datasets, methods):
    """
    Summarize results for a given batch size q.

    For each (dataset, method):
        - Load best_values_0.xlsx ... best_values_9.xlsx
        - Extract final values
        - Compute mean +- std (or variance)

    Returns
    -------
    df : pandas.DataFrame
        Table of results (dataset x method)
    """
    table = {}

    for dataset in datasets:
        table[dataset] = {}

        for method in methods:

            q_path = find_q_path(dataset, method, q)

            # If no results exist for this (dataset, method)
            if q_path is None:
                table[dataset][method] = np.nan
                continue

            values = []

            # Collect up to 10 runs
            for i in range(10):
                file_path = os.path.join(q_path, f"best_values_{i}.xlsx")

                if os.path.exists(file_path):
                    values.append(extract_last_value(file_path))

            # If no valid runs found
            if len(values) == 0:
                table[dataset][method] = np.nan
            else:
                mean = np.mean(values)

                if USE_VARIANCE:
                    var = np.var(values)
                    table[dataset][method] = f"{mean:.4f} +- {var:.4f}"
                else:
                    std = np.std(values)
                    table[dataset][method] = f"{mean:.4f} +- {std:.4f}"

    # Convert to DataFrame
    df = pd.DataFrame(table)

    # Enforce dataset order
    df = df.loc[methods]

    # Enforce method order
    df = df[datasets]

    return df


# ============================================================
# Main function
# ============================================================

def main():
    """
    Main pipeline:
        1. Collect methods automatically
        2. Use manually defined datasets
        3. Filter invalid datasets
        4. Reorder methods
        5. Generate summary tables for each q
    """

    # Collect methods only (datasets are manually defined)
    methods = collect_methods()

    # Filter out datasets that do not exist
    global datasets
    datasets = filter_existing_datasets(datasets)

    # Reorder methods according to predefined order
    ordered_methods = [m for m in METHOD_ORDER if m in methods]
    remaining_methods = [m for m in methods if m not in METHOD_ORDER]

    methods = ordered_methods + remaining_methods

    if remaining_methods:
        print("Warning: methods not in METHOD_ORDER:")
        print(remaining_methods)

    print("Datasets (filtered):", datasets)
    print("Methods:", methods)

    # Generate summary for each batch size
    for q in QS:
        df = summarize_one_q(q, datasets, methods)

        out_file = f"summary_{q}_T.xlsx"
        df.to_excel(out_file)

        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
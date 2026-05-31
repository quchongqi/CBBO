import multiprocessing as mp
import traceback
import time

import sys
import os
import re
import queue

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)


# ============================================================
# Experiment configuration
# ============================================================

result_path = 'results_ccbo_liner_add2000'

# Random seeds for repeated runs
seeds = list(range(10))

# Batch sizes (q) evaluated in batch Bayesian optimization
q_all = [2, 5, 10, 20, 50]

# Benchmark datasets
datasets_branin2 = ['branin2']
datasets_ackley = ['ackley2', 'ackley6', 'ackley10']
datasets_rastrigin = ['rastrigin2', 'rastrigin6', 'rastrigin10']
datasets_rosenbrock = ['rosenbrock2', 'rosenbrock6', 'rosenbrock10']
datasets_styblinskitang = ['styblinskitang2', 'styblinskitang6', 'styblinskitang10']

datasets_powell = ['powell4']
dataset_hartmann = ['hartmann3', 'hartmann6']
dataset_cosine8 = ['cosine8']
dataset_shekel = ['shekel4']

# add
dataset_beale2 = ["beale2"]
dataset_bukin2 = ["bukin2"]
dataset_griewank2 = ["griewank2"]
dataset_michalewicz = ["michalewicz2", "michalewicz6", "michalewicz10"]

# add -2
dataset_dropwave2 = ["dropwave2"]
dataset_dixonprice2 = ["dixonprice2"]
dataset_eggholder2 = ["eggholder2"]
dataset_holdertable2 = ["holdertable2"]
dataset_levy = ["levy4", "levy6", "levy10"]
dataset_sixhumpcamel2 = ["sixhumpcamel2"]
dataset_threehumpcamel2 = ["threehumpcamel2"]

# datasets = (datasets_branin2 + datasets_ackley + datasets_rastrigin + datasets_rosenbrock + datasets_styblinskitang + datasets_powell 
#            + dataset_hartmann + dataset_cosine8 + dataset_shekel) + dataset_beale2 + dataset_bukin2 + dataset_griewank2 + dataset_michalewicz


datasets = dataset_levy


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
methods = methods_3

# List of available GPUs (one GPU per worker process)

# Log files for completed and failed experiments
SUCCESS_FILE = "success_cbbo.txt"
ERROR_FILE = "error_cbbo.txt"
ERROR_FILE_LOG = "error_log_cbbo.txt"


# ============================================================
# Worker process definition
# Each worker is bound to a single GPU and continuously
# pulls tasks from the shared queue until it is empty.
# ============================================================
def worker(task_queue, num_threads=1):
    import torch
    from experiments.run_bo import run_bo

    # 
    torch.set_num_threads(num_threads)
    device = torch.device("cpu")

    while True:

        task = task_queue.get()

        if task is None:
            task_queue.task_done()
            break

        q, dataset, seed, method = task

        try:
            print(f"[CPU] Start: {method}, {dataset}, q={q}, seed={seed}")

            method_style = get_method_style(method)
            T, n_init= get_iterations(dataset, q)
            N = get_cbbo_N(method, dataset)

            run_bo(
                result_path=result_path,
                method=method,
                method_style=method_style,
                dataset=dataset,
                seed=seed,
                T=T,
                q=q,
                n_init=n_init,
                device=device,
                N=N,
            )

            with open(SUCCESS_FILE, "a") as f:
                f.write(f"{method}, {dataset}, q={q}, seed={seed}, device=cpu\n")

            print(f"[CPU] Done: {method}, {dataset}, q={q}, seed={seed}")

        except Exception:
            with open(ERROR_FILE, "a") as f:
                f.write(f"{method}, {dataset}, q={q}, seed={seed}, device=cpu\n")
            with open(ERROR_FILE_LOG, "a") as f:
                f.write(f"{method}, {dataset}, q={q}, seed={seed}, device=cpu\n")
                import traceback
                f.write(traceback.format_exc() + "\n")
            print(f"[CPU] Failed: {method}, {dataset}, q={q}, seed={seed}")
        
        finally:
            task_queue.task_done()


def get_method_style(method):
    style_1 = ['EI-KB', 'EI-CL', 'BUCB', 'UCB-PE']
    style_2 = ['qLogEI', 'qEI', 'qUCB','qKG', 'PPES', 'qMES', 'GIBBON', 'BEEBO']
    if 'LP' in method or method in style_1:
        method_style = 1
    elif method in style_2:
        method_style = 2
    elif 'CBBO' in method:
        method_style = 3
    else:
        raise ValueError(f"Unknow method:{method}")
    return method_style
     
def get_iterations(dataset, q):
    if q == 0:
        raise ValueError("q cann't be 0")
    name = dataset.lower()
    dim = int(re.findall(r"\d+", name)[0])
    n_init = max(4, dim)

    if dim < 8:
        q_budget = 100
    elif dim < 15:
        q_budget = 150
    elif dim < 25:
        q_budget = 200
    elif dim <= 50:
        q_budget = 300
    else:
        q_budget = 400

    T = int(q_budget/q)
    return T, n_init

def get_cbbo_N(method, dataset):
    if 'CBBO' in method:
        dim = int(re.findall(r"\d+", dataset)[0])
        N = 2000*dim + 2000
    else:
        N = None
    return N
    
def filter_methods_by_q(methods, q):
    
    # unstable_methods = ['PPES', 'qKG', 'qMES', 'qLogEI', 'qEI', 'qUCB', 'GIBBON']
    # if q > 4:
    #     return [m for m in methods if m not in unstable_methods]
    # else:
    #     return methods
    return methods


def load_success_set(success_file):
    """
    Load successful experiments into a set for fast lookup
    Key format: (method, dataset, q, seed)
    """
    success_set = set()

    if not os.path.exists(success_file):
        return success_set

    with open(success_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # EI-LP, ackley2, q=2, seed=0, gpu=1
            parts = [p.strip() for p in line.split(",")]
            method = parts[0]
            dataset = parts[1]
            q = int(parts[2].split("=")[1])
            seed = int(parts[3].split("=")[1])

            success_set.add((method, dataset, q, seed))

    return success_set


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    open(ERROR_FILE, "w").close()
    open(ERROR_FILE_LOG, "w").close()

    task_queue = mp.JoinableQueue()
    success_set = load_success_set(SUCCESS_FILE)

    for q in q_all:
        methods_q = filter_methods_by_q(methods, q)
        for dataset in datasets:
            for seed in seeds:
                for method in methods_q:
                    key = (method, dataset, q, seed)
                    if key in success_set:
                        continue
                    task_queue.put((q, dataset, seed, method))

    num_workers = 16  
    threads_per_worker = 6

    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(task_queue, threads_per_worker))
        p.start()
        processes.append(p)

    # wait tasks finished
    task_queue.join()

    # stop workers
    for _ in range(num_workers):
        task_queue.put(None)
        
    # wait processes exit
    for p in processes:
        p.join()

    print("All CPU experiments finished.")


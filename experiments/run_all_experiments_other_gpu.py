import multiprocessing as mp
import traceback
import time

import sys
import os
import re
pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)


# ============================================================
# Experiment configuration
# ============================================================

result_path = 'results_comparison'

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
methods = methods_1 + methods_2

# List of available GPUs (one GPU per worker process)
GPU_LIST = list(range(7))  # cuda:0 ~ cuda:6

# Log files for completed and failed experiments
SUCCESS_FILE = "success_comparison.txt"
ERROR_FILE = "error_comparison.txt"
ERROR_FILE_LOG = "error_log_comparison.txt"


# ============================================================
# Worker process definition
# Each worker is bound to a single GPU and continuously
# pulls tasks from the shared queue until it is empty.
# ============================================================
def worker(gpu_id, task_queue):

    # Bind this worker to a specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from experiments.run_bo import run_bo
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    while True:
        try:
            # Fetch a task from the queue (non-blocking)
            task = task_queue.get_nowait()
        except Exception:
            # Exit when no tasks remain
            break

        q, dataset, seed, method = task

        try:
            print(f"[{gpu_id}] Start: {method}, {dataset}, q={q}, seed={seed}")

            # Run one Bayesian optimization experiment
            method_style = get_method_style(method)
            T, n_init= get_iterations(dataset, q)
            N = get_cbbo_N(method, dataset)
            run_bo(
                result_path=result_path,
                method=method,
                method_style=method_style,     # optimize_acqf-based implementation
                dataset=dataset,
                seed=seed,
                T=T,                # number of BO iterations
                q=q,                # batch size
                n_init=n_init,           # initial design size
                device=device,      # GPU assigned to this worker
                N = N,
            )

            # Log successful runs
            with open(SUCCESS_FILE, "a") as f:
                f.write(
                    f"{method}, {dataset}, q={q}, seed={seed}, gpu={gpu_id}\n"
                )

            print(f"[{gpu_id}] Done")

        except Exception:
            # Log failed runs along with the full traceback

            with open(ERROR_FILE, "a") as f:
                f.write(
                    f"{method}, {dataset}, q={q}, seed={seed}, gpu={gpu_id}\n"
                )

            with open(ERROR_FILE_LOG, "a") as f:
                f.write(
                    f"{method}, {dataset}, q={q}, seed={seed}, gpu={gpu_id}\n"
                )
                f.write(traceback.format_exc() + "\n")

            print(f"[{gpu_id}] Failed")

        finally:
            # Optional cooldown to reduce GPU resource contention
            time.sleep(1)


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
    # Use spawn to ensure safe multiprocessing with CUDA
    mp.set_start_method("spawn", force=True)

    # Clear log files before launching workers
    open(ERROR_FILE, "w").close()
    open(ERROR_FILE_LOG, "w").close()

    # Shared task queue
    task_queue = mp.Queue()
    # success file
    success_set = load_success_set(SUCCESS_FILE)
    print(f"Loaded {len(success_set)} successful experiments.")

    # Enumerate all experimental configurations
    for q in q_all:
        methods = filter_methods_by_q(methods, q)
        for dataset in datasets:
            for seed in seeds:
                for method in methods:
                    key = (method, dataset, q, seed)
                    if key in success_set:
                        print(f"Skip finished: {key}")
                        continue

                    task_queue.put((q, dataset, seed, method))

    processes = []

    # Launch one worker per GPU
    for gpu_id in GPU_LIST:
        p = mp.Process(
            target=worker,
            args=(gpu_id, task_queue),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print("All experiments finished.")

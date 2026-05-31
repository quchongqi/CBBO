
import sys
import os
import torch
import matplotlib.pyplot as plt

pwd_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(pwd_dir))
print(project_root)
sys.path.append(project_root)

from cbbo.cbbo import coverage_batch_bo_step
from functools import partial


def toy_acquisition_1d_unimodal_1(X):
    """
    X: (N, 1)
    return: (N,)
    """
    mu = 0.6
    sigma = 0.08
    return torch.exp(-0.5 * ((X.squeeze() - mu) / sigma) ** 2)

def toy_acquisition_1d_unimodal_10(X):
    """
    X: (N, 1)
    return: (N,)
    """
    mu = 0.6
    sigma = 0.08
    return torch.exp(-0.5 * ((X.squeeze()/10 - mu) / sigma) ** 2)

def toy_acquisition_1d_unimodal_100(X):
    """
    X: (N, 1)
    return: (N,)
    """
    mu = 0.6
    sigma = 0.08
    return torch.exp(-0.5 * ((X.squeeze()/100 - mu) / sigma) ** 2)

def toy_acquisition_1d_unimodal_1000(X):
    """
    X: (N, 1)
    return: (N,)
    """
    mu = 0.6
    sigma = 0.08
    return torch.exp(-0.5 * ((X.squeeze()/1000 - mu) / sigma) ** 2)


Bounds = {
    '1': [[0.0, 1.0]],  
    '10': [[0.0, 10]],
    '100': [[0.0, 100]],
    '1000': [[0.0, 1000]],
}

Toy_acquisition = {
    '1': toy_acquisition_1d_unimodal_1,
    '10': toy_acquisition_1d_unimodal_10,
    '100': toy_acquisition_1d_unimodal_100,
    '1000': toy_acquisition_1d_unimodal_1000,
}


q = 5
# eta_ls = ['None', 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 10, 100]

eta_ls = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for eta_l in eta_ls:
    for name, bounds in Bounds.items():
        toy_acquisition = Toy_acquisition[name]
        batch_points, X, w, _= coverage_batch_bo_step(
            acqf_name=toy_acquisition,
            bounds=bounds,
            q=q,
            discretizer_methods='hypercube',
            N=4000,
            eta_l=eta_l,
            eta_tau=0.6,
        )

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # left axix: acquisition
        ax1.plot(X, toy_acquisition(X), label=r"$\alpha(x)$", linewidth=2, color='C0')
        ax1.scatter(batch_points,
                    toy_acquisition(batch_points),
                    color="red",
                    s=80,
                    zorder=5,
                    label="Batch points")
        ax1.set_xlabel("x", fontsize=18)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_ylabel(r"$\alpha(x)$", color='C0', fontsize=18)
        ax1.tick_params(axis='y', labelcolor='C0', labelsize=14)

        # rigjt axix: softmax weights
        ax2 = ax1.twinx()
        ax2.plot(X, w, label="w(x)",
                linestyle='--', color='C1')
        ax2.set_ylabel("w(x)", color='C1', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='C1', labelsize=14)

        #  legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=16)
        plt.grid(alpha=0.3)

        if eta_l == 'None':
            plt.title(rf" $\ell_i = 1  \quad (L_i =${bounds[0][1] - bounds[0][0]})", fontsize=20)
        else:
            plt.title(rf" $\ell_i = {eta_l} \cdot L_i \quad ( L_i =${bounds[0][1] - bounds[0][0]})", fontsize=20)
        plt.tight_layout()
        if eta_l == 'None':
            plt.savefig(f'figures/my_test_CBBO/1D_adaptive_kernel/fixed_kernel_{name}_{eta_l}.pdf')
        else:
            plt.savefig(f'figures/my_test_CBBO/1D_adaptive_kernel/adaptive_kernel_{name}_{eta_l}.pdf')
        
        plt.close()

        


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


def toy_acquisition_1d_unimodal(X, sigma):
    """
    X: (N, 1)
    return: (N,)
    """
    mu = 0.6
    sigma = sigma
    return torch.exp(-0.5 * ((X.squeeze() - mu) / sigma) ** 2)


def toy_acquisition_1d_multimodal(X, sigmas, weights):
    """
    X: (N, 2)
    return: (N,)
    """
    X = X.squeeze(-1)
    mus = torch.tensor([0.25, 0.55, 0.85])
    # sigmas = torch.tensor([0.04, 0.06, 0.05])
    # weights = torch.tensor([1.0, 0.6, 0.3])
    sigmas = sigmas
    weights = weights

    values = torch.zeros_like(X)
    for w, mu, sigma in zip(weights, mus, sigmas):
        values += w * torch.exp(-0.5 * ((X - mu) / sigma) ** 2)

    return values


bounds = [
    [0.0, 1.0],  # x
]

q = 5

sigmas_small = torch.tensor([0.02, 0.02, 0.02])
sigmas_medium = torch.tensor([0.05, 0.05, 0.05])
sigmas_big = torch.tensor([0.08, 0.08, 0.08])
sigmas_mix = torch.tensor([0.01, 0.1, 0.05])

weights_1 = [1, 1.2, 0.8]
weights_2 = [1, 0.6, 0.1]

Toy_acquisition = {
    'Unimodal_small_variance': partial(
        toy_acquisition_1d_unimodal,
        sigma=torch.tensor([0.02])
    ),
    'Unimodal_medium_variance': partial(
        toy_acquisition_1d_unimodal,
        sigma=torch.tensor([0.08])
    ),
    'Unimodal_big_variance': partial(
        toy_acquisition_1d_unimodal,
        sigma=torch.tensor([0.2])
    ),
    'Multimodal_small_variance': partial(
        toy_acquisition_1d_multimodal,
        sigmas=sigmas_small,
        weights=weights_1
    ),
    'Multimodal_medium_variance': partial(
        toy_acquisition_1d_multimodal,
        sigmas=sigmas_medium,
        weights=weights_1
    ),
    'Multimodal_big_variance': partial(
        toy_acquisition_1d_multimodal,
        sigmas=sigmas_big,
        weights=weights_1
    ),
    'Multimodal_mix_variance': partial(
        toy_acquisition_1d_multimodal,
        sigmas=sigmas_mix,
        weights=weights_2
    ),
}

eta_taus = ['None', 0.01, 0.2, 0.4, 0.6, 0.8, 1, 10, 100]
# eta_taus = ['None', 0.6]

for eta_tau in eta_taus:

    for name, toy_acquisition in Toy_acquisition.items():

        toy_acquisition = toy_acquisition

        batch_points, X, w, _ = coverage_batch_bo_step(
            acqf_name=toy_acquisition,  
            bounds=bounds,
            q=q,
            discretizer_methods='hypercube',
            N=4000,        
            eta_l=0.1,
            eta_tau=eta_tau,
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
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=14)

        if eta_tau == 'None':
            plt.title(rf" $\tau_t = 1:${name}", fontsize=20)
        else:
            plt.title(rf" $\tau_t = {eta_tau} \cdot  \operatorname{{std}}(\alpha_t):${name}", fontsize=20)
        plt.tight_layout()
        if eta_tau == 'None':
            plt.savefig(f'figures/my_test_CBBO/1D_adaptive_softmax/fixed_softmax_{name}_{eta_tau}.pdf')
        else:
            plt.savefig(f'figures/my_test_CBBO/1D_adaptive_softmax/adaptive_softmax_{name}_{eta_tau}.pdf')

        plt.close()






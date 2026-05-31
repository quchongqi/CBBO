
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


def toy_acquisition_2d_unimodal(X, sigma):
    """
    X: (N, 2)
    return: (N,)
    """
    mu = torch.tensor([0.6, 0.4])
    # sigma = torch.tensor([0.08, 0.12])
    sigma = sigma

    diff = (X - mu) / sigma
    return torch.exp(-0.5 * (diff ** 2).sum(dim=-1))


def toy_acquisition_2d_multimodal(X, sigmas, weights):
    """
    X: (N, 2)
    return: (N,)
    """
    mus = torch.tensor([
        [0.25, 0.25],
        [0.75, 0.30],
        [0.50, 0.75],
    ])

    # sigmas = torch.tensor([
    #     [0.06, 0.06],
    #     [0.08, 0.05],
    #     [0.05, 0.09],
    # ])
    # weights = torch.tensor([0.4, 0.35, 0.25])
    sigmas = sigmas
    weights = weights

    vals = []
    for mu, sigma, w in zip(mus, sigmas, weights):
        diff = (X - mu) / sigma
        vals.append(w * torch.exp(-0.5 * (diff ** 2).sum(dim=-1)))

    return torch.stack(vals, dim=0).sum(dim=0)


bounds = [
    [0.0, 0.0],  
    [1.0, 1.0],  
]

q = 5

sigmas_small = torch.tensor([
        [0.04, 0.06],
        [0.04, 0.06],
        [0.04, 0.06],
    ])

sigmas_medium = torch.tensor([
        [0.10, 0.12],
        [0.10, 0.12],
        [0.10, 0.12],
    ])

sigmas_big = torch.tensor([
        [0.20, 0.25],
        [0.20, 0.25],
        [0.20, 0.25],
    ])

sigmas_mix = torch.tensor([
        [0.04, 0.06],
        [0.40, 0.45],
        [0.10, 0.12],
    ])
weights_1 = [1, 1.2, 0.8]
weights_2 = [1, 1, 0.4]

Toy_acquisition = {
    'Unimodal_small_variance': partial(
        toy_acquisition_2d_unimodal,
        sigma=torch.tensor([0.04, 0.06])
    ),
    'Unimodal_medium_variance': partial(
        toy_acquisition_2d_unimodal,
        sigma=torch.tensor([0.10, 0.12])
    ),
    'Unimodal_big_variance': partial(
        toy_acquisition_2d_unimodal,
        sigma=torch.tensor([0.40, 0.45])
    ),
    'Multimodal_small_variance': partial(
        toy_acquisition_2d_multimodal,
        sigmas=sigmas_small,
        weights=weights_1
    ),
    'Multimodal_medium_variance': partial(
        toy_acquisition_2d_multimodal,
        sigmas=sigmas_medium,
        weights=weights_1
    ),
    'Multimodal_big_variance': partial(
        toy_acquisition_2d_multimodal,
        sigmas=sigmas_big,
        weights=weights_1
    ),
    'Multimodal_mix_variance': partial(
        toy_acquisition_2d_multimodal,
        sigmas=sigmas_mix,
        weights=weights_2
    ),
}

eta_taus = ['None', 0.01, 0.2, 0.4, 0.6, 0.8, 1, 10, 100]
# eta_taus = [0.6]

for eta_tau in eta_taus:

    for name, toy_acquisition in Toy_acquisition.items():

        toy_acquisition = toy_acquisition

        batch_points, X, w, _ = coverage_batch_bo_step(
            acqf_name=toy_acquisition,  
            bounds=bounds,
            q=q,
            discretizer_methods='hypercube',
            N=6000,        
            eta_l=0.1,
            eta_tau=eta_tau,
        )

        # grid for visualization
        grid_size = 100
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        Xg, Yg = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=-1)

        Z = toy_acquisition(XY).reshape(grid_size, grid_size)

        plt.figure(figsize=(8, 6))

        # contour
        plt.contourf(Xg, Yg, Z, levels=50, cmap="viridis")
        cbar = plt.colorbar()
        cbar.set_label(r"$\alpha(x)$", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # selected batch points
        plt.scatter(
            batch_points[:, 0],
            batch_points[:, 1],
            c="red",
            s=80,
            edgecolors="white",
            label="Batch points" 
        )

        plt.xlabel("x1", fontsize=18)
        plt.ylabel("x2", fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        if eta_tau == 'None':
            plt.title(rf" $\tau_t = 1:${name}", fontsize=20)
        else:
            plt.title(rf" $\tau_t = {eta_tau} \cdot  \operatorname{{std}}(\alpha_t):${name}", fontsize=20)
        plt.legend()
        plt.tight_layout()
        if eta_tau == 'None':
            plt.savefig(f'figures/my_test_CBBO/2D_adaptive_softmax/fixed_softmax_{name}_{eta_tau}.pdf')
        else:
            plt.savefig(f'figures/my_test_CBBO/2D_adaptive_softmax/adaptive_softmax_{name}_{eta_tau}.pdf')
        
        plt.close()





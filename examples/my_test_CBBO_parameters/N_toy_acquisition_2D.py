
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

def toy_acquisition_2d_unimodal_1(X, sigma):
    """
    X: (N, 2)
    return: (N,)
    """
    mu = torch.tensor([0.6, 0.4])
    sigma = sigma

    diff = (X - mu) / sigma
    return 5*torch.exp(-0.5 * (diff ** 2).sum(dim=-1))

def toy_acquisition_2d_unimodal_10(X, sigma):
    """
    X: (N, 2)
    return: (N,)
    """
    mu = torch.tensor([0.6, 0.4])
    sigma = sigma

    diff = (X/10 - mu) / sigma
    return 5*torch.exp(-0.5 * (diff ** 2).sum(dim=-1))

def toy_acquisition_2d_unimodal_100(X, sigma):
    """
    X: (N, 2)
    return: (N,)
    """
    mu = torch.tensor([0.6, 0.4])
    sigma = sigma

    diff = (X/100 - mu) / sigma
    return 5*torch.exp(-0.5 * (diff ** 2).sum(dim=-1))

def toy_acquisition_2d_unimodal_1000(X, sigma):
    """
    X: (N, 2)
    return: (N,)
    """
    mu = torch.tensor([0.6, 0.4])
    sigma = sigma

    diff = (X/1000 - mu) / sigma
    return 5*torch.exp(-0.5 * (diff ** 2).sum(dim=-1))


# Bounds = {
#     '1': [(0.0, 0.0), (1.0, 1.0)],  
#     '10': [(0.0, 0.0), (10, 10)],
#     '100': [(0.0, 0.0), (100, 100)],
#     '1000': [(0.0, 0.0), (1000, 1000)],
# }

Bounds = {
    '1': [(0.0, 0.0), (1.0, 1.0)],  
}


sigma = torch.tensor([0.2, 0.4])
# Toy_acquisition = {
#     '1': partial(
#         toy_acquisition_2d_unimodal_1,
#         sigma=sigma,
#     ),
#     '10': partial(
#         toy_acquisition_2d_unimodal_10,
#         sigma=sigma,
#     ),
#     '100': partial(
#         toy_acquisition_2d_unimodal_100,
#         sigma=sigma,
#     ),
#     '1000': partial(
#         toy_acquisition_2d_unimodal_1000,
#         sigma=sigma,
#     ),
# }

Toy_acquisition = {
    '1': partial(
        toy_acquisition_2d_unimodal_1,
        sigma=sigma,
    ),
}

q = 5

Ns = [20, 200, 2000, 20000]

for N in Ns:
    for name, bounds in Bounds.items():
        toy_acquisition = Toy_acquisition[name]
        batch_points, X, w, _= coverage_batch_bo_step(
            acqf_name=toy_acquisition,
            bounds=bounds,
            q=q,
            discretizer_methods='hypercube',
            N=N,
            eta_l=0.01,
            eta_tau=0.6,
        )
   
        # grid for visualization
        grid_size = 100
        (x_low, y_low), (x_high, y_high) = bounds
        x = torch.linspace(x_low, x_high, grid_size)
        y = torch.linspace(y_low, y_high, grid_size)
        Xg, Yg = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=-1)

        Z = toy_acquisition(XY).reshape(grid_size, grid_size)

        plt.figure(figsize=(6, 5))

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
        

        plt.title(f" N={N}", fontsize=20)
        plt.tight_layout()
        plt.savefig(f'figures/my_test_CBBO/2D_N/N_{name}_{N}.pdf')

        plt.close()

        

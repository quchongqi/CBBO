import torch
import itertools

def discretizer(
    discretizer_method,
    bounds: torch.Tensor,
    N: int,
    seed: int = None,
    device=None,
    dtype=torch.double,
):
    if device is None:
        device = bounds.device

    bounds = bounds.to(device=device, dtype=dtype)

    if seed is not None:
        torch.manual_seed(seed)

    if discretizer_method == "hypercube":
        X = hypercube_discretization(bounds, N)

    elif discretizer_method == "random":
        X = random_discretization(bounds, N)

    elif discretizer_method == "lhs":
        X = lhs_discretization(bounds, N)

    elif discretizer_method == "sobol":
        X = sobol_discretization(bounds, N)

    else:
        raise ValueError(f"Unknown discretizer method: {discretizer_method}")

    return X



def hypercube_discretization(bounds: torch.Tensor, N: int):
    d = bounds.shape[0]
    dtype = bounds.dtype

    m = int(round(N ** (1.0 / d)))
    m = max(m, 2)

    grids = [
        torch.linspace(bounds[i, 0], bounds[i, 1], m, device=bounds.device, dtype=dtype)
        for i in range(d)
    ]

    X = torch.cartesian_prod(*grids)  # (m^d, d)
    
    if d == 1:
        X = X.unsqueeze(-1)   # (N,) ¡ú (N,1)

    return X

def random_discretization(bounds: torch.Tensor, N: int):
    d = bounds.shape[0]
    dtype = bounds.dtype

    u = torch.rand(N, d, device=bounds.device, dtype=dtyp)
    l = bounds[:, 0]
    r = bounds[:, 1] - bounds[:, 0]

    X = l + u * r
    return X

def lhs_discretization(bounds: torch.Tensor, N: int):
    d = bounds.shape[0]
    device = bounds.device
    dtype = bounds.dtype

    # Step 1: [0,1] ÉÏµÄ LHS
    u = torch.rand(N, d, device=device, dtype=dtyp)
    perm = torch.stack([torch.randperm(N, device=device) for _ in range(d)], dim=1)

    X_unit = (perm + u) / N  # (N, d)

    # Step 2: scale to bounds
    l = bounds[:, 0]
    r = bounds[:, 1] - bounds[:, 0]

    X = l + X_unit * r
    return X

def sobol_discretization(bounds: torch.Tensor, N: int):
    d = bounds.shape[0]
    device = bounds.device
    dtype = bounds.dtype

    sobol = torch.quasirandom.SobolEngine(d, scramble=True)
    X_unit = sobol.draw(N).to(device=device, dtype=dtype)

    l = bounds[:, 0]
    r = bounds[:, 1] - bounds[:, 0]

    X = l + X_unit * r
    return X

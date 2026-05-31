import torch
from torch.distributions.normal import Normal
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement

# ===============================
# 1. Local penalization function ¦Õ(x; xj) (original formulation)
# ===============================
def local_penalty(
    X: torch.Tensor,          # (..., d)
    x_j: torch.Tensor,        # (1, d)
    mu_j: torch.Tensor,       # scalar
    sigma_j: torch.Tensor,    # scalar
    M: torch.Tensor,          # scalar
    L: float,
):
    """
    \alhpa(x; xj) = 0.5 * erfc(-z)
    """
    dist = torch.norm(X - x_j, dim=-1)

    z = (L * dist - M + mu_j) / (torch.sqrt(torch.tensor(2.0)) * sigma_j)
    phi = 0.5 * torch.erfc(-z)

    return phi


# ===============================
# 2. Local Penalized Acquisition Wrapper (compatible with any acquisition function)
# ===============================
class LocalPenalizedAcquisition(AcquisitionFunction):
    def __init__(self, base_acqf, model, selected_points, L: float, M: float):
        super().__init__(model=model)
        self.base_acqf = base_acqf
        self.L = L
        self.M = torch.tensor(M, dtype=selected_points.dtype, device=selected_points.device)

        # Precompute posterior mean and std for selected points
        mus = []
        sigmas = []
        for xj in selected_points:
            posterior = model.posterior(xj.unsqueeze(0))
            mu_j = posterior.mean.squeeze()
            sigma_j = posterior.variance.sqrt().clamp_min(1e-9)
            mus.append(mu_j)
            sigmas.append(sigma_j)
        self.mus = torch.stack(mus)       # [num_selected]
        self.sigmas = torch.stack(sigmas) # [num_selected]
        self.selected_points = selected_points  # [num_selected, d]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, d]
        base_value = self.base_acqf(X)     
        base_value = base_value.squeeze(0) 

        penalty = torch.ones_like(base_value)  
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=X.device))

        for xj, mu_j, sigma_j in zip(self.selected_points, self.mus, self.sigmas):
            dist = torch.norm(X - xj.view(1, -1), dim=-1)  
            z = (self.L * dist - self.M + mu_j) / (sigma_j * sqrt2)
            phi = 0.5 * torch.erfc(-z)     
            phi = phi.squeeze()                
            penalty = penalty * phi                        

        return base_value * penalty         



# ===============================
# 3. Sequential batch optimization with Local Penalization
# ===============================
def optimize_acqf_with_lp(
    model,
    base_acqf,
    bounds,
    q,
    L=10.0,
    num_restarts=10,
    raw_samples=128,
):
    selected_points = []

    # M = max_x ¦Ì(x), approximated using posterior means at training points
    with torch.no_grad():
        posterior = model.posterior(model.train_inputs[0])
        M = posterior.mean.max().item()
    
    for i in range(q):
        if len(selected_points) == 0:
            acqf = base_acqf
        else:
            acqf = LocalPenalizedAcquisition(
                base_acqf=base_acqf,
                model=model,
                selected_points=torch.stack(selected_points),
                L=L,
                M=M,
            )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        candidate = candidate.squeeze(0)
        
        selected_points.append(candidate)

    return torch.stack(selected_points)

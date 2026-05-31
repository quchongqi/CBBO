
#-------------Coverage-baesd Batch Bayesian Optimzation-----------
#  Edit Schedule: --- > 2025.12.30
#                 ---->
import torch
from .discretization import discretizer
from .acquisition_utils import compute_acquisition
from .acquisition_utils import softmax_normalize
from .coverage_kernels import adaptive_coverage_kernel
from .coverage_selectors import greedy_coverage_selection


def coverage_batch_bo_step(
                bounds, 
                q, 
                model=None,
                discretizer_methods='hypercube', 
                N=1000, 
                eta_l=0.1,
                eta_tau=0.6,
                D_t = None,
                acqf_name='EI',  
                device=None,
                **kwargs):
    """Main code of the CCBO method.
    Args:
        ---model: typically, a Gaussian Process (GP) is used; for testing purposes, a toy example can also be employed.
        ---acqf_name: acquisition function used for batch operations
        ---bounds: the range of the search space
        ---discretizer_methods: bounds discretization methods, e.g., hypercube-based discretization
        ---q: the number of query points in Batch BO
        ---N: the number of discretization points.
        ---eta_l: parameter used for the adaptive coverage kernel (l_i= eta_l* L_i)
        ---eta_tau: parameter used for the acquisition function normalization (softmax operation)
        ---D_t: the index set of previously evaluated points at iteration t
        ---device: run on either the CPU or the GPU
        ---**kwargs: additional parameters required by different methods
    Returns:
        q candidate points 
    """
    bounds = normalize_bounds(bounds)

    # === Discretization step (explicit!)
    X = discretizer(discretizer_methods, bounds, N)

    # === Acquisition
    alpha = compute_acquisition(model=model, acqf_name=acqf_name, X=X, **kwargs)
    # === Normalize
    if eta_tau == 'None':
        tau = torch.tensor(1)     
    else:   
        tau = eta_tau*alpha.std().clamp_min(1e-6)
    w = softmax_normalize(alpha, tau)

    # === Adaptive kernel
    kernel_fn = adaptive_coverage_kernel(X, bounds, eta_l) 

    # === Coverage greedy
    batch, D_t_new = greedy_coverage_selection(X, w, kernel_fn, q, D_t, device)

    return batch, X, w, D_t_new


def normalize_bounds(bounds):
    """
    Normalize bounds to shape (d, 2): [[l1, u1], ..., [ld, ud]]
    """
    bounds = torch.as_tensor(bounds)

    # Case 1: bounds is (2, d) -> transpose
    if bounds.ndim == 2 and bounds.shape[0] == 2:
        bounds = bounds.T

    # Final sanity check
    assert bounds.ndim == 2 and bounds.shape[1] == 2, \
        f"Invalid bounds shape: {bounds.shape}"

    return bounds

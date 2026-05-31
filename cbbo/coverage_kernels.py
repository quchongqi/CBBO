import torch

def adaptive_coverage_kernel(X, bounds, eta_l):
    """
    Returns a callable kernel function instead of an N x N matrix.
    Laplace (L1) adaptive coverage kernel.
    """
    bounds = normalize_bounds(bounds)
    if eta_l == 'None':
        ell = 1            # if there is no adaptive converage kernel lengthscale, ell = 1
    else:
        ell = eta_l * (bounds[1] - bounds[0])
    print(r'------{l_i}:', ell)
    def kernel_column(j):
        # X: (N, d)
        diff = (X - X[j]).abs() / ell
        return torch.exp(-diff.sum(dim=-1))  # (N,)

    return kernel_column

def normalize_bounds(bounds):
    """
    [(l1, u1), (l2, u2), (l3, u3)] to [(l1, l2, l3), (u1, u23, u3)]
    """
    bounds_tensor = bounds.T
    return bounds_tensor

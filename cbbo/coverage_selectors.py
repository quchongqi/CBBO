import torch

def greedy_coverage_selection(X, w, kernel_fn, q, D_t=None, device=None):
    """
    Greedy maximization of submodular coverage objective
    with hard exclusion of previously evaluated points.
    
    Args:
        X: candidate points, shape (N, d)
        w: normalized acquisition weights, shape (N,)
        kernel_fn: function kernel_fn(j) -> (N,)
        q: batch size
        D_t: indices of previously evaluated points
    """
    N = X.shape[0]

    if device is None:
        device = X.device
    
    if D_t is None:
        D_t = []
    
    # === mask for excluded (already evaluated) points
    excluded_mask = torch.zeros(N, dtype=torch.bool, device=device)

    if len(D_t) > 0:
        excluded_mask[D_t] = True

    covered = torch.zeros(N, device=device)

    selected = []
    selected_mask = excluded_mask.clone()   # ¡û already evaluated points are "pre-selected"

    for _ in range(q):
        gains = torch.full((N,), -float("inf"), device=device)

        for j in range(N):
            if selected_mask[j]:
                continue   

            Kj = kernel_fn(j)

            gains[j] = torch.sum(
                w * torch.clamp(Kj - covered, min=0.0)
            )


        s = torch.argmax(gains).item()
        selected.append(s)
        selected_mask[s] = True

        covered = torch.maximum(covered, kernel_fn(s))

    D_t = torch.where(selected_mask)[0]

    return X[selected], D_t



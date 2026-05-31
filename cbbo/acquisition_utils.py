import torch
from botorch.acquisition.analytic import (LogProbabilityOfImprovement, ProbabilityOfImprovement,
    ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound, PosteriorMean, PosteriorStandardDeviation,
    )
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

from beebo import BatchedEnergyEntropyBO


Acquisition_Function = {
    'logPI': LogProbabilityOfImprovement,
    'PI': ProbabilityOfImprovement,
    'EI': ExpectedImprovement,
    'LogEI': LogExpectedImprovement,
    'PM': PosteriorMean,
    'PSD': PosteriorStandardDeviation,
    'UCB': UpperConfidenceBound,
    'KG': qKnowledgeGradient,
    'PES': qPredictiveEntropySearch,
    'MES': qMaxValueEntropy,
    'EE': BatchedEnergyEntropyBO,
}

def compute_acquisition(model, acqf_name, X, **kwargs):
    """
    Compute acquisition function values for each candidate in X.

    Parameters
    ----------
    acquisition_function_name : str
        Name of the acquisition function.
    X : torch.Tensor, shape (N, d) or (q, N, d)
        Candidate points.
    kwargs : dict
        Arguments for the acquisition function constructor. For example:
        - EI / PI / logPI: model, best_f
        - UCB: model, beta
        - KG: model, num_fantasies, etc.

    Returns
    -------
    torch.Tensor
        Acquisition function values at X.
    """
    
    if model!=None:
        if acqf_name not in Acquisition_Function:
            raise ValueError(f"Unknown acquisition function: {acqf_name}")
        
        acq_fun_class = Acquisition_Function[acqf_name]
        acq_fun = acq_fun_class(model=model,
            **kwargs)  
        with torch.no_grad():
            cqf_values = acq_fun(X.unsqueeze(1))  
        return cqf_values.squeeze(-1)
    else:
        return acqf_name(X)



def softmax_normalize(alpha, tau, eps=1e-12):
    """
    Temperature-controlled softmax normalization.
    
    Args:
        alpha: acquisition values, shape (N,)
        tau: temperature (>0)
    Returns:
        w: normalized weights, sum to 1
    """
    tau = torch.clamp(tau, min=eps)

    z = alpha / tau
    z = z - z.max()              # numerical stability
    w = torch.exp(z)
    w = w / w.sum()

    return w